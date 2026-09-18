"""Profile old/repaired M8 on a private, fixed 60K Pi prefix without decoding it.

Separate clean wall-time passes from a short ROCm kernel-attribution pass.
All seven proposals are forced accepted: rates describe fixed work, not natural
Pi throughput. Raw tokens, logits and profiler traces stay in owned tmpfs.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

from benchmark_d7_equivalence import (
    EquivalenceProbe,
    EquivalenceWorkerExtension,
    private_root,
)
from d7_stage_attribution import PREFIX, attribute_trace, module_stage

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import (
    authenticate,
    digest,
    private_json,
    seal,
    write_private,
)


class StageScopes:
    def __init__(self, probe):
        import radiance_gdn

        self.probe = probe
        self.hooks = HookSet()
        self.inventory = []
        try:
            for name, module in probe.runner.model.named_modules():
                stage = "target_other" if not name else module_stage(name)
                if stage is not None:
                    self.wrap(module, "forward", stage, root=not name)
                    self.inventory.append({"module": name, "stage": stage})
            self.wrap(radiance_gdn, "conv_update", "gdn_convolution", native=True)
            self.wrap(radiance_gdn, "recurrent_update", "gdn_recurrence_gates_state", native=True)
        except BaseException:
            self.close()
            raise

    def wrap(self, owner, name, stage, *, root=False, native=False):
        original = getattr(owner, name)

        @functools.wraps(original)
        def call(*args, **kwargs):
            if native and not self.probe.in_target:
                return original(*args, **kwargs)
            previous = self.probe.in_target
            if root:
                self.probe.in_target = True
            try:
                with self.probe.scope(stage):
                    return original(*args, **kwargs)
            finally:
                self.probe.in_target = previous

        self.hooks.replace(owner, name, call)

    def close(self):
        self.hooks.close()


class StageProbe(EquivalenceProbe):
    def __init__(self, runner, task):
        super().__init__(runner, task)
        self.profiler = None
        self.profile_active = False
        self.scopes = None
        self.in_target = False
        self.profile_steps = 0
        self.final_logits = None

    def scope(self, name):
        if not self.profile_active:
            return contextlib.nullcontext()
        import torch

        return torch.profiler.record_function(PREFIX + name)

    def stop_profile(self):
        if self.profile_active:
            import torch

            torch.cuda.synchronize()
            self.profile_active = False
            self.profiler.stop()

    def attach(self):
        import torch

        if self.task.get("repair_manifest"):
            from stock_gdn_runtime import RuntimeRepairs

            self.repairs = RuntimeRepairs(
                self.task["repair_manifest"], target_model=self.runner.model
            )
        runner = self.runner
        original_prepare = runner.prepare_inputs
        original_logits = runner.model.compute_logits
        original_sample = runner.sample
        original_propose = runner.speculator.propose
        # The identical frozen successor/proposal values are loaded before timing.
        output_gpu = torch.tensor(self.schedule.output, device="cuda", dtype=torch.int64)

        def prepare(*args, **kwargs):
            if self.schedule.prefill_done and self.task["mode"] == "profile":
                start = self.task["warmup_steps"] * 8
                stop = start + self.task["profile_steps"] * 8
                if self.schedule.cursor == start:
                    torch.cuda.synchronize()
                    require(
                        torch.profiler.ProfilerActivity.CUDA
                        in torch.profiler.supported_activities(),
                        "ROCm GPU activity profiling is unavailable",
                    )
                    self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    )
                    self.profiler.start()
                    self.profile_active = True
                elif self.schedule.cursor == stop:
                    self.stop_profile()
            with self.scope("input_preparation"):
                batch = original_prepare(*args, **kwargs)
            with self.scope("forced_replay_control"):
                require(batch.num_reqs == 1, "unexpected concurrent request")
                positions = batch.positions[: batch.num_tokens].detach().cpu().tolist()
                inputs = batch.input_ids[: batch.num_tokens].detach().cpu().tolist()
                self.schedule.check_inputs(positions, inputs)
                self.input_hash.update(
                    bytes.fromhex(digest({"positions": positions, "inputs": inputs}))
                )
                self.input_batches += 1
            if self.profile_active:
                self.profile_steps += 1
            return batch

        def logits(*args, **kwargs):
            with self.scope("target_vocabulary_head"):
                result = original_logits(*args, **kwargs)
            if (
                self.schedule.prefill_done
                and self.schedule.cursor + 8 == len(self.schedule.output) - 1
            ):
                # Retain the final tensor; copy/hash only after the timed request.
                self.final_logits = result.detach()
            return result

        def sample(hidden_states, batch, grammar_output):
            require(grammar_output is None, "grammar transformations are not admitted")
            with self.scope("target_sampling"):
                result, ns, nr = original_sample(hidden_states, batch, grammar_output)
            with self.scope("forced_replay_control"):
                if not int(ns[0].item()):
                    return result, ns, nr
                step = self.schedule.commit(int(batch.num_draft_tokens))
                begin = 0 if step["prefill"] else step["start"] + 1
                result.sampled_token_ids.fill_(-1)
                result.sampled_token_ids[0, : step["count"]].copy_(
                    output_gpu[begin : begin + step["count"]]
                )
                ns.fill_(step["count"])
                nr.fill_(step["reject"])
            return result, ns, nr

        def propose(*args, **kwargs):
            with self.scope("drafter"):
                result = original_propose(*args, **kwargs)
            with self.scope("forced_replay_control"):
                require(tuple(result.shape) == (1, 7), "drafter width changed")
                begin = self.schedule.cursor + 1
                count = min(7, max(0, len(self.schedule.output) - begin))
                result.zero_()
                if count:
                    result[0, :count].copy_(output_gpu[begin : begin + count])
            return result

        self.hooks.replace(runner, "prepare_inputs", prepare)
        self.hooks.replace(runner.model, "compute_logits", logits)
        self.hooks.replace(runner, "sample", sample)
        self.hooks.replace(runner.speculator, "propose", propose)
        if self.task["mode"] == "profile":
            self.scopes = StageScopes(self)

    def finish(self):
        import torch

        require(self.schedule.done, "incomplete forced replay")
        self.stop_profile()
        torch.cuda.synchronize()
        require(
            self.final_logits is not None and self.final_logits.shape == (8, 248320),
            "missing final M8 logits",
        )
        values = self.final_logits.float().cpu().numpy()
        result = {
            "task": self.task["sha256"],
            "final_logits_sha256": hashlib.sha256(
                values.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "input_batches_sha256": self.input_hash.hexdigest(),
            "input_batches": self.input_batches,
            "positions": self.schedule.cursor,
            "repair_receipt": self.repairs.receipt() if self.repairs else None,
        }
        if self.profiler is not None:
            require(
                self.profile_steps == self.task["profile_steps"], "profile window was shortened"
            )
            # Retain raw evidence even when attribution coverage fails below.
            trace_path = self.root / "profile-trace.json"
            self.profiler.export_chrome_trace(str(trace_path))
            result["profile"] = attribute_trace(json.loads(trace_path.read_text())["traceEvents"])
            result["profile"]["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            result["profile"]["steps"] = self.profile_steps
            result["profile"]["module_inventory"] = self.scopes.inventory
            for row in result["profile"]["stages"].values():
                row["ms_per_step"] = row["kernel_us"] / (1000 * self.profile_steps)
            required = {
                "gdn_convolution",
                "gdn_recurrence_gates_state",
                "gdn_output_norm",
                "kv_cache_and_attention",
                "target_vocabulary_head",
                "drafter",
            }
            result["profile"]["missing_stages"] = sorted(
                k for k in required if not result["profile"]["stages"].get(k, {}).get("kernels", 0)
            )
            result["profile"]["coverage"] = (
                "INCOMPLETE"
                if result["profile"]["missing_stages"] or result["profile"]["unlinked_kernels"]
                else "COMPLETE"
            )
        result = seal(result)
        write_private(self.root / "result.json", result)
        return result

    def close(self):
        self.stop_profile()
        if self.scopes is not None:
            self.scopes.close()
        super().close()


class StageWorkerExtension(EquivalenceWorkerExtension):
    def qwen_equivalence_install(self, task_path):
        runner = self.model_runner
        require(not hasattr(runner, "_qwen_equivalence"), "probe already attached")
        probe = StageProbe(runner, private_json(Path(task_path)))
        try:
            probe.attach()
        except BaseException:
            probe.close()
            raise
        runner._qwen_equivalence = probe
        return {"installed": True}


def run_worker(args):
    from vllm import LLM, SamplingParams

    from qwen_r9700_lab.conformance_cli import reject_dead_native_rpcs
    from qwen_r9700_lab.conformance_model import Checkpoint
    from qwen_r9700_lab.conformance_radiance import verify_sources

    spec = private_json(args.spec)
    fixture = private_json(args.private / "fixture.json")
    authenticate(fixture)
    package = Path(importlib.util.find_spec("vllm").origin).parent.parent
    verify_sources(package, spec["binding"])
    require(os.environ.get("RADIANCE_VERIFY_HEAD") == "0", "full BF16 head is required")
    checkpoint = Checkpoint(Path(spec["native_config"]["model"]), spec["checkpoint_files"])
    write_private(args.output / f"{args.arm}-checkpoint.json", {"identity": checkpoint.identity})
    checkpoint.close()
    config = dict(spec["native_config"])
    config.update(
        enforce_eager=True,
        max_num_seqs=1,
        async_scheduling=False,
        enable_prefix_caching=True,
        disable_log_stats=True,
        worker_extension_cls="profile_d7_stages.StageWorkerExtension",
    )
    for key in ("kv_transfer_config", "scheduler_cls", "additional_config", "compilation_config"):
        config.pop(key, None)
    write_private(args.output / f"{args.arm}-config.json", seal(config))
    llm = LLM(**config)
    engine = llm.llm_engine
    with reject_dead_native_rpcs(engine.engine_core):
        write_private(
            args.output / f"{args.arm}-runtime.json",
            {"workers": llm.collective_rpc("qwen_equivalence_runtime")},
        )
    records = []
    if args.arm == "old" and args.old_reuse:
        measurement = private_json(args.old_reuse / "measurement.json")
        authenticate(measurement)
        require(
            measurement["fixture"] == fixture["sha256"]
            and measurement["binding"] == spec["binding"]["sha256"],
            "reused timing identity differs",
        )
        for path in sorted(args.old_reuse.glob("old-pass-*.json")):
            saved = private_json(path)
            authenticate(saved)
            if saved["mode"] == "clean":
                require(saved["timed_steps"] == args.steps, "reused timing window differs")
                records.append(saved)
        require(len(records) == args.repeats, "incomplete prior clean timings")
        write_private(
            args.output / "old-timing-reuse.json",
            seal(
                {
                    "measurement": measurement["sha256"],
                    "passes": [r["sha256"] for r in records],
                    "scope": "Reuse completed clean timings; new warmup and profile only.",
                }
            ),
        )
        modes = ["warmup", "profile"]
    else:
        modes = ["warmup"] + ["clean"] * args.repeats + ["profile"]
    for index, mode in enumerate(modes):
        root = args.private / f"{args.arm}-{index:02d}"
        task = seal(
            {
                "continuation": str(args.private / "fixture.json"),
                "continuation_sha256": fixture["sha256"],
                "private_output": str(root),
                "binding": spec["binding"],
                "speculation": True,
                "arm": "m8",
                "index": index,
                "report_root": str(args.output),
                "trace_rows": False,
                "repair_manifest": str(args.repair_manifest) if args.arm == "fixed" else None,
                "mode": mode,
                "warmup_steps": args.warmup_steps,
                "profile_steps": args.profile_steps,
            }
        )
        task_path = args.output / f"{args.arm}-task-{index:02d}.json"
        write_private(task_path, task)
        params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=-1,
            ignore_eos=True,
            max_tokens=len(fixture["output"]),
            detokenize=False,
        )
        request = f"qwen-stage-profile-{args.arm}-{index}"
        received = 0
        durations = []
        prefill_seconds = None
        try:
            with reject_dead_native_rpcs(engine.engine_core):
                llm.collective_rpc("qwen_equivalence_install", args=(str(task_path),))
                started = time.perf_counter()
                engine.add_request(
                    request,
                    {
                        "prompt_token_ids": fixture["prefix"],
                        "cache_salt": digest(
                            {"fixture": fixture["sha256"], "arm": args.arm, "pass": index}
                        ),
                    },
                    params,
                )
                while engine.has_unfinished_requests():
                    before = received
                    step_started = time.perf_counter()
                    outputs = engine.step()
                    elapsed = time.perf_counter() - step_started
                    for output in outputs:
                        require(output.request_id == request, "unexpected request output")
                        if output.outputs:
                            received = len(output.outputs[0].token_ids)
                    if before == 0 and received:
                        prefill_seconds = time.perf_counter() - started
                    if before >= 1 + args.warmup_steps * 8 and received > before:
                        require(received - before == 8, "timed step is not M8")
                        durations.append(elapsed)
                    if received >= len(fixture["output"]):
                        require(
                            list(outputs[-1].outputs[0].token_ids) == fixture["output"],
                            "forced output changed",
                        )
                        break
                require(received == len(fixture["output"]), "incomplete replay")
                receipts = llm.collective_rpc("qwen_equivalence_finish")
                require(len(receipts) == 1, "unexpected worker count")
                record = seal(
                    {
                        "mode": mode,
                        "index": index,
                        "prefill_seconds": prefill_seconds,
                        "step_seconds": durations,
                        "timed_steps": len(durations),
                        "median_step_ms": 1000 * statistics.median(durations),
                        "worker": receipts[0],
                    }
                )
                require(len(durations) == args.steps, "timed step budget changed")
                write_private(args.output / f"{args.arm}-pass-{index:02d}.json", record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "pass": index,
                            "mode": mode,
                            "median_step_ms": record["median_step_ms"],
                            "profile": "profile" in receipts[0],
                        }
                    ),
                    flush=True,
                )
        finally:
            engine.abort_request([request])
    hashes = {r["worker"]["final_logits_sha256"] for r in records}
    require(len(hashes) == 1, "profiling/repeated execution changed final logits")
    clean = [r for r in records if r["mode"] == "clean"]
    result = seal(
        {
            "arm": args.arm,
            "fixture": fixture["sha256"],
            "same_final_logits_all_passes": True,
            "passes": [r["sha256"] for r in records],
            "clean_median_step_ms": statistics.median([r["median_step_ms"] for r in clean]),
            "clean_total_step_seconds": sum(sum(r["step_seconds"]) for r in clean),
            "clean_steps": sum(r["timed_steps"] for r in clean),
            "prefill_seconds": [r["prefill_seconds"] for r in clean],
            "profile": records[-1]["worker"]["profile"],
            "scope": (
                "Eager TP1, 60K private prefix, forced D7 acceptance. Clean timings include "
                "minimal replay control; GPU profile is separate. Not natural Pi throughput."
            ),
        }
    )
    write_private(args.output / f"{args.arm}-summary.json", result)


def run(args):
    from stock_gdn_runtime import validate_manifest

    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    require(
        args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"),
        "GPU lease admission required",
    )
    validate_manifest(args.repair_manifest)
    private_root(args.corpus)
    args.private.mkdir(mode=0o700)
    private_root(args.private)
    args.output.mkdir(mode=0o700)
    manifest = private_json(args.corpus / "manifest.json")
    authenticate(manifest)
    source = private_json(args.corpus / manifest["continuations"][0]["name"])
    authenticate(source)
    require(source["sha256"] == manifest["continuations"][0]["sha256"], "source corpus changed")
    offset = 60000 - len(source["prefix"])
    count = 8 * (args.warmup_steps + args.steps)
    require(
        offset >= 0 and offset + count < len(source["output"]),
        "saved natural continuation is too short",
    )
    fixture = seal(
        {
            "prefix": source["prefix"] + source["output"][:offset],
            "output": source["output"][offset : offset + count + 1],
            "source": source["sha256"],
            "source_offset": offset,
        }
    )
    write_private(args.private / "fixture.json", fixture)
    spec = private_json(args.spec)
    write_private(
        args.output / "measurement.json",
        seal(
            {
                "fixture": fixture["sha256"],
                "source_corpus": manifest["sha256"],
                "prefix_tokens": 60000,
                "decode_positions": count,
                "binding": spec["binding"]["sha256"],
                "repair": private_json(args.repair_manifest)["sha256"],
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "attribution_sha256": hashlib.sha256(
                    Path(__file__).with_name("d7_stage_attribution.py").read_bytes()
                ).hexdigest(),
                "warmup_steps": args.warmup_steps,
                "timed_steps": args.steps,
                "profile_steps": args.profile_steps,
                "clean_repeats": args.repeats,
            }
        ),
    )
    env = worker_environment(spec, args.old_reuse or args.output)
    env["RADIANCE_VERIFY_HEAD"] = "0"
    env.pop("QWEN_CONFORMANCE_NATIVE_EXPERIMENT", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env["PYTHONPATH"]
    with gpu_lease(args.output / "gpu-lease"):
        for arm in ("old", "fixed"):
            argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--arm", arm]
            for key in (
                "spec",
                "corpus",
                "private",
                "output",
                "repair_manifest",
                "steps",
                "warmup_steps",
                "profile_steps",
                "repeats",
            ):
                argv += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            if args.old_reuse:
                argv += ["--old-reuse", str(args.old_reuse)]
            with OwnedProcess(
                argv, args.private / f"{arm}-process", env=env, timeout=7200
            ) as process:
                code = process.wait()
            write_private(args.output / f"{arm}-process-result.json", {"returncode": code})
            require(
                code == 0,
                "stage worker failed; private log retained without printing chat contents",
            )
    reports = {arm: private_json(args.output / f"{arm}-summary.json") for arm in ("old", "fixed")}
    for report in reports.values():
        authenticate(report)
    complete = all(r["profile"]["coverage"] == "COMPLETE" for r in reports.values())
    write_private(
        args.output / "summary.json",
        seal({"status": "MEASURED" if complete else "PROFILE_INCOMPLETE", **reports}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    for key in ("spec", "corpus", "private", "output", "repair-manifest"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--arm", choices=("old", "fixed"))
    parser.add_argument("--old-reuse", type=Path)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--profile-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(
        args.steps >= args.profile_steps > 0 and args.warmup_steps > 0 and args.repeats > 0,
        "invalid measurement window",
    )
    os.umask(0o077)
    (run if args.command == "run" else run_worker)(args)


if __name__ == "__main__":
    main()
