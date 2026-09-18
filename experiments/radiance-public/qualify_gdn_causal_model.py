"""Synthetic full-model qualification of the experimental causal GDN prefill.

Runs in child processes with a diagnostic hook; leaves the sealed matrix and
installed backend untouched. The candidate's scope is native M1 arithmetic,
including its halfway-away BF16 conversion, not stock RTNE equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def install_hook(build_path, prepared_qk=False):
    """Called by this experiment's sitecustomize in every worker, before probing."""
    from qwen_r9700_lab.conformance_instrumentation import HookSet
    from qwen_r9700_lab.conformance_radiance import RadianceProbe

    attach, detach = RadianceProbe.attach, RadianceProbe.detach

    def candidate_attach(self):
        import torch  # isort: skip

        import radiance_gdn as native
        from probe_gdn_causal_prefill import Candidate

        build = private_json(build_path / "build.json")
        authenticate(build)
        binary = build_path / "candidate.so"
        if hashlib.sha256(binary.read_bytes()).hexdigest() != build["binary_sha256"]:
            raise ValueError("causal GDN candidate library mismatch")
        candidate = Candidate(binary, native, prepared_qk)
        # Keep implementation replacements separate from observation wrappers;
        # each set refuses to instrument the same call site twice.
        replacements = HookSet()
        replacements.replace(native, "_CONV_PREP", candidate.conv)
        counters = {"scan_calls": 0, "tokens": 0}

        def no_wy(k, beta, g, cu, num_seqs, tokens, heads, qheads):
            return torch.empty(0, dtype=k.dtype, device=k.device)

        def causal_scan(
            q,
            k,
            v,
            matrix,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            core_attn_out,
            out=None,
        ):
            if (
                matrix.numel() != 0
                or not output_final_state
                or core_attn_out is not None
                or out is None
                or q.shape[0] != 1
                or initial_state.dtype != torch.float32
            ):
                raise ValueError("causal diagnostic reached an unsupported prefill call")
            if any(
                not t.is_contiguous() for t in (q, k, v, g, beta, initial_state, cu_seqlens, out)
            ):
                raise ValueError("causal diagnostic requires packed inputs and output")
            final = torch.empty_like(initial_state)
            values = candidate.normalized((q, k, v, g, beta), scale=float(scale))
            status = candidate.scan(
                *[t.data_ptr() for t in (*values, initial_state, out, final, cu_seqlens)],
                cu_seqlens.numel() - 1,
                v.shape[-2],
                q.shape[-2],
                float(scale),
                torch.cuda.current_stream().cuda_stream,
            )
            if status:
                raise RuntimeError(f"causal prefill failed: {status}")
            counters["scan_calls"] += 1
            counters["tokens"] += v.shape[1]
            return out, final

        replacements.replace(native, "kkt_solve", no_wy)
        replacements.replace(native, "fused_prefill", causal_scan)
        self._causal_candidate = (candidate, counters, build, replacements)
        attach(self)

    def candidate_detach(self):
        info = getattr(self, "_causal_candidate", None)
        if info is not None:
            _, counters, build, replacements = info
            destination = self.campaign.root / "causal-prefill-candidate.json"
            if not destination.exists():
                write_private(
                    destination,
                    seal(
                        {
                            **counters,
                            "build": build["sha256"],
                            "binary": build["binary_sha256"],
                            "prepared_fp32_qk": prepared_qk,
                            "hook": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "candidate_wrapper": hashlib.sha256(
                                Path(__file__).with_name("probe_gdn_causal_prefill.py").read_bytes()
                            ).hexdigest(),
                            "arithmetic_reference": (
                                "R4D M1; FP32 normalized QK and state; halfway-away BF16 output"
                            ),
                            "installed_sources_modified": False,
                        }
                    ),
                )
        try:
            return detach(self)
        finally:
            if info is not None:
                replacements.close()

    RadianceProbe.attach, RadianceProbe.detach = candidate_attach, candidate_detach


def worker(args):
    from qwen_r9700_lab.conformance_scenarios import native, plan_for, tokens
    from qwen_r9700_lab.conformance_state import compare_frames

    spec = private_json(args.spec)
    # The context/seed are the same synthetic prefix as the preserved diagnosis.
    plan = plan_for(spec, {"context": 2049, "seed": 0}, tokens(spec, 3, 1701), accepted=[0, 0])
    paths = [
        native(spec, plan, args.output / name, speculation=enabled)
        for name, enabled in (("serial", False), ("d7", True))
    ]
    schedules = [private_json(path / "schedule.json") for path in paths]
    if len(schedules[0]["frames"]) != 3 or len(schedules[1]["frames"]) != 3:
        raise ValueError("incomplete full-model comparison domain")
    comparisons = []
    for left, right in zip(schedules[0]["frames"], schedules[1]["frames"], strict=True):
        if (left["consumed"], left["pending"]) != (right["consumed"], right["pending"]):
            raise ValueError("different logical token histories")
        report = compare_frames(paths[0] / left["name"], paths[1] / right["name"])
        write_private(args.output / (left["name"] + "-comparison.json"), report)
        comparisons.append(
            {
                "consumed": left["consumed"],
                "equal": report["equal"],
                "first_difference": report["first_difference"],
                "sha256": report["sha256"],
            }
        )
    receipts = [private_json(path / "causal-prefill-candidate.json") for path in paths]
    if any(r["scan_calls"] <= 0 or r["tokens"] <= 0 for r in receipts):
        raise ValueError("candidate was not dispatched in both arms")
    report = seal(
        {
            "status": "TESTED" if all(r["equal"] for r in comparisons) else "DISCREPANCY",
            "comparisons": comparisons,
            "dispatch": receipts,
            "initial_prefill_exact": comparisons[0]["equal"],
            "scope": (
                "Synthetic 2049-token M1/D7 prefill and two forced transitions; isolated candidate."
            ),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "TESTED" else 2


def main(args):
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
    from qwen_r9700_lab.conformance_runtime import worker_environment
    from qwen_r9700_lab.conformance_transport import OwnedProcess

    os.umask(0o077)
    if args.worker:
        return worker(args)
    args.output.mkdir(mode=0o700)
    hook = args.output / "hook"
    hook.mkdir(mode=0o700)
    hook_source = (
        "from pathlib import Path\nfrom qualify_gdn_causal_model import install_hook\n"
        f"install_hook(Path({str(args.build)!r}), prepared_qk={args.prepared_qk!r})\n"
    )
    if args.stock_conv_repair is not None:
        hook_source += (
            "from capture_gdn_decode_transition import install_transition_hook\n"
            f"install_transition_hook(Path({str(args.stock_conv_repair)!r}))\n"
        )
    (hook / "sitecustomize.py").write_text(hook_source)
    spec = private_json(args.spec)
    env = worker_environment(spec, args.output)
    env["PYTHONPATH"] = f"{hook}:{Path(__file__).resolve().parent}:{env['PYTHONPATH']}"
    with gpu_lease(args.output / "gpu-lease"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--allow-gpu",
            "--spec",
            str(args.spec),
            "--build",
            str(args.build),
            "--output",
            str(args.output),
        ]
        if args.prepared_qk:
            command.append("--prepared-qk")
        if args.stock_conv_repair is not None:
            command.extend(["--stock-conv-repair", str(args.stock_conv_repair)])
        with OwnedProcess(command, args.output / "process", env=env, timeout=3600) as process:
            code = process.wait()
        write_private(
            args.output / "result.json",
            seal(
                {"returncode": code, "status": "COMPLETED" if code == 0 else "FAILED_OR_DISCREPANT"}
            ),
        )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepared-qk", action="store_true")
    parser.add_argument("--stock-conv-repair", type=Path)
    raise SystemExit(main(parser.parse_args()))
