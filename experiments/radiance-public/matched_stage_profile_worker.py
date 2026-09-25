"""Isolated serving profiler; never installed in the normal release worker.

Controls run the unchanged model and sampler. Only host round-boundary clocks
are collected; no forced tokens, tensor reads, stage events or round fences.
Trace activation/export happens between declared chunks, outside retained rounds.
"""

import json
import os
import time
import uuid
from pathlib import Path

from optimized_d7_worker import GraphObservation, OptimizedWorker


class MatchedStageWorker(OptimizedWorker):
    def qwen_timing_status(self):
        if not hasattr(self, "_timing_instance"):
            self._timing_instance = uuid.uuid4().hex
        return {"instance": self._timing_instance, "arm_active": hasattr(self, "_timing_run")}

    def qwen_timing_identity(self):
        """Bind resumable captures to this worker; no hot-path or GPU work."""
        model = self.vllm_config.model_config
        return {
            **self.qwen_timing_status(),
            "model": {key: str(getattr(model, key, None)) for key in
                      ("model", "revision", "tokenizer", "tokenizer_revision",
                       "dtype", "quantization")},
            "metadata": self.qwen_optimized_metadata(),
            "target_head_environment": {name: os.environ.get(name) for name in (
                "RADIANCE_VERIFY_HEAD", "RADIANCE_VERIFY_HEAD_GLOBAL_TOPK",
                "RADIANCE_VERIFY_HEAD_MAX_M", "RADIANCE_DRAFT_RERANK")},
        }

    def qwen_timing_arm(self, root, mode, warmup=64, chunk_rounds=128, max_rounds=1152):
        if mode not in {"control", "profile"} or hasattr(self, "_timing_run"):
            raise ValueError("invalid timing arm or unfinished run")
        root = Path(root)
        root.mkdir(mode=0o700)
        self._timing_run = {
            "root": root, "mode": mode, "warmup": int(warmup),
            "chunk_rounds": int(chunk_rounds), "max_rounds": int(max_rounds),
            "rows": [], "decode_count": 0, "chunks": [], "active": None,
            "observer": None, "captured_rounds": 0,
        }
        return {"armed": True, "metadata": self.qwen_optimized_metadata()}

    def _timing_stop_chunk(self):
        run = self._timing_run
        if run["observer"] is None:
            return
        run["observer"].stop_profile()
        observation = run["observer"].close()
        run["active"].update(observation=observation)
        run["chunks"].append(run["active"])
        run["active"] = run["observer"] = None

    def execute_model(self, scheduler_output, *args, **kwargs):
        run = getattr(self, "_timing_run", None)
        if run is None:
            return super().execute_model(scheduler_output, *args, **kwargs)
        tokens = int(scheduler_output.total_num_scheduled_tokens)
        if not 1 <= tokens <= 8:
            return super().execute_model(scheduler_output, *args, **kwargs)
        index = run["decode_count"]
        run["decode_count"] += 1
        active = run["active"]
        if active is not None and index >= active["end_exclusive"]:
            self._timing_stop_chunk()
        if (run["mode"] == "profile" and run["observer"] is None
                and index >= run["warmup"]
                and run["captured_rounds"] < run["max_rounds"]):
            size = min(run["chunk_rounds"], run["max_rounds"] - run["captured_rounds"])
            # Capture an extra boundary round; never invent a next-round start.
            root = run["root"] / f"chunk-{len(run['chunks']):03d}"
            observer = GraphObservation(self.model_runner, root, profile=True)
            observer.start_profile()
            run["observer"] = observer
            run["active"] = {"first_decode_index": index, "end_exclusive": index + size + 1}
            run["captured_rounds"] += size
        # The same entry-to-entry boundary is available in both modes. Profiling
        # markers let the analyzer locate these boundaries on the GPU trace clock.
        entry_ns = time.perf_counter_ns()
        row = {"decode_index": index, "entry_ns": entry_ns,
               "scheduled_tokens": tokens, "profiled": run["observer"] is not None}
        run["rows"].append(row)
        if run["observer"] is None:
            return super().execute_model(scheduler_output, *args, **kwargs)
        import torch
        with torch.profiler.record_function(f"qwen_timing_round/{index}"):
            return super().execute_model(scheduler_output, *args, **kwargs)

    def qwen_timing_finish(self):
        run = self._timing_run
        self._timing_stop_chunk()
        report = {k: v for k, v in run.items() if k not in {"root", "observer", "active"}}
        report["metadata"] = self.qwen_optimized_metadata()
        report["timing_contract"] = {
            "round_boundary": "worker_execute_entry_to_next_worker_execute_entry",
            "forced_replay_hooks": False, "per_stage_event_probes": False,
            "added_synchronization_inside_retained_rounds": False,
            "control_instrumentation": "One host clock and numeric record per decode round; ordinary production telemetry remains enabled.",
            "privacy": "No prompt, output tokens, tensors or chat text recorded.",
        }
        (run["root"] / "worker.json").write_text(json.dumps(report, indent=2) + "\n")
        del self._timing_run
        return report
