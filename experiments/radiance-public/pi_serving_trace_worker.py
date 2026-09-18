"""Temporary qualification worker: bounded traces of the real serving path.

No prompt, output, tensor values, shapes or Python stacks are recorded. Install
only in an isolated diagnostic server, never in the production release payload.
"""

import cProfile
import json
import os
import time
from pathlib import Path

from optimized_d7_worker import GraphObservation, OptimizedWorker


class ServingTraceWorker(OptimizedWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._trace_root = Path(os.environ["QWEN_SERVING_TRACE_ROOT"])
        self._trace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._trace_counts = {"prefill": 0, "decode": 0}
        self._trace_done = set()
        self._trace_active = None
        self._trace_rows = []
        self._trace_observer = None
        self._trace_cpu = None
        self._trace_previous_start = None
        self._trace_events = []
        return result

    def execute_model(self, scheduler_output, *args, **kwargs):
        if not hasattr(self, "_trace_counts"):
            return super().execute_model(scheduler_output, *args, **kwargs)
        tokens = int(scheduler_output.total_num_scheduled_tokens)
        phase = "prefill" if tokens > 8 else "decode"
        if tokens:
            self._trace_counts[phase] += 1
        if self._trace_active:
            prior, stop = self._trace_active
            if phase != prior or self._trace_counts[phase] >= stop:
                if self._trace_cpu is not None:
                    self._trace_cpu.disable()
                    self._trace_cpu.dump_stats(str(self._trace_root / prior / "cpu.pstats"))
                    self._trace_cpu = None
                self._trace_observer.stop_profile()
                observed = self._trace_observer.close()
                (self._trace_root / prior / "observation.json").write_text(
                    json.dumps(observed, indent=2) + "\n"
                )
                self._trace_done.add(prior)
                self._trace_active = self._trace_observer = None
        start = int(
            os.environ.get(
                "QWEN_TRACE_PREFILL_START" if phase == "prefill" else "QWEN_TRACE_DECODE_START",
                "3" if phase == "prefill" else "9",
            )
        )
        if (
            tokens
            and phase not in self._trace_done
            and self._trace_active is None
            and self._trace_counts[phase] == start
            and os.environ.get("QWEN_TRACE_DISABLE_PROFILE") != "1"
        ):
            self._trace_observer = GraphObservation(
                self.model_runner, self._trace_root / phase, profile=True
            )
            self._trace_observer.start_profile()
            self._trace_active = (phase, start + (1 if phase == "prefill" else 8))
            if os.environ.get("QWEN_TRACE_CPU") == "1":
                self._trace_cpu = cProfile.Profile()
                self._trace_cpu.enable()
        before = time.perf_counter()
        interval = (
            None if self._trace_previous_start is None else before - self._trace_previous_start
        )
        self._trace_previous_start = before
        events = None
        if os.environ.get("QWEN_TRACE_LIGHT_EVENTS") == "1":
            import torch

            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        result = super().execute_model(scheduler_output, *args, **kwargs)
        if events:
            events[1].record()
            self._trace_events.append((len(self._trace_rows), "gpu_stream_ms", events))
        metadata = getattr(scheduler_output, "kv_connector_metadata", None)
        self._trace_rows.append(
            {
                "phase": phase,
                "tokens": tokens,
                "profiled": bool(self._trace_active),
                "seconds": time.perf_counter() - before,
                "start_interval_seconds": interval,
                "store_jobs": len(getattr(metadata, "store_jobs", {})),
                "load_jobs": len(getattr(metadata, "load_jobs", {})),
            }
        )
        if len(self._trace_rows) % 32 == 0:
            self._write_trace_counts()
        return result

    def sample_tokens(self, *args, **kwargs):
        if not hasattr(self, "_trace_rows") or not self._trace_rows:
            return super().sample_tokens(*args, **kwargs)
        before = time.perf_counter()
        row = self._trace_rows[-1]
        events = None
        if os.environ.get("QWEN_TRACE_LIGHT_EVENTS") == "1":
            import torch

            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        result = super().sample_tokens(*args, **kwargs)
        row["sample_host_seconds"] = time.perf_counter() - before
        if events:
            events[1].record()
            self._trace_events.append((len(self._trace_rows) - 1, "sample_gpu_ms", events))
        return result

    def _write_trace_counts(self):
        import radiance_mxfp4

        pending = []
        for index, key, events in self._trace_events:
            if events[1].query():
                self._trace_rows[index][key] = events[0].elapsed_time(events[1])
            else:
                pending.append((index, key, events))
        self._trace_events = pending

        # Numeric dispatch evidence only; never serialize model/scheduler objects.
        report = {
            "phases": self._trace_counts,
            "steps": self._trace_rows,
            "gemm_binary": str(radiance_mxfp4._ext.__file__),
            "performance": self._qwen_performance_repairs.receipt(),
        }
        target = self._trace_root / "steps.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(target)
