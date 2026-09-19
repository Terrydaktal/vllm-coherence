"""Temporary, explicit interventions for a private diagnostic backend."""

import time


class RoundLatencyProbe:
    def qwen_round_latency_profile(self, operation, root, mode="gpu"):
        import torch
        from pathlib import Path

        if mode not in ("cpu", "gpu"):
            raise ValueError("unsupported profile mode")
        if operation == "start":
            if hasattr(self, "_round_latency_profile"):
                raise RuntimeError("profile already active")
            path = Path(root)
            path.mkdir(mode=0o700)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if mode == "gpu":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # Deliberately omit synchronization: it erases the slow state.
            profile = torch.profiler.profile(
                activities=activities, record_shapes=False,
                profile_memory=False, with_stack=False,
            )
            self._round_latency_profile = profile
            self._round_latency_profile_root = path
            profile.start()
        elif operation == "stop":
            profile = self._round_latency_profile
            profile.stop()
            profile.export_chrome_trace(str(self._round_latency_profile_root / "profile-trace.json"))
            del self._round_latency_profile, self._round_latency_profile_root
        else:
            raise ValueError("unsupported profile operation")
        return {"profile_operation": operation, "mode": mode, "explicit_presynchronization": False}

    def qwen_round_latency_probe(self, operation):
        import torch

        started = time.perf_counter()
        result = {"operation": operation}
        if operation == "noop":
            pass
        elif operation == "device_sync":
            torch.cuda.synchronize()
        elif operation == "stream_sync":
            torch.cuda.current_stream().synchronize()
        elif operation == "host_pause":
            time.sleep(0.05)
        elif operation == "timing_events":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            end.record()
            end.synchronize()
            result["event_elapsed_ms"] = begin.elapsed_time(end)
        else:
            raise ValueError("unsupported diagnostic operation")
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return result
