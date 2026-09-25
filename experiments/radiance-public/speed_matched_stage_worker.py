"""Measure the banked candidate without reverting its full target graph.

Only profile arms install host annotations. Controls use the same captured GPU
graphs without stage events, tensor reads, forced tokens or extra round fences.
"""

import contextlib
import hashlib
from pathlib import Path

import torch
from matched_stage_profile_worker import MatchedStageWorker
from optimized_d7_worker import GraphObservation
from speed_candidate_worker import SpeedCandidateWorker

from qwen_r9700_lab.conformance_instrumentation import HookSet


class FullGraphObservation(GraphObservation):
    def __init__(self, *args, **kwargs):
        self.target_depth = 0
        super().__init__(*args, **kwargs)
        original = torch.cuda.CUDAGraph.replay

        def replay(graph):
            # Piecewise launches already sit inside model.forward's scope.
            # A full replay bypasses that Python method entirely.
            if self.draft or self.target_depth:
                return original(graph)
            self.counts["full_target_graph_replays"] += 1
            with self.scope("target_body"):
                return original(graph)

        self.replay_hooks = HookSet()
        self.replay_hooks.replace(torch.cuda.CUDAGraph, "replay", replay)

    def close(self):
        self.replay_hooks.close()
        return super().close()

    @contextlib.contextmanager
    def scope(self, name):
        target = name == "target_body"
        self.target_depth += int(target)
        try:
            with super().scope(name):
                yield
        finally:
            self.target_depth -= int(target)


class SpeedMatchedStageWorker(MatchedStageWorker, SpeedCandidateWorker):
    observation_class = FullGraphObservation

    def qwen_optimized_metadata(self):
        metadata = super().qwen_optimized_metadata()
        full = getattr(self, "_qwen_speed_full_graph", True)
        # These are immutable qualification identities and explicit selectors,
        # not growing dispatch counters. Cache reuse must bind all of them.
        metadata["speed_candidate"] = {
            "full_graph_enabled": full,
            "target_graph_launches_per_round": 1 if full else 65,
            "draft_attention": getattr(self, "_qwen_speed_draft_attention_candidate", None),
            "draft_attention_enabled": getattr(self, "_qwen_speed_draft_attention_enabled", None),
            "target_gemm": getattr(self, "_qwen_speed_target_gemm_candidate", None),
            "target_gemm_enabled": getattr(self, "_qwen_speed_gemm_enabled", None),
            "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("speed_matched_stage_worker.py", "speed_candidate_worker.py",
                                     "matched_stage_profile_worker.py")},
        }
        return metadata

    def qwen_timing_arm(self, *args, **kwargs):
        if (getattr(self, "_qwen_speed_head_graph", None)
                or getattr(self, "_qwen_speed_sampler_graph", None)):
            raise ValueError("this profile admits only the three banked speed candidates")
        return super().qwen_timing_arm(*args, **kwargs)
