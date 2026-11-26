"""Install source-bound performance candidates before the first graph capture."""

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

from optimized_d7_worker import OptimizedWorker

from qwen_r9700_lab.diagnostic_contract import authenticate

BUILD_ROOT = Path("/coherence-experiments")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_build(name, binding):
    build = BUILD_ROOT / name
    manifest = (build / "build.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != binding["artifact"]:
        raise ValueError(f"performance build receipt changed: {name}")
    receipt = json.loads(manifest)
    authenticate(receipt)
    for filename, sha in receipt["files"].items():
        path = build / filename
        if Path(filename).name != filename or digest(path) != sha:
            raise ValueError(f"performance build file changed: {name}/{filename}")
    return build


class PerformanceWorker(OptimizedWorker):
    def model_load_context(self):
        return nullcontext()

    def load_model(self, **kwargs):
        experiments = json.loads(os.environ["QWEN_PERFORMANCE_EXPERIMENTS"])
        if digest(__file__) != experiments.pop("worker"):
            raise ValueError("performance worker changed")
        if experiments.keys() - {
            "cache_buffer_views",
            "gdn_prefill_direct",
            "head_live_rows",
            "mxfp4",
            "prompt_lookup",
            "proposal_temperature",
        }:
            raise ValueError("unsupported performance experiment")
        for binding in experiments.values():
            for filename, sha in binding["sources"].items():
                if (
                    Path(filename).name != filename
                    or digest(Path(__file__).with_name(filename)) != sha
                ):
                    raise ValueError(f"performance adapter changed: {filename}")
        mxfp4 = experiments.get("mxfp4")
        if mxfp4 is not None:
            if mxfp4["preprocess"] is not True:
                raise ValueError("MXFP4 serving requires weight preprocessing")
            mxfp4_build = verified_build("mxfp4", mxfp4)
        if "head_live_rows" in experiments:
            from head_live_rows import install

            install()
        with self.model_load_context():
            super().load_model(**kwargs)
        self._performance_counters = {}
        if "proposal_temperature" in experiments:
            from dflash_proposal_temperature import install

            set_temperature = install(self.model_runner)
            set_temperature(experiments["proposal_temperature"]["temperature"])
        if "prompt_lookup" in experiments:
            from dflash_prompt_lookup import install

            lookup = experiments["prompt_lookup"]
            self._performance_counters["prompt_lookup"] = install(
                self.model_runner,
                skip_draft=lookup["skip_draft"],
                streak_only=lookup["skip_draft"],
            )
        if mxfp4 is not None:
            from mxfp4_prefill_candidate import install

            self._performance_counters["mxfp4"] = install(
                self,
                library=mxfp4_build,
                slab_ahead=mxfp4["slab_ahead"],
            )
        gdn = experiments.get("gdn_prefill_direct")
        if gdn is not None:
            from gdn_prefill_direct import install_projection, install_scan

            if gdn["scan"]:
                self._performance_counters["gdn_direct_scan"] = install_scan(self)
            if gdn["projection"]:
                self._performance_counters["gdn_direct_projection"] = (
                    install_projection(self)
                )

    def qwen_performance_counters(self):
        return {
            name: counters.tolist() if hasattr(counters, "tolist") else counters
            for name, counters in self._performance_counters.items()
        }


class HostEmbeddingPerformanceWorker(PerformanceWorker):
    def model_load_context(self):
        from host_embedding import host_embedding_during_load

        return host_embedding_during_load(self)
