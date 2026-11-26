from extended_context_worker import ExtendedContextWorker
from fourteen_retained import after_load, before_load
from speed_matched_stage_worker import SpeedMatchedStageWorker


class Worker(SpeedMatchedStageWorker, ExtendedContextWorker):
    def load_model(self, **kwargs):
        before_load(self)
        result = super().load_model(**kwargs)
        after_load(self)
        return result

    def qwen_optimized_metadata(self):
        metadata = super().qwen_optimized_metadata()
        metadata["retained_build"] = self._fourteen_prepared
        return metadata

    def qwen_fourteen_counts(self):
        return {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in self._fourteen_results.items()
        }
