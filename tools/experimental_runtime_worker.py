"""Install host embedding storage during startup of the pinned worker."""

from host_embedding import host_embedding_during_load
from optimized_d7_worker import OptimizedWorker


class ExperimentalRuntimeWorker(OptimizedWorker):
    def load_model(self, **kwargs):
        with host_embedding_during_load(self):
            return super().load_model(**kwargs)
