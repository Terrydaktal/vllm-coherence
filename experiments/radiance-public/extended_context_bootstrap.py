"""Check the frozen release before selecting the extended-context worker."""

import os
import runpy
import sys

STOCK = "speed_candidate_worker.SpeedCandidateWorker"
EXTENDED = "extended_context_worker.ExtendedContextWorker"
ENTRYPOINT = "/opt/radiance_entrypoint.sh"
execv = os.execv


def start(path, arguments):
    window = os.environ.get("QWEN_EXTENDED_CONTEXT_TOKENS", "319000")
    if (
        path != ENTRYPOINT
        or arguments.count("--worker-cls") != 1
        or arguments[arguments.index("--worker-cls") + 1] != STOCK
        or arguments.count("--max-model-len") != 1
        or arguments[arguments.index("--max-model-len") + 1] != window
    ):
        raise RuntimeError("unexpected extended-context release entry point")
    forwarded = arguments.copy()
    forwarded[forwarded.index("--worker-cls") + 1] = EXTENDED
    execv(path, forwarded)


if __name__ == "__main__":
    os.execv = start
    sys.path.insert(0, "/patches")
    runpy.run_path("/patches/bootstrap_radiance_release.py", run_name="__main__")
