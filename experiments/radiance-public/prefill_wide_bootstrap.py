"""Check the frozen parent, then launch an explicitly isolated wider experiment.

The normal release bootstrap and worker admission remain unchanged. This entry
point requires the test port, test snapshot directory and experiment mount.
"""

import json
import os
import runpy
import sys
from pathlib import Path


def main():
    root = Path(os.environ["QWEN_PREFILL_SPEED_EXPERIMENT"])
    if not root.is_relative_to("/prefill-diagnosis"):
        raise ValueError("isolated experiment mount required")
    original = os.execv

    def execute(path, args):
        def option(name):
            if args.count(name) != 1:
                raise ValueError("experimental option must appear once: " + name)
            return args[args.index(name) + 1]

        cache = json.loads(option("--kv-transfer-config"))
        tiers = cache["kv_connector_extra_config"]["secondary_tiers"]
        if (
            path != "/opt/radiance_entrypoint.sh"
            or option("--port") != "8081"
            or option("--worker-cls") != "speed_candidate_worker.SpeedCandidateWorker"
            or any(
                tier.get("root_dir") != "/cache/prefill-divergence-20260927/data"
                for tier in tiers
            )
            or not tiers
        ):
            raise ValueError("wider experiment cannot use the serving endpoint/cache")
        args[args.index("--worker-cls") + 1] = "prefill_wide_worker.WidePrefillWorker"
        print("UNQUALIFIED isolated wider-prefill experiment", flush=True)
        original(path, args)

    sys.path.insert(0, "/patches")
    os.execv = execute
    runpy.run_path("/patches/bootstrap_radiance_release.py", run_name="__main__")


if __name__ == "__main__":
    main()
