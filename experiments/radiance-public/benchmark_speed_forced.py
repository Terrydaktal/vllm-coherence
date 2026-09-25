"""Retained 320-position replay driver for the disposable /work test lane.

Consumes an authenticated private fixture; publishes only numeric evidence.
The isolated interpreter must stage benchmark_d7_equivalence.py under the
speed_equivalence_adapter import name. This is not a serving-speed test.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)
a = p.parse_args()
root = pathlib.Path("/work")
patches = pathlib.Path("/patches")
manifest = json.loads((patches / "optimized-release.json").read_text())
sys.path[:0] = manifest["pythonpath"]
from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private

fixture = json.loads((root / "fixture-60k.json").read_text())
authenticate(fixture)
assert len(fixture["prefix"]) == 60000 and len(fixture["output"]) == 321
profile = json.loads((patches / "runtime-radiance-1.0.16.json").read_text())
package = pathlib.Path("/opt/vllm/lib/python3.12/site-packages")
names = (
    set(profile["source_preimages"])
    | set(profile["kernel_hashes"])
    | {
        "speed_candidate_worker.py",
        "speed_equivalence_adapter.py",
        "radiance_r4d_attn.py",
    }
)
binding = seal(
    {
        "schema": "urn:qwen:radiance-native-binding:v1",
        "files": {
            n: hashlib.sha256((package / n).read_bytes()).hexdigest()
            for n in sorted(names)
        },
        "release_manifest_sha256": profile["optimized_d7"]["manifest_sha256"],
        "scope": "Installed source identity; numerical evidence measured separately",
    }
)
(root / (a.arm + "-forced-progress")).mkdir(mode=0o700)
task = seal(
    {
        "index": 0,
        "report_root": str(root / (a.arm + "-forced-progress")),
        "continuation": str(root / "fixture-60k.json"),
        "continuation_sha256": fixture["sha256"],
        "private_output": str(root / (a.arm + "-forced-rows")),
        "binding": binding,
        "speculation": True,
        "arm": "m8",
        "allow_approximate_head": True,
        "head_admission": "Qualification top_k=129 exceeds Global-512 capacity 128 and forces full BF16 fallback; summarize_logits rejects all non-finite or truncated vocabulary rows",
    }
)
path = root / (a.arm + "-forced-task.json")
write_private(path, task)


def post(endpoint, body, timeout=1200):
    req = urllib.request.Request(
        "http://127.0.0.1:8081" + endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        data = f.read()
        return json.loads(data) if data else {}


def rpc(method, args=None):
    return post(
        "/collective_rpc", {"method": method, "args": args or [], "timeout": 90}, 120
    )["results"][0]


post("/reset_prefix_cache", {})
rpc("qwen_speed_forced_begin", [str(root / (a.arm + "-forced-graphs")), str(path)])
result = post(
    "/v1/completions",
    {
        "model": "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate",
        "prompt": fixture["prefix"],
        "max_tokens": 321,
        "ignore_eos": True,
        "seed": 113,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 129,
        "stream": False,
        "return_token_ids": True,
        "add_special_tokens": False,
        "cache_salt": a.arm + "-forced",
    },
)
tokens = result["choices"][0]["token_ids"]
if tokens != fixture["output"]:
    raise RuntimeError("forced output token IDs differ from fixture")
evidence = rpc("qwen_optimized_finish")
write_private(
    root / (a.arm + "-forced-result.json"),
    {
        "evidence": evidence,
        "source_binding": binding,
        "production_head": "global512",
        "qualification_head": "full BF16 forced by qualification top_k=129, all full-vocabulary values required finite",
        "serving_speed_measurement": False,
    },
)
print(
    json.dumps(
        {
            "arm": a.arm,
            "forced_output_tokens": len(tokens),
            "observation": evidence["observation"],
        }
    ),
    flush=True,
)
