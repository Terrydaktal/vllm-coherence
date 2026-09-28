"""Compare optimized cold prefill with preserved corrected decode activations.

Only explicitly synthetic corpora are admitted. Reference token identities and
positions are checked by the worker; full hidden vectors and full BF16 logits
are compared, not only the generated winner. Unchanged decode captures avoid
repeating a slow reference execution for each memory-layout optimization.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from qualify_speed_lifecycle import Suite

from qwen_r9700_lab.diagnostic_contract import seal, write_private


async def run(args):
    args.output.mkdir(mode=0o700)
    cfg = SimpleNamespace(
        base_url=args.base_url,
        abi=args.abi,
        run_id=f"prefill-qualified-speed-{time.time_ns()}",
        output=args.output / "suite.json",
        status_prefix=Path("/dev/shm/qwen-prefill-diagnosis"),
        observe=False,
        case_group="prefill-qualified-speed",
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=1200) as client:
        suite = Suite(cfg, client)
        suite.sampling = {"temperature": 0, "top_k": 129, "seed": 0}
        await suite.rpc("qwen_prefill_probe_reload")
        sources = await suite.rpc(
            "qwen_prefill_load_candidate_sources", [str(args.sources)]
        )
        # A newer frozen server already owns prefill hooks. Remove those before
        # installing the explicitly selected candidate, as the paired timing
        # runner does; stacking wrappers would test a different implementation.
        await suite.rpc("qwen_prefill_legacy_timing")
        installed = await suite.rpc(
            "qwen_prefill_install_runtime",
            [str(args.attention), str(args.projection), True],
        )
        report = {
            "status": "RUNNING",
            "sources": sources,
            "installed": installed,
            "cases": [],
        }
        for reference in args.references:
            corpus_path = reference / "synthetic-corpus.json"
            corpus = json.loads(corpus_path.read_text())
            if corpus.get("synthetic") is not True:
                raise ValueError("explicit synthetic corpus required")
            prefix, output = corpus["prefix"], corpus["output"]
            if not prefix or len(output) < 2:
                raise ValueError("empty corpus")
            destination = args.output / f"context-{len(prefix)}"
            await suite.rpc(
                "qwen_prefill_capture_begin", [str(destination), len(prefix)]
            )
            try:
                job = suite.stream(
                    f"prefill-{len(prefix)}",
                    prefix + output[:-1],
                    1,
                    suite.chat("qualified-speed", str(len(prefix))),
                )
                await asyncio.wait_for(asyncio.shield(job.task), 1200)
                result = await job.result()
                await suite.absent(job.chat)
            finally:
                await suite.rpc("qwen_prefill_capture_finish")
            comparison = await suite.rpc(
                "qwen_prefill_compare", [str(reference / "decode"), str(destination)]
            )
            n = len(output) - 1
            if (
                comparison["positions"] != n
                or comparison["hidden_exact_rows"] != n
                or comparison["logits"]["full_logits_exact"] != n
            ):
                raise AssertionError(
                    f"prefill speed candidate differs at {len(prefix)} context"
                )
            row = {
                "context": len(prefix),
                "comparison": comparison,
                "request": result,
                "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                "reference_files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((reference / "decode").glob("*.pt"))
                },
            }
            runtime = await suite.rpc("qwen_prefill_runtime_info")
            calls = runtime["candidate"]["calls"]
            if not (
                calls["attention_prefill"] > 0
                and calls["projection_prefill"] > 0
                and calls["input_tiled"] > 0
                and calls["gdn"]["prepared"] > 0
                and calls["convolution"]["calls"]["prefill"] > 0
            ):
                raise AssertionError("an intended prefill optimization was not exercised")
            row["runtime_calls"] = calls
            report["cases"].append(row)
            write_private(args.output / f"evidence-{len(prefix)}.json", seal(row))
            print(
                json.dumps({"context": len(prefix), "comparison": comparison}),
                flush=True,
            )
        report["status"] = "SAMPLE_CHECKED"
        write_private(args.output / "evidence.json", seal(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8081")
    parser.add_argument("--abi", required=True)
    for name in ("sources", "attention", "projection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    asyncio.run(run(parser.parse_args()))
