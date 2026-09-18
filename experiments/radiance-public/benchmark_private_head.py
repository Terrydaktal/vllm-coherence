"""Private Pi head benchmark: real requests, consecutive captures, isolated GPU timings.

Run inside the pinned qualification container under its GPU lease. Raw requests,
outputs and captures remain in an owned private tmpfs directory. Normal reports
contain hashes, counts, timing summaries and no decoded conversation text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def write(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def prepare_environment(spec, root, report_root):
    from qwen_r9700_lab import conformance_runtime

    original_environment = conformance_runtime.worker_environment
    overlay = root / "python-overlay"
    overlay.mkdir(mode=0o700)
    original_startup = Path(conformance_runtime.__file__).resolve().parents[1] / "sitecustomize.py"
    (overlay / "sitecustomize.py").write_text(
        "import os, runpy\n"
        + (f"runpy.run_path({str(original_startup)!r})\n" if original_startup.exists() else "")
        + "if os.environ.get('QWEN_PRIVATE_HEAD_ROOT'):\n"
        "    try:\n"
        "        from private_head_probe import install\n"
        "        install()\n"
        "    except Exception:\n"
        "        os.write(2, b'private benchmark hook installation failed\\n')\n"
        "        os._exit(87)\n"
    )

    def benchmark_environment(bound_spec, private_root):
        env = original_environment(bound_spec, private_root)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(overlay), str(Path(__file__).parent), env["PYTHONPATH"])
        )
        # /dev/shm is mounted noexec. Compiled libraries contain no chat data
        # and need an executable filesystem; requests/state remain in tmpfs.
        for key in (
            "VLLM_CACHE_ROOT",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
            "TORCH_EXTENSIONS_DIR",
            "CUDA_CACHE_PATH",
            "AITER_ROOT_DIR",
        ):
            if key not in env:
                continue
            target = (
                Path(
                    os.environ.get("QWEN_PRIVATE_HEAD_COMPILE_ROOT", report_root / "compiled-cache")
                )
                / key.lower()
            )
            if key == "AITER_ROOT_DIR" and not target.exists():
                shutil.copytree(env[key], target)
            else:
                target.mkdir(parents=True, mode=0o700, exist_ok=True)
            env[key] = str(target)
        return env

    conformance_runtime.worker_environment = benchmark_environment

    env = benchmark_environment(spec, root)
    os.environ.update(env)
    os.environ["QWEN_PRIVATE_HEAD_ROOT"] = str(root)
    spec["environment"]["QWEN_PRIVATE_HEAD_ROOT"] = str(root)
    spec["environment"]["PYTHONPATH"] = str(Path(__file__).parent)
    spec["server_config"]["enable_log_requests"] = False
    spec["server_config"]["max_log_len"] = 0
    spec["case_timeout_seconds"] = 7200


def request(
    server,
    root,
    tokens,
    mode,
    seed,
    *,
    capture=False,
    limit=1024,
    capture_calls=256,
    compact_capture=False,
    progress=None,
    chat_id=None,
):
    index = len(list(root.glob("output-*.json")))
    write(
        root / "control.json",
        {
            "mode": mode,
            "capture": capture,
            "capture_calls": capture_calls,
            "compact_capture": compact_capture,
            "trial": index,
        },
    )
    import urllib.request

    body = {
        "model": server.settings["config"].get(
            "served_model_name", server.settings["config"]["model"]
        ),
        "prompt": tokens,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 20,
        "seed": seed,
        "max_tokens": limit,
        "ignore_eos": False,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": "qwen-chat-cache-v1:" + (chat_id or digest(tokens)) + ":" + digest(str(root)),
        "kv_transfer_params": {
            "qwen_chat": {
                "id": chat_id or digest(tokens),
                "generation": digest(str(root)),
                "cwd": str(root),
                "title": "isolated private head benchmark",
            },
            "qwen_snapshot_abi": server.spec["binding"]["live_data_abi"],
        },
    }
    url = server.client.base + "/v1/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + server.client.nonce,
        },
    )
    first = last = None
    first_count = 0
    output_ids = []
    usage, finish = {}, None
    start = time.perf_counter()
    last_progress = start
    with server.client.opener.open(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    now = time.perf_counter()
                    if first is None:
                        first, first_count = now, len(ids)
                    last = now
                    output_ids.extend(ids)
                    if progress is not None and now - last_progress >= 5:
                        write(
                            progress,
                            {
                                "trial": index,
                                "mode": mode,
                                "capture": capture,
                                "output_tokens": len(output_ids),
                                "elapsed_seconds": now - start,
                            },
                        )
                        last_progress = now
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.perf_counter() - start
    if not output_ids or first is None or last is None:
        raise ValueError("benchmark stream did not provide token IDs")
    # Token IDs are sensitive too. They remain in tmpfs and are never printed.
    write(root / f"output-{index:02d}.json", {"tokens": output_ids})
    elapsed = last - first
    return {
        "trial": index,
        "mode": mode,
        "seed": seed,
        "capture_enabled": capture,
        "input_tokens": len(tokens),
        "output_tokens": len(output_ids),
        "first_chunk_tokens": first_count,
        "first_token_seconds": first - start,
        "post_first_seconds": elapsed,
        "wall_seconds": total,
        "post_first_tps": (len(output_ids) - first_count) / elapsed if elapsed else None,
        "finish_reason": finish,
        "usage": usage,
        "output_sha256": digest(output_ids),
    }


def serve(spec, root, report_root):
    from qwen_r9700_lab.conformance_runtime import NativeServer

    records = []
    prepare_environment(spec, root, report_root)
    write(root / "control.json", {"mode": "full", "capture": False})
    payload = json.loads((root / "payload.json").read_text())
    with NativeServer(spec, root / "server", allow_gpu=True, head=True, observe=False) as server:
        # /tokenize applies the actual pinned server template and tool schema.
        tokenize_body = {
            k: payload[k]
            for k in ("messages", "tools", "model", "chat_template_kwargs")
            if k in payload
        }
        tokenize_body["add_generation_prompt"] = True
        import urllib.request

        req = urllib.request.Request(
            server.client.base + "/tokenize",
            data=json.dumps(tokenize_body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + server.client.nonce,
            },
        )
        with server.client.opener.open(req, timeout=120) as response:
            tokenized = json.load(response)
        tokens = tokenized["tokens"]
        if not 55000 <= len(tokens) <= 66000:
            raise ValueError("real Pi prompt is outside the intended approximately 60K window")
        write(root / "fixture.json", {"tokens": tokens, "sha256": digest(tokens)})
        write(
            report_root / "fixture.json",
            {
                "source": "legitmoney Pi session, intact request boundary",
                "tokens": len(tokens),
                "sha256": digest(tokens),
                "message_count": len(payload["messages"]),
                "tools": len(payload.get("tools", [])),
                "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 20},
            },
        )
        # Warm prefix/model; then compare interleaved variants on fixed seeds.
        warm = request(server, root, tokens, "full", 0, limit=32)
        if not list(root.glob("hook-*.json")):
            raise ValueError("target hook absent after warmup; comparison refused")
        write(report_root / "warmup.json", warm)
        for seed, order in (
            (0, ("full", "block80", "global128", "global256")),
            (17, ("global256", "global128", "block80", "full")),
            (42, ("block80", "full", "global256", "global128")),
        ):
            for mode in order:
                result = request(server, root, tokens, mode, seed)
                records.append(result)
                write(report_root / "generation.json", records)
                print(json.dumps({"phase": "generation", **result}), flush=True)
        # Capture unbiased consecutive target-head calls on a reference continuation.
        capture = request(server, root, tokens, "full", 73, capture=True, limit=1536)
        write(report_root / "capture.json", capture)
        # The next head invocation flushes the previous trial's counters once,
        # avoiding a synchronized stats write on every decoding step.
        request(server, root, tokens, "full", 0, limit=1)
        write(
            report_root / "head-calls.json",
            [json.loads(p.read_text()) for p in sorted(root.glob("trial-*.json"))],
        )
        if not list(root.glob("head-*.pt")) or not list(root.glob("hook-*.json")):
            raise ValueError("target head capture hook did not execute")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.private_root
    if not str(root).startswith("/dev/shm/qwen-private-head-") or root.stat().st_mode & 0o077:
        raise ValueError("private root must be private tmpfs")
    args.report_root.mkdir(mode=0o700, exist_ok=False)
    spec = json.loads(args.spec.read_text())
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    with gpu_lease(args.report_root / "gpu-lease"):
        print(json.dumps({"phase": "gpu_acquired"}), flush=True)
        serve(spec, root, args.report_root)
        from benchmark_private_head_stage import run as run_stage

        run_stage(spec, root, args.report_root)
    print(json.dumps({"phase": "generation_complete"}), flush=True)


if __name__ == "__main__":
    main()
