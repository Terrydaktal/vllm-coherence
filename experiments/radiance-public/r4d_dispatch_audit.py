"""Count or forbid native R4D dispatch in isolated correctness experiments.

Installed before model imports in both comparison arms. Records only native
entrypoint names, counts, process identity and source hashes. It never inspects
tensor values or synchronizes the GPU; timings from these runs are not a speed
baseline. This is a kernel-dispatch check, not a generation or repetition guard.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

NATIVE_SHA256 = "daa7a3bf79d2a1e0a7909a6ed9ddac2f0f3ac74b878569f4eabc2ccae839aecc"
METADATA_EXPORTS = {
    "constraints", "explain", "kernels", "ops", "select", "selections",
    "attn_decode_h256_gqa6_scratch_bytes",
}
GPU_EXPORTS = {
    "ar_ipc_alloc", "ar_ipc_enable_peer", "ar_ipc_free", "ar_ipc_memzero",
    "ar_ipc_open", "ar_oneshot_2rank_exact", "ar_oneshot_2rank_exact_nq",
    "ar_oneshot_2rank_wht6", "attn_decode_h256_gqa6_bf16kv",
    "attn_decode_h256_gqa6_fp8kv", "attn_prefill_h256_gqa6_bf16kv",
    "attn_prefill_h256_gqa6_fp8kv", "attn_vit_h72_bf16",
    "dflash_conv_t2_g16_bf16", "gdn_chunk_scan_k128_v128_c64_bf16",
    "gdn_conv_prep_w4_h128_bf16", "gdn_conv_update_w4_h128_bf16",
    "gdn_fused_update_w4k128v128_bf16", "gdn_gated_rmsnorm_h128_bf16",
    "gdn_kkt_solve_k128_c64_bf16", "gdn_recurrent_update_k128_v128_bf16_fp32state",
    "gemm_bf16_nt_m16", "gemm_bf16_nt_m64", "gemm_w4a16_nt_m64",
    "gemm_w4a8_nt_m64", "quant_act_i8",
}


class ForbiddenR4DDispatch(RuntimeError):  # noqa: N818 - qualified diagnostic exception name
    pass


class DispatchAudit:
    def __init__(self, mode, root, names, native_sha256):
        if mode not in {"count", "forbid"}:
            raise ValueError("invalid R4D diagnostic mode")
        self.mode, self.root = mode, Path(root)
        self.names, self.native_sha256 = tuple(sorted(names)), native_sha256
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.reset_process()

    def reset_process(self):
        # A fork must not inherit its parent's counts, lock or reporter thread.
        self.pid, self.started_ns = os.getpid(), time.time_ns()
        self.calls = dict.fromkeys(self.names, 0)
        self.blocked = dict.fromkeys(self.names, 0)
        self.lock = threading.Lock()
        self.report_lock = threading.Lock()
        self.path = self.root / f"dispatch-{self.pid}-{self.started_ns}.json"

    def wrap(self, name, native):
        @functools.wraps(native)
        def call(*args, **kwargs):
            with self.lock:
                self.calls[name] += 1
                if self.mode == "forbid":
                    self.blocked[name] += 1
            if self.mode == "forbid":
                # Even callers that catch this exception cannot conceal a
                # forbidden dispatch from the experiment's final validation.
                self.write()
                raise ForbiddenR4DDispatch("R4D dispatch forbidden by diagnostic control")
            return native(*args, **kwargs)

        return call

    def write(self):
        with self.lock:
            report = {
                "schema": "qwen-r4d-native-dispatch-v1",
                "pid": self.pid, "started_ns": self.started_ns,
                "updated_at": time.time(), "mode": self.mode,
                "native_sha256": self.native_sha256,
                "master_switch": os.environ.get("RADIANCE_USE_R4D"),
                "calls": dict(self.calls), "blocked": dict(self.blocked),
            }
        with self.report_lock:
            temporary = self.path.with_suffix(".tmp")
            with temporary.open("w") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(report, handle, sort_keys=True)
            temporary.replace(self.path)

    def start_reporter(self):
        self.write()

        def report():
            while True:
                time.sleep(1)
                self.write()

        threading.Thread(target=report, name="r4d-dispatch-audit", daemon=True).start()

    def after_fork(self):
        self.reset_process()
        self.start_reporter()


_audit = None


def bind_r4d(native, mode, root):
    observed = {name for name in dir(native) if inspect.isbuiltin(getattr(native, name))}
    if observed != GPU_EXPORTS | METADATA_EXPORTS:
        raise ValueError("R4D native export set changed")
    digest = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if digest != NATIVE_SHA256:
        raise ValueError("R4D native binary changed")
    if not {entry["name"] for entry in native.kernels()} <= GPU_EXPORTS:
        raise ValueError("R4D registry has an unaudited kernel")
    audit = DispatchAudit(mode, root, GPU_EXPORTS, digest)
    for name in GPU_EXPORTS:
        setattr(native, name, audit.wrap(name, getattr(native, name)))
    return audit


def install_from_environment():
    global _audit
    mode = os.environ.get("QWEN_R4D_AUDIT_MODE")
    if not mode or _audit is not None:
        return
    # site.py normally swallows .pth exceptions. An invalid audit must prevent
    # the experimental process from serving unaudited requests instead.
    try:
        source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if source_digest != os.environ["QWEN_R4D_AUDIT_SHA256"]:
            raise ValueError("audit source changed")
        already_bound = [name for name in sys.modules
                         if name.startswith(("radiance_", "vllm.model_executor"))]
        if already_bound:
            raise ValueError("audit was installed after model modules")
        # Load libtorch before its pybind extension; no CUDA call.
        importlib.import_module("torch")
        r4d = importlib.import_module("r4d")

        _audit = bind_r4d(r4d, mode, os.environ["QWEN_R4D_AUDIT_ROOT"])
        _audit.start_reporter()
        atexit.register(_audit.write)
        os.register_at_fork(after_in_child=_audit.after_fork)
    except Exception as error:
        print(json.dumps({"r4d_audit_install_failed": type(error).__name__}),
              file=sys.stderr, flush=True)
        os._exit(97)


def validate_reports(root, mode, require_generation=True, engine_pids=()):
    """Require observed dispatch for the on arm and none for the off arm."""
    rows = [json.loads(path.read_text()) for path in Path(root).glob("dispatch-*.json")]
    if not rows:
        raise ValueError("no process installed the R4D audit")
    if not set(engine_pids) <= {row.get("pid") for row in rows}:
        raise ValueError("an inference engine has no R4D dispatch attestation")
    totals = dict.fromkeys(sorted(GPU_EXPORTS), 0)
    for row in rows:
        if (row.get("schema") != "qwen-r4d-native-dispatch-v1"
                or row.get("mode") != mode or row.get("native_sha256") != NATIVE_SHA256
                or row.get("master_switch") != ("1" if mode == "count" else "0")
                or set(row.get("calls", {})) != GPU_EXPORTS
                or set(row.get("blocked", {})) != GPU_EXPORTS):
            raise ValueError("R4D dispatch attestation does not match the experiment")
        if any(row["blocked"].values()):
            raise ValueError("the R4D-off experiment attempted a native dispatch")
        for name, count in row["calls"].items():
            totals[name] += count
    if mode == "forbid" and any(totals.values()):
        raise ValueError("R4D kernels ran in the off arm")
    if mode == "count" and require_generation:
        for prefix in ("attn_prefill_", "attn_decode_", "gdn_chunk_scan_"):
            if not any(count for name, count in totals.items() if name.startswith(prefix)):
                raise ValueError("the R4D-on experiment did not exercise all required paths")
    return {"mode": mode, "processes": len(rows), "engine_pids": list(engine_pids), "calls": totals,
            "blocked_calls": 0, "native_sha256": NATIVE_SHA256}
