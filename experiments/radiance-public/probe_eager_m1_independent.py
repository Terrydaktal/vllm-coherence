"""Small native counterexamples to eager-M1 independent arithmetic contracts.

Reads public checkpoint tensors and a source-pinned CPU audit bundle. Does not
load a model, read sessions, change the server, or install candidate kernels.
An idle check is observational, not an exclusive reservation of the Pi server.
Results are operator evidence, never a full-model correctness certificate.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def idle(api):
    with urllib.request.urlopen(api.rstrip("/") + "/metrics", timeout=3) as response:
        body = response.read().decode()
    counts = {"running": [], "waiting": []}
    for line in body.splitlines():
        match = re.fullmatch(
            r"vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+(\S+)(?:\s+\d+)?",
            line,
        )
        if match:
            counts[match[1]].append(float(match[2]))
    if any(
        not values or any(value != 0 for value in values) for values in counts.values()
    ):
        raise RuntimeError(
            "backend is busy or request counters are missing; probe stopped"
        )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def device_memory():
    devices = [
        path
        for path in Path("/sys/class/drm").glob("card*/device")
        if re.fullmatch(r"card\d+", path.parent.name)
        and (path / "vendor").read_text().strip() == "0x1002"
    ]
    if len(devices) != 1:
        raise RuntimeError("requires exactly one AMD GPU")
    path = devices[0]
    total = int((path / "mem_info_vram_total").read_text())
    used = int((path / "mem_info_vram_used").read_text())
    return total, total - used


def compare(torch, actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype or a.numel() == 0:
        raise ValueError("comparison requires identical nonempty representations")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("non-finite values cannot establish numerical agreement")
    difference = a.float() - b.float()
    return {
        "different": int((a != b).sum()),
        "elements": a.numel(),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(
            difference.double().norm() / b.double().norm().clamp_min(1e-30)
        ),
    }


def checkpoint_reader(root, expected):
    import torch
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def tensor(name):
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        if name in expected:
            row = expected[name]
            sha = hashlib.sha256(
                value.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if sha != row["sha256"] or list(value.shape) != row["shape"]:
                raise RuntimeError(f"checkpoint tensor differs from audit: {name}")
        return value

    return tensor


def decode_mxfp4_rows(torch, packed, scales):
    """Independent CPU oracle: exact E2M1 codes times stored E8M0 scales."""
    if packed.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("the independent decoding oracle runs on CPU")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("packed weights and scale bytes must be uint8")
    if packed.ndim != 2 or scales.shape != (packed.shape[0], packed.shape[1] // 16):
        raise ValueError("expected one scale per 32 decoded coefficients")
    if packed.shape[1] % 16 or (scales == 255).any():
        raise ValueError("unsupported partial block or non-finite E8M0 scale")
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    codes = torch.stack([packed & 15, packed >> 4], -1).reshape(packed.shape[0], -1)
    return levels[codes.long()] * torch.exp2(scales.double() - 127).repeat_interleave(
        32, -1
    )


def softplus_probe(args, torch, tensor):
    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as native

    source = Path(native.__file__)
    expected_sha = args.source_hashes[
        "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"
    ]
    if digest(source) != expected_sha:
        raise RuntimeError("installed packed recurrence differs from audited source")
    text = source.read_text()
    old = "tl.log(1.0 + tl.exp(x))"
    if text.count(old) != 1:
        raise RuntimeError(
            "softplus experiment requires exactly one source replacement"
        )
    patched = args.output / "isolated_recurrent_log1p.py"
    patched.write_text(text.replace(old, "tl.extra.libdevice.log1p(tl.exp(x))"))
    candidate = load_module(
        patched, "vllm.third_party.flash_linear_attention.ops.isolated_recurrent_log1p"
    )
    prefix = "model.language_model.layers.12.linear_attn."
    al = tensor(prefix + "A_log").float()
    bias = tensor(prefix + "dt_bias").bfloat16()
    a = (-18.0 - bias.float()).bfloat16().view(1, 48)
    b = torch.zeros_like(a)
    mixed = torch.zeros((1, 10240), dtype=torch.bfloat16)
    mixed[0, torch.arange(16) * 128] = 1
    initial = torch.zeros((2, 48, 128, 128), dtype=torch.float32)
    initial[1, :, :, 0] = 1
    x = a.float()[0] + bias.float()
    log_decay = -al.double().exp() * torch.log1p(x.double().exp())
    expected_decay = log_decay.exp().float()
    inputs = [item.cuda() for item in (mixed, a, b, al, bias)]
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    state = initial.cuda()
    out = torch.empty((1, 1, 48, 128), dtype=torch.bfloat16, device="cuda")

    def launch(module):
        module.fused_recurrent_gated_delta_rule_packed_decode(
            *inputs, 128**-0.5, state, out, indices, True
        )

    records = {}
    for name, module in (("installed", native), ("isolated_log1p", candidate)):
        idle(args.api)
        state.copy_(initial)
        launch(module)
        torch.cuda.synchronize()
        first = state[1, :, 0, 0].cpu()
        observed = {
            "single_step": compare(torch, first, expected_decay),
            "head29_decay": float(first[29]),
            "head29_expected": float(expected_decay[29]),
        }
        # An explicit zero-K/V transition isolates retention without pretending
        # this is the distribution of gates in a real conversation.
        for step in range(1, 2048):
            if step % 64 == 0:
                idle(args.api)
            launch(module)
        torch.cuda.synchronize()
        observed["head29_state_after_2048"] = float(state[1, 29, 0, 0])
        observed["head29_output_after_2048"] = float(out[0, 0, 29, 0])
        records[name] = observed
    records.update(
        {
            "scope": "48 actual parameter pairs; synthetic zero-key/value transitions",
            "native_source_sha256": digest(source),
            "candidate_source_sha256": digest(patched),
            "steps": 2048,
            "head29_a": float(a[0, 29]),
            "head29_x": float(x[29]),
            "head29_ideal_state_after_2048": float(log_decay[29].mul(2048).exp()),
        }
    )
    # Probe the cancellation boundary and ordinary gates with the same real
    # parameter pairs. An isolated state element again equals the decay factor.
    sweep = []
    for requested in (-24, -20, -18, -17, -16, -14, -12, -8, -4, 0, 4, 16, 24):
        idle(args.api)
        gate = (requested - bias.float()).bfloat16().view(1, 48)
        inputs[1].copy_(gate)
        effective = gate.double()[0] + bias.double()
        expected = (
            (-al.double().exp() * torch.nn.functional.softplus(effective)).exp().float()
        )
        row = {"requested_x": requested}
        for name, module in (("installed", native), ("isolated_log1p", candidate)):
            state.copy_(initial)
            launch(module)
            observed = state[1, :, 0, 0].cpu()
            row[name] = compare(torch, observed, expected)
            row[name]["decay_one_instead_of_below_one"] = int(
                ((observed == 1) & (expected < 1)).sum()
            )
        sweep.append(row)
    records["gate_sweep"] = sweep
    return records


def gemm_probe(args, torch, tensor):
    import numpy as np

    import radiance_mxfp4 as wrapper

    build = json.loads((args.gemm_build / "build.json").read_text())
    binary = args.gemm_build / "candidate/radiance_mxfp4_fp8.so"
    if digest(binary) != build["variants"]["candidate"]["binary_sha256"]:
        raise RuntimeError("GEMM binary differs from build metadata")
    if (
        digest(args.gemm_build / "build.json")
        != args.samples["performance_config"]["gemm_dispatch"]["build_sha256"]
    ):
        raise RuntimeError("GEMM build differs from current CPU audit")
    ext = load_module(binary, "eager_m1_probe.radiance_mxfp4_fp8")
    base = "model.language_model.layers.15.self_attn."
    weights = torch.cat(
        [tensor(base + part + ".weight") for part in ("q_proj", "k_proj", "v_proj")]
    )
    q_count = tensor(base + "q_proj.weight").shape[0]
    if (
        hashlib.sha256(weights[:q_count].numpy().tobytes()).hexdigest()
        != args.samples["fold_blocks"]["weight_sha256"]
    ):
        raise RuntimeError("checkpoint projection differs from audited weights")
    scales = torch.cat(
        [
            tensor(base + part + ".weight_scale")
            for part in ("q_proj", "k_proj", "v_proj")
        ]
    )
    if (
        hashlib.sha256(scales[:q_count].numpy().tobytes()).hexdigest()
        != args.samples["fold_blocks"]["scale_sha256"]
    ):
        raise RuntimeError("checkpoint projection scales differ from audit")
    n, halfk = weights.shape
    k = halfk * 2
    selected = [9544, 10568, 10711, 12048, 0, 9543]
    exact = decode_mxfp4_rows(torch, weights[selected], scales[selected])
    weight_fp64 = exact.numpy()
    packed = wrapper.permute_w(weights, n, k).cuda()
    ws = scales.T.contiguous().cuda()
    ref = scales.amax(1).contiguous().cuda()
    # The actual TP1 fused qkv shape selects KS=1 and requires no partial slab.
    if n != 14336 or k != 5120:
        raise RuntimeError("unexpected TP1 projection geometry")
    ext.set_decode_scratch(0, 0, 0)
    results = {
        "binary_sha256": digest(binary),
        "M": 1,
        "N": n,
        "K": k,
        "selected_rows": selected,
    }
    # Every affected checkpoint block contributes all its input columns. For a
    # one-hot input the reference is the stored coefficient, without reduction
    # order, FMA, softmax or model-reference ambiguity.
    columns = sorted(
        {
            block["block"] * 32 + j
            for block in args.samples["fold_blocks"]["blocks"]
            for j in range(32)
        }
    )
    acts = torch.zeros((len(columns), k), dtype=torch.float32)
    acts[torch.arange(len(columns)), columns] = 1
    generator = torch.Generator(device="cpu").manual_seed(20260924)
    random = torch.randn((320, k), generator=generator).bfloat16().float()
    for label, source in (("one_hot", acts), ("synthetic_normal", random)):
        idle(args.api)
        scale = (source.abs().amax(-1) / 448).clamp_min(1 / (448 * 512))
        qcpu = (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        decoded = qcpu.float().double() * scale.double()[:, None]
        expected = torch.from_numpy(
            np.asarray(decoded.numpy() @ weight_fp64.T)
        ).bfloat16()
        qgpu, sgpu = qcpu.cuda(), scale.cuda()
        output = torch.empty((source.shape[0], n), dtype=torch.bfloat16, device="cuda")
        for start in range(source.shape[0]):
            if start % 32 == 0:
                idle(args.api)
            ext.launch(
                qgpu[start:].data_ptr(),
                packed.data_ptr(),
                ws.data_ptr(),
                ref.data_ptr(),
                sgpu[start:].data_ptr(),
                output[start:].data_ptr(),
                1,
                n,
                k,
                torch.cuda.current_stream().cuda_stream,
            )
        torch.cuda.synchronize()
        actual = output[:, selected].cpu()
        results[label] = {
            "inputs": source.shape[0],
            "projection": compare(torch, actual, expected),
            "affected_gate_rows": compare(torch, actual[:, :4], expected[:, :4]),
            "control_rows": compare(torch, actual[:, 4:], expected[:, 4:]),
            "after_bf16_sigmoid": compare(
                torch, actual[:, :4].sigmoid(), expected[:, :4].sigmoid()
            ),
        }
        if label == "one_hot":
            row = columns.index(170)
            results[label]["witness"] = {
                "row": 9544,
                "column": 170,
                "native": float(actual[row, 0]),
                "checkpoint": float(expected[row, 0]),
            }
        del qgpu, sgpu, output
    # The existing wref=0 route is a per-block-scaled control, not an installed
    # repair. Its original kernel expects checkpoint-order weights. Compare both
    # layouts explicitly: the normal WPERM loader also permutes per-block layers.
    idle(args.api)
    control_columns = [0, 170, 172, 173]
    x = torch.zeros((len(control_columns), k), dtype=torch.float32)
    x[torch.arange(len(control_columns)), control_columns] = 1
    x = x.to(torch.float8_e4m3fn).cuda()
    xs = torch.ones(len(control_columns), dtype=torch.float32, device="cuda")
    raw = weights.cuda()
    expected = exact[:, control_columns].T.bfloat16()
    results["perblock_layout_control"] = {}
    for label, w in (("checkpoint_order", raw), ("fragment_order", packed)):
        out = torch.empty(
            (len(control_columns), n), dtype=torch.bfloat16, device="cuda"
        )
        for i in range(len(control_columns)):
            ext.launch(
                x[i:].data_ptr(),
                w.data_ptr(),
                ws.data_ptr(),
                0,
                xs[i:].data_ptr(),
                out[i:].data_ptr(),
                1,
                n,
                k,
                torch.cuda.current_stream().cuda_stream,
            )
        torch.cuda.synchronize()
        results["perblock_layout_control"][label] = compare(
            torch, out[:, selected], expected
        )
    return results


def norm_probe(args, torch, tensor):
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as native

    if (
        digest(native.__file__)
        != args.source_hashes[
            "vllm/third_party/flash_linear_attention/ops/layernorm_guard.py"
        ]
    ):
        raise RuntimeError("GDN norm source changed")
    generator = torch.Generator(device="cpu").manual_seed(20260925)
    aggregate = {"different": 0, "elements": 0, "max_abs": 0.0}
    native_error = hf_error = denominator = 0.0
    per_layer = []
    for layer in range(64):
        if layer % 4 == 3:
            continue
        idle(args.api)
        weight = tensor(
            f"model.language_model.layers.{layer}.linear_attn.norm.weight"
        ).bfloat16()
        x = torch.randn((32, 48, 128), generator=generator).bfloat16()
        z = torch.randn((32, 48, 128), generator=generator).bfloat16()
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + 1e-6
        )
        expected = (weight * normalized.bfloat16()).float() * torch.nn.functional.silu(
            z.float()
        )
        expected = expected.bfloat16()
        xd, zd, wd = x.cuda(), z.cuda(), weight.cuda()
        outputs = []
        for i in range(32):
            outputs.append(
                native.layer_norm_fwd(
                    xd[i],
                    wd,
                    None,
                    1e-6,
                    z=zd[i],
                    norm_before_gate=True,
                    is_rms_norm=True,
                    activation="silu",
                )[0]
            )
        actual = torch.stack(outputs).cpu()
        row = compare(torch, actual, expected)
        aggregate["different"] += row["different"]
        aggregate["elements"] += row["elements"]
        aggregate["max_abs"] = max(aggregate["max_abs"], row["max_abs"])
        oracle = x.double() * torch.rsqrt(
            x.double().square().mean(-1, keepdim=True) + 1e-6
        )
        oracle = oracle * weight.double() * torch.nn.functional.silu(z.double())
        native_error += float((actual.double() - oracle).square().sum())
        hf_error += float((expected.double() - oracle).square().sum())
        denominator += float(oracle.square().sum())
        per_layer.append({"layer": layer, **row})
        del xd, zd, wd, outputs
    return {
        "native_vs_plain_hf": aggregate,
        "synthetic_positions_per_layer": 32,
        "layers": len(per_layer),
        "per_layer": per_layer,
        "native_relative_l2_vs_fp64_formula": math.sqrt(native_error / denominator),
        "hf_relative_l2_vs_fp64_formula": math.sqrt(hf_error / denominator),
        "meaning": "HF finite-precision contract differs; FP64 comparison does not prove model quality",
        "native_source_sha256": digest(native.__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gemm-build", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("softplus", "gemm", "norm"),
        default=["softplus", "gemm", "norm"],
    )
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise RuntimeError("requires explicit GPU admission and shared lease path")
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    args.samples = json.loads((args.audit / "checkpoint-samples.json").read_text())
    args.source_hashes = json.loads((args.audit / "sources.json").read_text())
    with open(os.environ["QWEN_CONFORMANCE_GPU_LOCK"], "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_admitted(args)


def run_admitted(args):
    idle(args.api)
    total, free = device_memory()
    if free < 512 * 1024**2:
        raise RuntimeError("less than 512 MiB free; no GPU initialization attempted")
    os.environ["TRITON_CACHE_DIR"] = str(args.output / "triton-cache")
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
    for key, value in {
        "RADIANCE_MXFP4": "1",
        "RADIANCE_MXFP4_W4A8": "1",
        "RADIANCE_MXFP4_WPERM": "1",
        "RADIANCE_MXFP4_DECODE_MAX_M": "64",
        "RADIANCE_MXFP4_DECODE_NT": "1",
        "RADIANCE_MXFP4_W4A8_MIN_M": "0",
    }.items():
        os.environ[key] = value
    import torch

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(192 * 1024**2 / total)
    tensor = checkpoint_reader(args.model, args.samples["small_tensors"])
    report = {
        "status": "RUNNING",
        "started_unix": time.time(),
        "source_sha256": digest(__file__),
        "audit_inputs_sha256": {
            name: digest(args.audit / name)
            for name in ("checkpoint-samples.json", "sources.json")
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "scope": "installed eager M1 operators on public weights and synthetic inputs",
        "comparison": "finite numeric element equality, not a bitwise or full-model certificate",
        "private_chat_read": False,
        "server_restarted": False,
        "initial_free_MiB": free / 1024**2,
        "allocator_budget_MiB": 192,
        "checks": {},
    }
    destination = args.output / "result.json"
    try:
        for name, function in (
            ("softplus", softplus_probe),
            ("gemm", gemm_probe),
            ("norm", norm_probe),
        ):
            if name not in args.stages:
                continue
            report["current_stage"] = name
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"stage": name, "status": "starting"}), flush=True)
            report["checks"][name] = function(args, torch, tensor)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"stage": name, "status": "recorded"}), flush=True)
        report["status"] = "OPERATOR_INVESTIGATION_COMPLETE_NOT_MODEL_QUALIFICATION"
    except BaseException as exc:
        report["status"] = "INCOMPLETE"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["allocator_peak_MiB"] = torch.cuda.max_memory_allocated() / 1024**2
        report["elapsed_seconds"] = time.time() - report["started_unix"]
        destination.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
