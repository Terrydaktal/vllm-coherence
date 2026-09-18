"""Pinned native FP8 producers; the usual BF16 consumer remains the fallback."""

import ctypes
import hashlib
import json
from pathlib import Path


class StockFP8Epilogue:
    def __init__(self, build):
        self.build = Path(build)
        self.manifest = json.loads((self.build / "build.json").read_text())
        library = self.build / "candidate.so"
        if (
            self.manifest["status"] != "BUILT_UNTESTED"
            or self.manifest["abi"] != "qwen-stock-fp8-epilogue-v1"
            or hashlib.sha256(library.read_bytes()).hexdigest() != self.manifest["binary_sha256"]
        ):
            raise ValueError("FP8 epilogue binary does not match its build")
        self.library = ctypes.CDLL(str(library))
        self.norm_launch = self.library.qwen_stock_fp8_norm
        self.norm_launch.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        self.norm_launch.restype = ctypes.c_int
        self.silu_launch = self.library.qwen_stock_fp8_silu
        self.silu_launch.argtypes = [ctypes.c_void_p] * 3 + [
            ctypes.c_int,
            ctypes.c_long,
            ctypes.c_void_p,
        ]
        self.silu_launch.restype = ctypes.c_int

    @staticmethod
    def admit(x, width, *, max_rows=8):
        import torch

        if (
            x.ndim != 2
            or not 1 <= x.shape[0] <= max_rows
            or x.shape[1] != width
            or x.dtype != torch.bfloat16
            or x.device.type != "cuda"
            or x.stride(1) != 1
        ):
            raise ValueError("FP8 producer requires its admitted BF16 layout")

    def norm(self, x, residual, weight, epsilon):
        import torch

        self.admit(x, 5120, max_rows=2048)
        if (
            weight.shape != (5120,)
            or weight.dtype != x.dtype
            or weight.device != x.device
            or not weight.is_contiguous()
        ):
            raise ValueError("FP8 producer norm weight changed")
        if residual is not None:
            self.admit(residual, 5120, max_rows=2048)
            if residual.shape != x.shape or residual.device != x.device:
                raise ValueError("FP8 residual changed")
        q = torch.empty_like(x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format)
        scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        carry = (
            torch.empty_like(x, memory_format=torch.contiguous_format)
            if residual is not None
            else None
        )
        rc = self.norm_launch(
            x.data_ptr(),
            residual.data_ptr() if residual is not None else None,
            weight.data_ptr(),
            q.data_ptr(),
            scale.data_ptr(),
            carry.data_ptr() if carry is not None else None,
            x.shape[0],
            x.stride(0),
            residual.stride(0) if residual is not None else 0,
            epsilon,
            torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError(f"FP8 norm launch failed: {rc}")
        return q, scale, carry

    def silu(self, gu):
        import torch

        self.admit(gu, 34816)
        q = torch.empty((gu.shape[0], 17408), device=gu.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((gu.shape[0], 1), device=gu.device, dtype=torch.float32)
        rc = self.silu_launch(
            gu.data_ptr(),
            q.data_ptr(),
            scale.data_ptr(),
            gu.shape[0],
            gu.stride(0),
            torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError(f"FP8 SiLU launch failed: {rc}")
        return q, scale
