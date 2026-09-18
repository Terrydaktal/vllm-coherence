"""Real child-process checks: offline commands cannot initialize a GPU runtime."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_all_conformance_modules_import_and_prove_with_gpu_imports_forbidden(tmp_path):
    source = r"""
import importlib.abc
import importlib
import pathlib
import sys
class ForbidGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'cupy', 'pynvml', 'pycuda'}:
            raise AssertionError('GPU library imported by offline conformance: ' + fullname)
sys.meta_path.insert(0, ForbidGPU())
import qwen_r9700_lab
for path in pathlib.Path(qwen_r9700_lab.__file__).parent.glob('conformance_*.py'):
    importlib.import_module('qwen_r9700_lab.' + path.stem)
from qwen_r9700_lab.conformance_proofs import run_obligations
assert run_obligations(pathlib.Path(sys.argv[1]))['all_expected_results']
print('CPU-only imports and proofs completed')
"""
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path / "proofs")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "QWEN_CONFORMANCE_GPU": "0",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CPU-only imports and proofs completed"


@pytest.mark.parametrize("quiescent,armed", [(False, "0"), (True, "0"), (False, "1")])
def test_recovery_rpc_requires_gpu_authorization_and_quiescent_boundary(
    monkeypatch, quiescent, armed
):
    from qwen_r9700_lab.conformance_radiance import capture_committed_state
    from qwen_r9700_lab.diagnostic_contract import DiagnosticError

    monkeypatch.setenv("QWEN_CONFORMANCE_GPU", armed)
    with pytest.raises(DiagnosticError, match="armed, quiescent"):
        capture_committed_state(
            None,
            output_path="unused",
            request_id="unused",
            expected={},
            plan_path="unused",
            binding_path="unused",
            quiescent=quiescent,
        )
