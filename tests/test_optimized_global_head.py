"""Exercise the installed target head's gating and repaired fallback on CPU."""

import ast
import io
import types
from pathlib import Path
from types import SimpleNamespace as Ns

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "experiments/radiance-public/radiance_verifyhead_global.py"
)


def functions():
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = {
        "GLOBAL_TOPK": 256,
        "MAX_ROWS": 32,
        "_GLOBAL_MAX_ROWS": 32,
        "_NO_LOGPROBS": -1,
        "types": types,
        "sys": Ns(stderr=io.StringIO()),
        "_state": {"failed": False, "armed": False, "full_calls": 0},
        "_dh": Ns(RERANK=80, KCAND=8),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def batch():
    ss = Ns(
        temperature=Ns(np=np.array([0.6, 0.6, 0.0])),
        top_k=Ns(np=np.array([20, 1024, -1])),
        min_p=Ns(np=np.zeros(3)),
        num_logprobs=np.full(3, -1),
    )
    sampler = Ns(
        sampling_states=ss,
        penalties_state=Ns(use_penalty=np.zeros(3, dtype=bool)),
        logit_bias_state=Ns(use_logit_bias=np.zeros(3, dtype=bool)),
        bad_words_state=Ns(num_bad_words=Ns(np=np.zeros(3))),
        logprob_token_ids_state=Ns(num_token_ids=Ns(np=np.zeros(3))),
        thinking_budget_state=Ns(enabled=False),
    )
    return Ns(sampler=sampler), Ns(
        idx_mapping_np=np.array([2, 0]), num_reqs=2, logits_indices=np.zeros(9)
    )


def test_pi_top20_admitted_without_changing_drafter_capacity():
    ns = functions()
    runner, request = batch()
    assert ns["_batch_is_safe"](runner, request, None)
    assert ns["_dh"].KCAND == 8
    runner.sampler.penalties_state.use_penalty[1] = True  # unselected request
    assert ns["_batch_is_safe"](runner, request, None)


@pytest.mark.parametrize(
    "restriction", ["grammar", "bias", "penalty", "logprobs", "min_p", "top_k", "rows", "layout"]
)
def test_unsupported_sampling_uses_corrected_full_head(restriction):
    ns = functions()
    runner, request = batch()
    sampler = runner.sampler
    grammar = None
    if restriction == "grammar":
        grammar = object()
    elif restriction == "bias":
        sampler.logit_bias_state.use_logit_bias[0] = True
    elif restriction == "penalty":
        sampler.penalties_state.use_penalty[0] = True
    elif restriction == "logprobs":
        sampler.sampling_states.num_logprobs[0] = 0
    elif restriction == "min_p":
        sampler.sampling_states.min_p.np[0] = 0.1
    elif restriction == "top_k":
        sampler.sampling_states.top_k.np[0] = 100
    elif restriction == "rows":
        request.logits_indices = np.zeros(33)
    else:
        del sampler.penalties_state
    assert not ns["_batch_is_safe"](runner, request, grammar)


@pytest.mark.parametrize("quantize_failure", [False, True])
def test_arm_preserves_instance_repair_instead_of_original_class_head(quantize_failure):
    ns = functions()

    class LP:
        def _apply_head(self, *_args):
            raise AssertionError("unrepaired class method was called")

    lp = LP()

    def repaired(*_args):
        return "corrected M4-pair result"

    lp._apply_head = repaired
    weight = Ns(dim=lambda: 2, data=Ns(abs=lambda: Ns(max=lambda: 1.0)))
    ns["_find_target_lp"] = lambda _: (lp, Ns(weight=weight))

    def quantize(target, _head):
        target._apply_head = lambda *_: "temporary approximate head"
        if quantize_failure:
            raise RuntimeError("injected allocation failure")
        target._radiance_wq = object()
        return "packed"

    ns["_dh"]._quantize_head_now = quantize
    if quantize_failure:
        with pytest.raises(RuntimeError, match="injected"):
            ns["_arm"](None)
        assert lp._apply_head is repaired
    else:
        ns["_arm"](None)
        assert lp._radiance_exact_head is repaired
        assert lp._apply_head(None, None) == "corrected M4-pair result"
