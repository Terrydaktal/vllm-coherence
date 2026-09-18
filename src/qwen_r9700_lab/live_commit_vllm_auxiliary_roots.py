"""Branch-private vLLM runner state for serial-before-commit validation.

The KV/GDN tensors are owned by :mod:`live_commit_vllm_physical_roots`.
This module owns the other persistent state which a decode step mutates: request
tokens and counters, sampler/penalty/thinking state, DFlash proposals, and the
host Quest continuity identity.  The supported production ABI is intentionally
narrow and fail closed: TP=1, max-num-seqs=1, synchronous text-only generation,
greedy sampling, and ``mamba-cache-mode=none``.

The model runner remains scratch space.  Durable branch state lives in snapshots
keyed by the same ``AtomicRootTableBank`` root identifier as the physical cache.
Consequently publication still requires one pointer change.  Entering a private
arm restores its snapshot; leaving captures the completed snapshot and restores
the canonical scratch state even when execution raises.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from qwen_r9700_lab.live_commit_scheduler_authority import PhysicalRoot
from qwen_r9700_lab.live_commit_vllm_root_tables import (
    TOTAL_GROUPS,
    AtomicRootTableBank,
)

AUXILIARY_ROOT_SCHEMA = "urn:qwen-r9700:vllm-auxiliary-root:v1"


class VllmAuxiliaryRootError(RuntimeError):
    """Runner state cannot satisfy the fixed private-root ABI."""


def _resolve(parent: object, path: str) -> object:
    value = parent
    for component in path.split("."):
        if not hasattr(value, component):
            raise VllmAuxiliaryRootError(f"runner ABI omits {path}")
        value = getattr(value, component)
    return value


def _clone(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    return copy.deepcopy(value)


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return np.asarray(value)
    current = value
    for operation in ("detach", "contiguous", "cpu"):
        method = getattr(current, operation, None)
        if callable(method):
            current = method()
    method = getattr(current, "numpy", None)
    if callable(method):
        result = method()
        if isinstance(result, np.ndarray):
            return result
    raise VllmAuxiliaryRootError("runner tensor cannot be materialized")


def _equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(_numpy(left), _numpy(right), equal_nan=True))
        except TypeError:  # NumPy versions before equal_nan for some dtypes.
            return bool(np.array_equal(_numpy(left), _numpy(right)))
    if hasattr(left, "numpy") or hasattr(left, "detach"):
        try:
            return bool(np.array_equal(_numpy(left), _numpy(right), equal_nan=True))
        except (TypeError, VllmAuxiliaryRootError):
            return False
    return left == right


def _digest(value: object) -> str:
    if isinstance(value, np.ndarray) or hasattr(value, "detach") or hasattr(value, "numpy"):
        array = np.ascontiguousarray(_numpy(value))
        payload = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
        }
    elif isinstance(value, np.generic):
        payload = value.item()
    else:
        payload = value
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _Binding:
    label: str
    capture: Callable[[], object]
    restore: Callable[[object], None]


@dataclass(frozen=True)
class AuxiliaryRootSnapshot:
    schema: str
    root_id: str
    values: tuple[tuple[str, object], ...]

    def value_map(self) -> dict[str, object]:
        return dict(self.values)


def _tensor_binding(owner: object, attribute: str, index: object, label: str) -> _Binding:
    tensor = _resolve(owner, attribute)

    def capture() -> object:
        return _clone(tensor[index])  # type: ignore[index]

    def restore(value: object) -> None:
        destination = tensor[index]  # type: ignore[index]
        copier = getattr(destination, "copy_", None)
        if callable(copier):
            copier(value)
        else:
            tensor[index] = value  # type: ignore[index]

    capture()
    return _Binding(label, capture, restore)


def _attribute_binding(owner: object, attribute: str, label: str) -> _Binding:
    if not hasattr(owner, attribute):
        raise VllmAuxiliaryRootError(f"{label} is absent")

    def capture() -> object:
        return _clone(getattr(owner, attribute))

    def restore(value: object) -> None:
        setattr(owner, attribute, _clone(value))

    capture()
    return _Binding(label, capture, restore)


def _numpy_binding(owner: object, attribute: str, index: object, label: str) -> _Binding:
    array = _resolve(owner, attribute)
    if not isinstance(array, np.ndarray):
        raise VllmAuxiliaryRootError(f"{label} is not a NumPy array")

    def capture() -> object:
        return np.asarray(array[index]).copy()

    def restore(value: object) -> None:
        array[index] = value

    return _Binding(label, capture, restore)


def _uva_binding(owner: object, attribute: str, index: object, label: str) -> _Binding:
    buffer = _resolve(owner, attribute)
    source = getattr(buffer, "np", None)
    gpu = getattr(buffer, "gpu", None)
    copy_to_uva = getattr(buffer, "copy_to_uva", None)
    if not isinstance(source, np.ndarray) or gpu is None or not callable(copy_to_uva):
        raise VllmAuxiliaryRootError(f"{label} is not a UVA-backed tensor")

    def capture() -> object:
        source_value = np.asarray(source[index]).copy()
        gpu_value = _clone(gpu[index])
        if not _equal(source_value, gpu_value):
            raise VllmAuxiliaryRootError(f"{label} CPU/UVA mirrors differ")
        return source_value

    def restore(value: object) -> None:
        source[index] = value
        buffer.copy_to_uva()

    capture()
    return _Binding(label, capture, restore)


def _staged_binding(owner: object, attribute: str, index: object, label: str) -> _Binding:
    buffer = _resolve(owner, attribute)
    gpu = getattr(buffer, "gpu", None)
    if gpu is None:
        raise VllmAuxiliaryRootError(f"{label} is not a staged tensor")

    def require_quiescent() -> None:
        for pending in (
            "_staged_write_indices",
            "_staged_write_starts",
            "_staged_write_contents",
            "_staged_write_cu_lens",
        ):
            value = getattr(buffer, pending, None)
            if not isinstance(value, list) or value:
                raise VllmAuxiliaryRootError(f"{label} has pending staged writes")

    def capture() -> object:
        require_quiescent()
        return _clone(gpu[index])

    def restore(value: object) -> None:
        require_quiescent()
        destination = gpu[index]
        copier = getattr(destination, "copy_", None)
        if callable(copier):
            copier(value)
        else:
            gpu[index] = value

    capture()
    return _Binding(label, capture, restore)


class VllmRunnerAuxiliaryRoots:
    """Exact snapshots of every supported persistent per-request runner field."""

    def __init__(
        self,
        runner: object,
        *,
        request_id: str,
        roots: Sequence[PhysicalRoot],
        table_bank: AtomicRootTableBank,
        synchronize_device: Callable[[], None],
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise VllmAuxiliaryRootError("request ID is invalid")
        if not callable(synchronize_device):
            raise VllmAuxiliaryRootError("device synchronizer is invalid")
        if table_bank.request_id != request_id:
            raise VllmAuxiliaryRootError("table bank request differs")
        self.runner = runner
        self.request_id = request_id
        self.table_bank = table_bank
        self._synchronize_device = synchronize_device
        self._roots = {root.root_id: root for root in roots}
        if len(self._roots) != 3 or set(self._roots) != set(table_bank.root_ids):
            raise VllmAuxiliaryRootError("auxiliary and physical root sets differ")
        self._validate_fixed_abi()
        req_states = _resolve(runner, "req_states")
        req_map = getattr(req_states, "req_id_to_index", None)
        if req_map != {request_id: 0}:
            raise VllmAuxiliaryRootError("runner request-slot map is not exact single-sequence")
        self.req_index = 0
        self._bindings = self._build_bindings()
        labels = [binding.label for binding in self._bindings]
        if len(labels) != len(set(labels)):
            raise VllmAuxiliaryRootError("auxiliary binding labels are not unique")
        self._snapshots: dict[str, AuxiliaryRootSnapshot] = {}
        self._quest_tokens = {root_id: object() for root_id in self._roots}

    def _validate_fixed_abi(self) -> None:
        runner = self.runner
        req_states = _resolve(runner, "req_states")
        if getattr(req_states, "max_num_reqs", None) != 1:
            raise VllmAuxiliaryRootError("runner max-num-seqs is not one")
        if getattr(req_states, "num_speculative_steps", None) != 7:
            raise VllmAuxiliaryRootError("runner is not the DFlash7 lane")
        for attribute in ("pooling_runner", "pp_handler", "encoder_cache", "adaptive_verification"):
            if getattr(runner, attribute, None) is not None:
                raise VllmAuxiliaryRootError(f"unsupported runner state is active: {attribute}")
        model_state = _resolve(runner, "model_state")
        if getattr(model_state, "_align_mode", None) is not False:
            raise VllmAuxiliaryRootError("runner is not mamba-cache-mode=none")
        if getattr(model_state, "rope_state", None) is not None:
            raise VllmAuxiliaryRootError("non-canonical RoPE request state is active")
        if getattr(model_state, "prompt_embeds_state", None) is not None:
            raise VllmAuxiliaryRootError("prompt-embedding request state is active")
        if not isinstance(getattr(runner, "_qwen_quest_request_tokens", None), dict):
            raise VllmAuxiliaryRootError("Quest request continuity is unavailable")
        if not isinstance(getattr(runner, "_qwen_quest_block_tables", None), dict):
            raise VllmAuxiliaryRootError("Quest block continuity is unavailable")
        block_tables = _resolve(runner, "block_tables")
        if len(getattr(block_tables, "block_tables", ())) != TOTAL_GROUPS:
            raise VllmAuxiliaryRootError("worker block-table group count differs")
        sampler = _resolve(runner, "sampler")
        states = _resolve(sampler, "sampling_states")
        if float(states.temperature.np[0]) != 0.0:
            raise VllmAuxiliaryRootError("serial validation requires greedy sampling")
        if float(states.top_p.np[0]) != 1.0 or int(states.top_k.np[0]) != 1:
            raise VllmAuxiliaryRootError("serial validation sampling contract differs")

    def _build_bindings(self) -> tuple[_Binding, ...]:
        runner = self.runner
        req = _resolve(runner, "req_states")
        model = _resolve(runner, "model_state")
        sampler = _resolve(runner, "sampler")
        bindings: list[_Binding] = []

        bindings.extend(
            _staged_binding(req, attribute, 0, f"request.{attribute}")
            for attribute in ("all_token_ids", "total_len", "num_computed_tokens")
        )
        bindings.extend(
            _uva_binding(req, attribute, 0, f"request.{attribute}")
            for attribute in ("prompt_len", "prefill_len")
        )
        bindings.extend(
            _numpy_binding(req, attribute, 0, f"request.{attribute}")
            for attribute in (
                "num_computed_prefill_tokens",
                "num_computed_tokens_np",
                "max_seq_len",
            )
        )
        bindings.extend(
            _tensor_binding(req, attribute, index, f"request.{attribute}")
            for attribute, index in (
                ("last_sampled_tokens", 0),
                ("draft_tokens", 0),
                ("next_prefill_tokens", (slice(None), 0)),
            )
        )

        bindings.append(
            _tensor_binding(model, "num_accepted_tokens_gpu", 0, "model.num_accepted")
        )
        if getattr(model, "_qwen_recoverssm_precopy_accepted_gpu", None) is not None:
            raise VllmAuxiliaryRootError("align-only RecoverSSM state is unexpectedly active")
        recoverssm = getattr(model, "recoverssm", None)
        if recoverssm is not None and getattr(recoverssm, "_step", None) is not None:
            raise VllmAuxiliaryRootError("RecoverSSM step is not quiescent")

        # EngineCore obtains these proposals only after the selected state is
        # published.  They therefore belong to the branch root: otherwise the
        # serial validator's later execution overwrites the candidate result.
        # Streams/events are synchronization mechanisms rather than state; the
        # device is synchronized before every capture and the completed pinned
        # buffer is preserved here.
        bindings.extend(
            _attribute_binding(runner, attribute, f"draft_output.{attribute}")
            for attribute in (
                "_draft_token_ids",
                "_draft_probs",
                "_draft_prob_req_ids",
                "_draft_token_req_ids",
                "prev_num_spec_tokens",
            )
        )
        bindings.append(
            _tensor_binding(
                runner,
                "draft_token_ids_cpu",
                0,
                "draft_output.draft_token_ids_cpu",
            )
        )
        bindings.append(
            _tensor_binding(
                runner,
                "sampled_token_ids_pinned_cpu",
                0,
                "draft_output.sampled_token_ids_pinned_cpu",
            )
        )
        for attribute in (
            "_num_valid_draft_tokens",
            "_num_valid_draft_tokens_cpu",
            "valid_sampled_token_count_cpu",
        ):
            if getattr(runner, attribute, None) is not None:
                raise VllmAuxiliaryRootError(
                    f"unsupported asynchronous/ngram state is active: {attribute}"
                )

        sampling = _resolve(sampler, "sampling_states")
        bindings.extend(
            _uva_binding(sampling, attribute, 0, f"sampling.{attribute}")
            for attribute in ("temperature", "top_k", "top_p", "min_p", "seeds")
        )
        bindings.extend(
            _numpy_binding(sampling, attribute, 0, f"sampling.{attribute}")
            for attribute in ("seeds_set", "num_logprobs")
        )
        bindings.append(
            _numpy_binding(sampler, "needs_logits_processing", 0, "sampling.needs_processing")
        )

        penalties = _resolve(sampler, "penalties_state")
        if getattr(penalties, "_new_penalties_reqs", None) != []:
            raise VllmAuxiliaryRootError("penalty state has pending requests")
        bindings.extend(
            _uva_binding(penalties, attribute, 0, f"penalty.{attribute}")
            for attribute in (
                "repetition_penalty",
                "frequency_penalty",
                "presence_penalty",
            )
        )
        bindings.append(_numpy_binding(penalties, "use_penalty", 0, "penalty.enabled"))
        bindings.extend(
            _tensor_binding(penalties, attribute, 0, f"penalty.{attribute}")
            for attribute in ("prompt_bin_mask", "output_bin_counts")
        )

        bias = _resolve(sampler, "logit_bias_state")
        bindings.extend(
            _uva_binding(bias, attribute, 0, f"bias.{attribute}")
            for attribute in (
                "num_allowed_token_ids",
                "num_logit_bias",
                "min_lens",
                "num_stop_token_ids",
            )
        )
        bindings.extend(
            _staged_binding(bias, attribute, 0, f"bias.{attribute}")
            for attribute in (
                "allowed_token_ids",
                "logit_bias_token_ids",
                "logit_bias",
                "stop_token_ids",
            )
        )
        bindings.append(_numpy_binding(bias, "use_logit_bias", 0, "bias.enabled"))

        bad_words = _resolve(sampler, "bad_words_state")
        bindings.append(_uva_binding(bad_words, "num_bad_words", 0, "bad_words.count"))
        bindings.extend(
            _staged_binding(bad_words, attribute, 0, f"bad_words.{attribute}")
            for attribute in ("bad_word_token_ids", "bad_word_offsets")
        )

        logprobs = _resolve(sampler, "logprob_token_ids_state")
        bindings.append(_uva_binding(logprobs, "num_token_ids", 0, "logprobs.count"))
        bindings.append(_staged_binding(logprobs, "token_ids", 0, "logprobs.token_ids"))

        thinking = _resolve(sampler, "thinking_budget_state")
        if getattr(thinking, "enabled", None) is not True:
            raise VllmAuxiliaryRootError("thinking-budget state is not enabled")
        if getattr(thinking, "_reset_reqs", None) != [] or getattr(
            thinking, "_budget_dirty", None
        ) is not False:
            raise VllmAuxiliaryRootError("thinking-budget state has pending writes")
        bindings.append(
            _uva_binding(thinking, "thinking_token_budget", 0, "thinking.budget")
        )
        bindings.append(
            _numpy_binding(thinking, "use_thinking_budget", 0, "thinking.enabled")
        )
        bindings.extend(
            _tensor_binding(thinking, attribute, 0, f"thinking.{attribute}")
            for attribute in ("cached_last_start", "cached_last_end", "cached_scan_pos")
        )
        return tuple(bindings)

    def _capture(self, root_id: str) -> AuxiliaryRootSnapshot:
        values = tuple((binding.label, binding.capture()) for binding in self._bindings)
        return AuxiliaryRootSnapshot(AUXILIARY_ROOT_SCHEMA, root_id, values)

    def _restore(self, snapshot: AuxiliaryRootSnapshot) -> None:
        if snapshot.schema != AUXILIARY_ROOT_SCHEMA:
            raise VllmAuxiliaryRootError("auxiliary snapshot schema differs")
        values = snapshot.value_map()
        if set(values) != {binding.label for binding in self._bindings}:
            raise VllmAuxiliaryRootError("auxiliary snapshot binding set differs")
        for binding in self._bindings:
            binding.restore(values[binding.label])
        recoverssm = getattr(_resolve(self.runner, "model_state"), "recoverssm", None)
        if recoverssm is not None:
            recoverssm._step = None
        self._activate_block_tables(self._roots[snapshot.root_id])

    def _activate_block_tables(self, root: PhysicalRoot) -> None:
        groups = (
            tuple((slot,) for slot in root.gdn_slots)
            + root.target_blocks
            + root.draft_blocks
        )
        if len(groups) != TOTAL_GROUPS:
            raise VllmAuxiliaryRootError("root block group count differs")
        tables = _resolve(self.runner, "block_tables")
        staged = tables.block_tables
        blocks_per = tables.blocks_per_kv_block
        num_blocks = tables.num_blocks
        for group_index, (ids, multiplier) in enumerate(zip(groups, blocks_per, strict=True)):
            expanded = [
                block * multiplier + offset
                for block in ids
                for offset in range(multiplier)
            ]
            row = staged[group_index].gpu[0]
            if len(expanded) > int(row.shape[0]):
                raise VllmAuxiliaryRootError("root exceeds worker block-table capacity")
            destination = row[: len(expanded)]
            copier = getattr(destination, "copy_", None)
            if callable(copier):
                maker = getattr(row, "new_tensor", None)
                if not callable(maker):
                    raise VllmAuxiliaryRootError("worker block table cannot encode IDs")
                copier(maker(expanded))
            else:
                row[: len(expanded)] = expanded
            num_blocks.np[group_index, 0] = len(expanded)
        num_blocks.copy_to_uva()
        quest_tables = tuple(
            tuple(block * multiplier + offset for block in ids for offset in range(multiplier))
            for ids, multiplier in zip(groups, blocks_per, strict=True)
        )
        self.runner._qwen_quest_request_tokens[self.request_id] = self._quest_tokens[root.root_id]
        self.runner._qwen_quest_block_tables[self.request_id] = quest_tables

    def prepare_private_roots(
        self,
        canonical: PhysicalRoot,
        candidate: PhysicalRoot,
        serial: PhysicalRoot,
    ) -> None:
        if self.table_bank.canonical_root_id != canonical.root_id:
            raise VllmAuxiliaryRootError("physical canonical root changed before preparation")
        prior = self._snapshots.get(canonical.root_id)
        if prior is not None:
            # The runner is scratch space and may still show the previous root
            # after cancellation/recovery.  The root pointer is authoritative.
            self._restore(prior)
        self._synchronize_device()
        base = self._capture(canonical.root_id)
        self._snapshots = {
            canonical.root_id: base,
            candidate.root_id: AuxiliaryRootSnapshot(
                AUXILIARY_ROOT_SCHEMA,
                candidate.root_id,
                tuple((label, _clone(value)) for label, value in base.values),
            ),
            serial.root_id: AuxiliaryRootSnapshot(
                AUXILIARY_ROOT_SCHEMA,
                serial.root_id,
                tuple((label, _clone(value)) for label, value in base.values),
            ),
        }

    def publish_atomic(self, *, expected_canonical: str, selected_root: str) -> str:
        """Restore selected scratch, then publish all roots with one pointer write."""

        if self.table_bank.canonical_root_id != expected_canonical:
            raise VllmAuxiliaryRootError("canonical root changed before publication")
        try:
            old = self._snapshots[expected_canonical]
            selected = self._snapshots[selected_root]
        except KeyError as error:
            raise VllmAuxiliaryRootError("publication auxiliary root is absent") from error
        try:
            self._restore(selected)
            self._synchronize_device()
            # No operation after this assignment may fail.  Auxiliary canonical
            # identity is read from this same bank; there is no second pointer.
            return self.table_bank.publish(
                expected_canonical=expected_canonical,
                selected_root=selected_root,
            )
        except BaseException:
            self._restore(old)
            self._synchronize_device()
            raise

    def prepare_external_publication(
        self, *, expected_canonical: str, selected_root: str
    ) -> None:
        """Stage selected scratch for an externally owned composite root swap.

        When the table bank follows ``SchedulerRootAuthority`` directly, that
        authority's immutable-state assignment is the sole publication pointer.
        This method performs every fallible restore/synchronization operation
        before that assignment and deliberately does not publish the table bank.
        """

        if not self.table_bank.canonical_is_external:
            raise VllmAuxiliaryRootError("canonical root is not externally bound")
        if self.table_bank.canonical_root_id != expected_canonical:
            raise VllmAuxiliaryRootError("canonical root changed before publication")
        try:
            old = self._snapshots[expected_canonical]
            selected = self._snapshots[selected_root]
        except KeyError as error:
            raise VllmAuxiliaryRootError("publication auxiliary root is absent") from error
        try:
            self._restore(selected)
            self._synchronize_device()
        except BaseException:
            self._restore(old)
            self._synchronize_device()
            raise

    @contextmanager
    def execute_private(self, root: PhysicalRoot) -> Iterator[None]:
        canonical_id = self.table_bank.canonical_root_id
        if root.root_id == canonical_id or root.root_id not in self._snapshots:
            raise VllmAuxiliaryRootError("private auxiliary root is not prepared")
        canonical = self._snapshots.get(canonical_id)
        selected = self._snapshots[root.root_id]
        if canonical is None:
            raise VllmAuxiliaryRootError("canonical auxiliary snapshot is absent")
        with self.table_bank.select_private(root.root_id):
            try:
                self._restore(selected)
                self._synchronize_device()
                yield
                self._synchronize_device()
                self._snapshots[root.root_id] = self._capture(root.root_id)
            finally:
                self._restore(canonical)
                self._synchronize_device()

    def compare_auxiliary_state(self, left: PhysicalRoot, right: PhysicalRoot) -> None:
        try:
            left_values = self._snapshots[left.root_id].value_map()
            right_values = self._snapshots[right.root_id].value_map()
        except KeyError as error:
            raise VllmAuxiliaryRootError("auxiliary root snapshot is absent") from error
        if set(left_values) != set(right_values):
            raise VllmAuxiliaryRootError("auxiliary root fields differ")
        for label in sorted(left_values):
            if not _equal(left_values[label], right_values[label]):
                raise VllmAuxiliaryRootError(f"auxiliary state differs at {label}")

    def capture_auxiliary_state(self, root: PhysicalRoot) -> object:
        try:
            snapshot = self._snapshots[root.root_id]
        except KeyError as error:
            raise VllmAuxiliaryRootError("auxiliary root snapshot is absent") from error
        return {
            "schema": snapshot.schema,
            "fields": {label: _digest(value) for label, value in snapshot.values},
        }
