"""Concrete 69-group private-root copier/comparator for Qwen3.8 + DFlash2.

The fixed production topology is 48 GDN groups, 16 target-attention groups,
and five draft-attention groups.  Immutable attention prefixes are shared;
only explicitly declared tail blocks and all recurrent slots differ between
roots.  This runtime copies canonical mutable state into two disjoint roots and
compares their exact device bytes before either root can be published.

This is the worker-side physical primitive.  Scheduler/allocator ownership and
position/decoder state remain separately supplied mandatory callbacks and are
combined by the scheduler authority into the complete-state contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qwen_r9700_lab.live_commit_scheduler_authority import PhysicalRoot

MAMBA_LAYER_NAMES = tuple(
    f"language_model.model.layers.{layer}.linear_attn"
    for quartet in range(16)
    for layer in range(quartet * 4, quartet * 4 + 3)
)
FULL_ATTENTION_LAYER_NAMES = tuple(
    f"language_model.model.layers.{quartet * 4 + 3}.self_attn.attn"
    for quartet in range(16)
)
DFLASH_LAYER_COUNT = 5


class VllmPhysicalRootError(RuntimeError):
    """The real worker state cannot satisfy the private-root contract."""


@dataclass(frozen=True)
class MutableRootCapture:
    """Address-independent exact digests for every branch-mutable GPU unit."""

    target_kv_sha256_by_layer: tuple[str, ...]
    draft_kv_sha256_by_layer: tuple[str, ...]
    gdn_state_sha256_by_layer: tuple[str, ...]
    convolution_state_sha256_by_layer: tuple[str, ...]
    auxiliary_state_sha256: str
    combined_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes_sha256(tensors: tuple[Any, ...]) -> str:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - vLLM always includes Torch.
        raise VllmPhysicalRootError("Torch is unavailable") from error
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise VllmPhysicalRootError("physical root contains a non-tensor value")
    digest = hashlib.sha256(b"qwen-private-root-tensors-v1\0")
    for tensor in tensors:
        contiguous = tensor.detach().contiguous()
        descriptor = {
            "device": str(contiguous.device),
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "stride": list(contiguous.stride()),
        }
        digest.update(_canonical(descriptor))
        try:
            raw = contiguous.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        except BaseException as error:
            raise VllmPhysicalRootError("physical tensor bytes could not be read") from error
        digest.update(raw)
    return digest.hexdigest()


def _require_equal(left: Any, right: Any, label: str) -> None:
    import torch

    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.dtype != right.dtype
        or left.shape != right.shape
        or not torch.equal(left, right)
    ):
        raise VllmPhysicalRootError(f"{label} differs")


class VllmWorkerPhysicalRoots:
    """Copy and compare the actual mutable tensors for one TP=1 model runner."""

    def __init__(
        self,
        runner: object,
        *,
        copy_auxiliary_state: Callable[[PhysicalRoot, PhysicalRoot, PhysicalRoot], None],
        compare_auxiliary_state: Callable[[PhysicalRoot, PhysicalRoot], None],
        capture_auxiliary_state: Callable[[PhysicalRoot], object],
        synchronize_device: Callable[[], None],
    ) -> None:
        for label, callback in (
            ("auxiliary copier", copy_auxiliary_state),
            ("auxiliary comparator", compare_auxiliary_state),
            ("auxiliary capture", capture_auxiliary_state),
            ("device synchronizer", synchronize_device),
        ):
            if not callable(callback):
                raise VllmPhysicalRootError(f"{label} is invalid")
        self.runner = runner
        self._copy_auxiliary_state = copy_auxiliary_state
        self._compare_auxiliary_state = compare_auxiliary_state
        self._capture_auxiliary_state = capture_auxiliary_state
        self._synchronize_device = synchronize_device
        self._target_modules, self._gdn_modules, self._draft_modules = (
            self._resolve_modules()
        )

    def _resolve_modules(self) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        context = getattr(
            getattr(self.runner, "compilation_config", None),
            "static_forward_context",
            None,
        )
        if not isinstance(context, dict):
            raise VllmPhysicalRootError("model runner has no static forward context")
        try:
            target = tuple(context[name] for name in FULL_ATTENTION_LAYER_NAMES)
            gdn = tuple(context[name] for name in MAMBA_LAYER_NAMES)
        except KeyError as error:
            raise VllmPhysicalRootError(
                "model runner is missing a fixed Qwen target layer"
            ) from error
        draft_model = getattr(self.runner, "get_draft_model", lambda: None)()
        draft = getattr(getattr(draft_model, "model", None), "_attn_layers", None)
        if not isinstance(draft, list) or len(draft) != DFLASH_LAYER_COUNT:
            raise VllmPhysicalRootError("model runner has the wrong DFlash topology")
        return target, gdn, tuple(draft)

    @staticmethod
    def _attention_cache(module: object, label: str) -> tuple[Any, Any]:
        import torch

        cache = getattr(module, "kv_cache", None)
        if (
            isinstance(cache, tuple)
            and len(cache) == 2
            and all(isinstance(item, torch.Tensor) for item in cache)
        ):
            key, value = cache
        else:
            try:
                from vllm.v1.attention.ops.paged_attn import PagedAttention

                key, value = PagedAttention.split_kv_cache(
                    cache,
                    module.num_kv_heads,
                    module.head_size,
                )
            except BaseException as error:
                raise VllmPhysicalRootError(
                    f"{label} attention cache cannot be split"
                ) from error
        if (
            not isinstance(key, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or key.ndim < 2
            or value.ndim < 2
            or key.shape[0] != value.shape[0]
        ):
            raise VllmPhysicalRootError(f"{label} attention cache layout differs")
        return key, value

    @staticmethod
    def _gdn_cache(module: object, label: str) -> tuple[Any, Any]:
        import torch

        cache = getattr(module, "kv_cache", None)
        if (
            not isinstance(cache, tuple)
            or len(cache) != 2
            or any(not isinstance(item, torch.Tensor) or item.ndim < 1 for item in cache)
        ):
            raise VllmPhysicalRootError(f"{label} GDN cache layout differs")
        convolution, gdn = cache
        return convolution, gdn

    @staticmethod
    def _copy_attention_group(
        module: object,
        source_blocks: tuple[int, ...],
        destination_blocks: tuple[int, ...],
        writable_blocks: tuple[int, ...],
        label: str,
    ) -> None:
        if len(source_blocks) != len(destination_blocks):
            raise VllmPhysicalRootError(f"{label} logical block count differs")
        writable = set(writable_blocks)
        differing_positions = {
            index
            for index, (source, destination) in enumerate(
                zip(source_blocks, destination_blocks, strict=True)
            )
            if source != destination
        }
        declared_positions = {
            index
            for index, block in enumerate(destination_blocks)
            if block in writable
        }
        if differing_positions != declared_positions or len(writable) != len(
            writable_blocks
        ):
            raise VllmPhysicalRootError(f"{label} writable block map differs")
        key, value = VllmWorkerPhysicalRoots._attention_cache(module, label)
        capacity = int(key.shape[0])
        for index in sorted(differing_positions):
            source = source_blocks[index]
            destination = destination_blocks[index]
            if not 0 <= source < capacity or not 0 <= destination < capacity:
                raise VllmPhysicalRootError(f"{label} block is outside storage")
            key[destination].copy_(key[source])
            value[destination].copy_(value[source])

    def _copy_root(self, canonical: PhysicalRoot, destination: PhysicalRoot) -> None:
        for index, module in enumerate(self._target_modules):
            self._copy_attention_group(
                module,
                canonical.target_blocks[index],
                destination.target_blocks[index],
                destination.writable_target_blocks[index],
                f"target layer {index}",
            )
        for index, module in enumerate(self._draft_modules):
            self._copy_attention_group(
                module,
                canonical.draft_blocks[index],
                destination.draft_blocks[index],
                destination.writable_draft_blocks[index],
                f"draft layer {index}",
            )
        for index, module in enumerate(self._gdn_modules):
            convolution, gdn = self._gdn_cache(module, f"GDN layer {index}")
            source_gdn = canonical.gdn_slots[index]
            destination_gdn = destination.gdn_slots[index]
            source_conv = canonical.convolution_slots[index]
            destination_conv = destination.convolution_slots[index]
            if any(
                slot < 0 or slot >= int(storage.shape[0])
                for slot, storage in (
                    (source_gdn, gdn),
                    (destination_gdn, gdn),
                    (source_conv, convolution),
                    (destination_conv, convolution),
                )
            ):
                raise VllmPhysicalRootError(f"GDN layer {index} slot is outside storage")
            gdn[destination_gdn].copy_(gdn[source_gdn])
            convolution[destination_conv].copy_(convolution[source_conv])

    @staticmethod
    def _mutable_rows(
        module: object,
        root: PhysicalRoot,
        *,
        target: bool,
        group_index: int,
    ) -> tuple[Any, ...]:
        key, value = VllmWorkerPhysicalRoots._attention_cache(
            module, ("target" if target else "draft") + f" layer {group_index}"
        )
        blocks = (
            root.writable_target_blocks[group_index]
            if target
            else root.writable_draft_blocks[group_index]
        )
        if any(block >= int(key.shape[0]) for block in blocks):
            raise VllmPhysicalRootError("mutable attention block is outside storage")
        rows: list[Any] = []
        for block in blocks:
            rows.extend((key[block], value[block]))
        return tuple(rows)

    def compare_roots(self, left: PhysicalRoot, right: PhysicalRoot) -> None:
        """Require exact equality of every mutable GPU and auxiliary state unit."""

        for index, module in enumerate(self._target_modules):
            left_rows = self._mutable_rows(module, left, target=True, group_index=index)
            right_rows = self._mutable_rows(module, right, target=True, group_index=index)
            if len(left_rows) != len(right_rows):
                raise VllmPhysicalRootError(f"target layer {index} row count differs")
            for row, (left_tensor, right_tensor) in enumerate(
                zip(left_rows, right_rows, strict=True)
            ):
                _require_equal(left_tensor, right_tensor, f"target layer {index} row {row}")
        for index, module in enumerate(self._draft_modules):
            left_rows = self._mutable_rows(module, left, target=False, group_index=index)
            right_rows = self._mutable_rows(module, right, target=False, group_index=index)
            if len(left_rows) != len(right_rows):
                raise VllmPhysicalRootError(f"draft layer {index} row count differs")
            for row, (left_tensor, right_tensor) in enumerate(
                zip(left_rows, right_rows, strict=True)
            ):
                _require_equal(left_tensor, right_tensor, f"draft layer {index} row {row}")
        for index, module in enumerate(self._gdn_modules):
            convolution, gdn = self._gdn_cache(module, f"GDN layer {index}")
            _require_equal(
                gdn[left.gdn_slots[index]],
                gdn[right.gdn_slots[index]],
                f"GDN layer {index} state",
            )
            _require_equal(
                convolution[left.convolution_slots[index]],
                convolution[right.convolution_slots[index]],
                f"GDN layer {index} convolution",
            )
        try:
            self._compare_auxiliary_state(left, right)
        except BaseException as error:
            raise VllmPhysicalRootError("auxiliary root state differs") from error

    def prepare_private_roots(
        self,
        canonical: PhysicalRoot,
        candidate: PhysicalRoot,
        serial: PhysicalRoot,
    ) -> None:
        """Copy canonical mutable bytes to both disjoint roots, then verify them."""

        if len({canonical.root_id, candidate.root_id, serial.root_id}) != 3:
            raise VllmPhysicalRootError("physical root IDs alias")
        self._copy_root(canonical, candidate)
        self._copy_root(canonical, serial)
        try:
            self._copy_auxiliary_state(canonical, candidate, serial)
            self._synchronize_device()
        except BaseException as error:
            raise VllmPhysicalRootError("private root copy did not complete") from error
        self.compare_roots(canonical, candidate)
        self.compare_roots(canonical, serial)

    def capture_mutable_state(self, root: PhysicalRoot) -> MutableRootCapture:
        target = tuple(
            _tensor_bytes_sha256(
                self._mutable_rows(module, root, target=True, group_index=index)
            )
            for index, module in enumerate(self._target_modules)
        )
        draft = tuple(
            _tensor_bytes_sha256(
                self._mutable_rows(module, root, target=False, group_index=index)
            )
            for index, module in enumerate(self._draft_modules)
        )
        gdn: list[str] = []
        convolution: list[str] = []
        for index, module in enumerate(self._gdn_modules):
            convolution_cache, gdn_cache = self._gdn_cache(
                module, f"GDN layer {index}"
            )
            gdn.append(
                _tensor_bytes_sha256((gdn_cache[root.gdn_slots[index]],))
            )
            convolution.append(
                _tensor_bytes_sha256(
                    (convolution_cache[root.convolution_slots[index]],)
                )
            )
        try:
            auxiliary = _sha256(_canonical(self._capture_auxiliary_state(root)))
        except BaseException as error:
            raise VllmPhysicalRootError("auxiliary root state cannot be captured") from error
        body = {
            "target": target,
            "draft": draft,
            "gdn": gdn,
            "convolution": convolution,
            "auxiliary": auxiliary,
        }
        return MutableRootCapture(
            target,
            draft,
            tuple(gdn),
            tuple(convolution),
            auxiliary,
            _sha256(_canonical(body)),
        )
