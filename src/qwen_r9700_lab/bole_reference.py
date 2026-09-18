"""Small, dependency-free reference for Bole tree verification.

This module deliberately uses Python lists instead of NumPy or torch.  It is a
correctness oracle for the future HIP implementation, not a performance path.
The recurrence is one Gated-DeltaNet state head with scalar beta/gamma gates.
For a real Qwen layer, the caller runs this independently for each state head.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class TreePlan:
    """Topologically ordered proposal tree metadata."""

    parent: tuple[int, ...]
    depth: tuple[int, ...]
    ancestor_mask: tuple[int, ...]

    @property
    def node_count(self) -> int:
        return len(self.parent)

    @property
    def max_depth(self) -> int:
        return max(self.depth, default=0)


@dataclass(frozen=True)
class TreeInputs:
    """Per-node GDN operands for one state head."""

    q: tuple[Vector, ...]
    k: tuple[Vector, ...]
    v: tuple[Vector, ...]
    beta: tuple[float, ...]
    gamma: tuple[float, ...]


@dataclass(frozen=True)
class TreeResult:
    """Node outputs and compact factors retained until acceptance."""

    outputs: tuple[Vector, ...]
    factors_p: tuple[float, ...]
    factors_u: tuple[Vector, ...]


def _check_matrix(matrix: Sequence[Sequence[float]], rows: int, cols: int, name: str) -> None:
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must have shape [{rows},{cols}]")
    if any(not isfinite(value) for row in matrix for value in row):
        raise ValueError(f"{name} contains a non-finite value")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _outer(left: Sequence[float], right: Sequence[float]) -> Matrix:
    return [[a * b for b in right] for a in left]


def _mat_vec_transpose(state: Matrix, vector: Sequence[float]) -> Vector:
    return [
        sum(state[row][col] * vector[row] for row in range(len(state)))
        for col in range(len(state[0]))
    ]


def _state_add_scaled_outer(
    state: Matrix, scale: float, left: Sequence[float], right: Sequence[float]
) -> Matrix:
    return [
        [state[row][col] + scale * left[row] * right[col] for col in range(len(state[row]))]
        for row in range(len(state))
    ]


def build_tree_plan(parent: Sequence[int]) -> TreePlan:
    """Validate a parent-before-child tree and build reusable masks."""

    if not parent:
        raise ValueError("tree must contain at least one node")
    depths: list[int] = []
    masks: list[int] = []
    for index, raw_parent in enumerate(parent):
        parent_index = int(raw_parent)
        if parent_index < -1 or parent_index >= index:
            raise ValueError(f"parent[{index}]={parent_index} is not parent-before-child")
        if parent_index < 0:
            depths.append(0)
            masks.append(0)
        else:
            depths.append(depths[parent_index] + 1)
            masks.append(masks[parent_index] | (1 << parent_index))
    return TreePlan(tuple(int(value) for value in parent), tuple(depths), tuple(masks))


def _validate_inputs(state_pre: Matrix, plan: TreePlan, inputs: TreeInputs) -> tuple[int, int]:
    if len(state_pre) == 0 or len(state_pre[0]) == 0:
        raise ValueError("state_pre must be non-empty")
    dk, dv = len(state_pre), len(state_pre[0])
    _check_matrix(state_pre, dk, dv, "state_pre")
    if (
        len(inputs.q) != plan.node_count
        or len(inputs.k) != plan.node_count
        or len(inputs.v) != plan.node_count
    ):
        raise ValueError("q, k, and v must have one row per tree node")
    if len(inputs.beta) != plan.node_count or len(inputs.gamma) != plan.node_count:
        raise ValueError("beta and gamma must have one value per tree node")
    _check_matrix(inputs.q, plan.node_count, dk, "q")
    _check_matrix(inputs.k, plan.node_count, dk, "k")
    _check_matrix(inputs.v, plan.node_count, dv, "v")
    if any(not isfinite(value) for value in (*inputs.beta, *inputs.gamma)):
        raise ValueError("beta/gamma contains a non-finite value")
    return dk, dv


def sequential_verify(state_pre: Matrix, plan: TreePlan, inputs: TreeInputs) -> TreeResult:
    """Reference parent-by-parent GDN verification.

    This intentionally materializes one state per node.  It is the oracle for
    the factorized implementation and must never be used in production.
    """

    dk, _ = _validate_inputs(state_pre, plan, inputs)
    states: list[Matrix] = []
    outputs: list[Vector] = []
    factors_p: list[float] = []
    factors_u: list[Vector] = []
    for index, parent in enumerate(plan.parent):
        parent_state = state_pre if parent < 0 else states[parent]
        p = inputs.gamma[index] if parent < 0 else factors_p[parent] * inputs.gamma[index]
        tilde = [[inputs.gamma[index] * value for value in row] for row in parent_state]
        u = _scale_vector(
            inputs.beta[index],
            _vector_sub(inputs.v[index], _mat_vec_transpose(tilde, inputs.k[index])),
        )
        state = _state_add_scaled_outer(tilde, 1.0, inputs.k[index], u)
        output = _mat_vec_transpose(state, inputs.q[index])
        if len(state) != dk:
            raise AssertionError("internal state dimension changed")
        states.append(state)
        outputs.append(output)
        factors_p.append(p)
        factors_u.append(u)
    return TreeResult(tuple(outputs), tuple(factors_p), tuple(factors_u))


def _vector_sub(left: Sequence[float], right: Sequence[float]) -> Vector:
    return [a - b for a, b in zip(left, right, strict=True)]


def _scale_vector(scale: float, vector: Sequence[float]) -> Vector:
    return [scale * value for value in vector]


def bole_verify(state_pre: Matrix, plan: TreePlan, inputs: TreeInputs) -> TreeResult:
    """Exact Bole factorized verification for one GDN state head.

    For node ``i`` define ``P_i`` as the product of gamma gates on its path.
    The rank-one update factors satisfy ``(I + G) U = R`` where ``G`` is
    strictly lower triangular in topological node order.  The finite Neumann
    series terminates after ``plan.max_depth`` terms, so this implementation
    has no approximation and no per-node state snapshots.
    """

    _, dv = _validate_inputs(state_pre, plan, inputs)
    node_count = plan.node_count
    factors_p = [0.0] * node_count
    for index, parent in enumerate(plan.parent):
        factors_p[index] = (
            inputs.gamma[index] if parent < 0 else factors_p[parent] * inputs.gamma[index]
        )

    # R_i = beta_i (v_i - P_i S_pre^T k_i); G_ij is nonzero only for ancestors.
    rhs: list[Vector] = []
    gram: list[list[float]] = [[0.0] * node_count for _ in range(node_count)]
    for index in range(node_count):
        projected = _mat_vec_transpose(state_pre, inputs.k[index])
        rhs.append(
            [
                inputs.beta[index] * (inputs.v[index][col] - factors_p[index] * projected[col])
                for col in range(dv)
            ]
        )
        ancestor_mask = plan.ancestor_mask[index]
        ancestor = ancestor_mask
        while ancestor:
            bit = ancestor & -ancestor
            parent_index = bit.bit_length() - 1
            gram[index][parent_index] = (
                inputs.beta[index]
                * (factors_p[index] / factors_p[parent_index])
                * _dot(inputs.k[index], inputs.k[parent_index])
            )
            ancestor ^= bit

    # U = R - G R + G^2 R - ... .  Use a fixed number of rounds based on tree depth.
    factors_u = [row[:] for row in rhs]
    z = [row[:] for row in rhs]
    for _ in range(plan.max_depth):
        next_z = [[0.0] * dv for _ in range(node_count)]
        for index in range(node_count):
            for ancestor in range(index):
                coefficient = -gram[index][ancestor]
                if coefficient == 0.0:
                    continue
                for col in range(dv):
                    next_z[index][col] += coefficient * z[ancestor][col]
        for index in range(node_count):
            for col in range(dv):
                factors_u[index][col] += next_z[index][col]
        z = next_z

    outputs: list[Vector] = []
    for index in range(node_count):
        output = [
            factors_p[index] * value for value in _mat_vec_transpose(state_pre, inputs.q[index])
        ]
        for ancestor in range(index + 1):
            if ancestor != index and not (plan.ancestor_mask[index] & (1 << ancestor)):
                continue
            coefficient = (
                factors_p[index] / factors_p[ancestor] * _dot(inputs.q[index], inputs.k[ancestor])
            )
            for col in range(dv):
                output[col] += coefficient * factors_u[ancestor][col]
        outputs.append(output)
    return TreeResult(tuple(outputs), tuple(factors_p), tuple(factors_u))


def commit_state(
    state_pre: Matrix,
    plan: TreePlan,
    result: TreeResult,
    inputs: TreeInputs,
    accepted_node: int,
) -> Matrix:
    """Reconstruct only the selected path, matching Bole Eq. (10)."""

    if accepted_node < 0 or accepted_node >= plan.node_count:
        raise ValueError("accepted_node is outside the tree")
    path: list[int] = []
    current = accepted_node
    while current >= 0:
        path.append(current)
        current = plan.parent[current]
    path.reverse()
    committed = [[result.factors_p[accepted_node] * value for value in row] for row in state_pre]
    for node in path:
        coefficient = result.factors_p[accepted_node] / result.factors_p[node]
        committed = _state_add_scaled_outer(
            committed, coefficient, inputs.k[node], result.factors_u[node]
        )
    return committed
