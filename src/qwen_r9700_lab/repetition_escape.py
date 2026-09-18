"""Exact-token policy for conservative, bounded greedy-loop escape.

The functions in this module are deliberately independent of vLLM and Torch.
They are the executable reference contract for the serving overlay: detection
uses only target-approved output token IDs, a normal EOS is never intercepted,
and a mutation is allowed only at the final token of a target-approved round
whose complete sequence still ends in a long exact periodic suffix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

DEFAULT_MAX_PERIOD = 64
DEFAULT_MIN_COPIES = 8
DEFAULT_MIN_REPEATED_TOKENS = 256
DEFAULT_MAX_ESCAPES = 4
DEFAULT_MAX_CANDIDATES = 16


@dataclass(frozen=True)
class PeriodicSuffix:
    """Evidence that the exact token suffix is periodic."""

    copies: int
    period: int
    pattern_token_ids: tuple[int, ...]
    repeated_suffix_tokens: int

    def as_dict(self) -> dict[str, int | list[int]]:
        document = asdict(self)
        document["pattern_token_ids"] = list(self.pattern_token_ids)
        return document


@dataclass(frozen=True)
class EscapePolicy:
    """Validated bounds shared by the reference and serving implementations."""

    max_period: int = DEFAULT_MAX_PERIOD
    min_copies: int = DEFAULT_MIN_COPIES
    min_repeated_tokens: int = DEFAULT_MIN_REPEATED_TOKENS
    max_escapes: int = DEFAULT_MAX_ESCAPES
    max_candidates: int = DEFAULT_MAX_CANDIDATES

    def __post_init__(self) -> None:
        if self.max_period < 1:
            raise ValueError("max_period must be positive")
        if self.min_copies < 2:
            raise ValueError("min_copies must be at least two")
        if self.min_repeated_tokens < 1:
            raise ValueError("min_repeated_tokens must be positive")
        if self.min_repeated_tokens < 2 * self.max_period:
            raise ValueError(
                "min_repeated_tokens must be at least twice max_period"
            )
        if self.max_escapes < 1:
            raise ValueError("max_escapes must be positive")
        if self.max_candidates < 2:
            raise ValueError("max_candidates must include an alternative")

    def periodic_kwargs(self) -> dict[str, int]:
        return {
            "max_period": self.max_period,
            "min_copies": self.min_copies,
            "min_repeated_tokens": self.min_repeated_tokens,
        }


DEFAULT_POLICY = EscapePolicy()


@dataclass(frozen=True)
class EscapeBoundary:
    """Last target-approved token when a round ends in a periodic loop."""

    round_index: int
    looping_token_id: int
    evidence: PeriodicSuffix


@dataclass(frozen=True)
class EscapeDecision:
    """A bounded serving action; ``None`` means the round is untouched."""

    action: Literal["escape", "terminate", "blocked"]
    boundary: EscapeBoundary
    replacement_token_id: int | None
    prior_escape_count: int
    reason: str

    @property
    def retained_round_tokens(self) -> int:
        """Number of original round tokens retained before the replacement."""

        return self.boundary.round_index

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "boundary_round_index": self.boundary.round_index,
            "looping_token_id": self.boundary.looping_token_id,
            "replacement_token_id": self.replacement_token_id,
            "prior_escape_count": self.prior_escape_count,
            "reason": self.reason,
            "periodic_suffix": self.boundary.evidence.as_dict(),
        }


def periodic_suffix(
    tokens: Sequence[int],
    *,
    max_period: int = DEFAULT_MAX_PERIOD,
    min_copies: int = DEFAULT_MIN_COPIES,
    min_repeated_tokens: int = DEFAULT_MIN_REPEATED_TOKENS,
) -> PeriodicSuffix | None:
    """Return the smallest proven period meeting both copy and span thresholds."""

    if max_period < 1 or min_copies < 2 or min_repeated_tokens < 1:
        raise ValueError("invalid periodic-suffix policy")
    limit = min(max_period, len(tokens) // min_copies)
    for period in range(1, limit + 1):
        required_copies = max(min_copies, (min_repeated_tokens + period - 1) // period)
        required_span = required_copies * period
        if required_span > len(tokens):
            continue
        pattern = tuple(tokens[-period:])
        copies = 1
        cursor = len(tokens) - 2 * period
        while cursor >= 0 and tuple(tokens[cursor : cursor + period]) == pattern:
            copies += 1
            cursor -= period
        if copies >= required_copies:
            return PeriodicSuffix(
                copies=copies,
                period=period,
                pattern_token_ids=pattern,
                repeated_suffix_tokens=copies * period,
            )
    return None


def would_continue_period(
    history: Sequence[int],
    candidate: int,
    **policy: int,
) -> PeriodicSuffix | None:
    """Return evidence only when appending ``candidate`` crosses the policy boundary."""

    before = periodic_suffix(history, **policy)
    after = periodic_suffix([*history, candidate], **policy)
    return after if before is None else None


def first_periodic_boundary(
    history: Sequence[int],
    target_approved_round: Sequence[int],
    *,
    policy: EscapePolicy = DEFAULT_POLICY,
) -> EscapeBoundary | None:
    """Return a safe mutation boundary only when the verified round ends looping.

    A speculative round can cross the repetition threshold and then escape on
    its own.  Rewriting that earlier token would throw away valid target-approved
    tokens computed after it.  The serving hook can safely replace the *last*
    sampled token because that token has not yet been incorporated into model
    state; it therefore acts only when the complete accepted round still has a
    long exact periodic suffix.
    """

    if not target_approved_round:
        return None
    evidence = periodic_suffix(
        [*history, *target_approved_round], **policy.periodic_kwargs()
    )
    if evidence is None:
        return None
    index = len(target_approved_round) - 1
    return EscapeBoundary(index, target_approved_round[index], evidence)


def select_escape_token(
    *,
    history: Sequence[int],
    ranked_target_token_ids: Sequence[int],
    eos_token_ids: frozenset[int] = frozenset(),
    allowed: Callable[[int], bool] | None = None,
    immediately_repeats: Callable[[int], bool] | None = None,
    policy: EscapePolicy = DEFAULT_POLICY,
) -> int:
    """Choose the highest-ranked admissible target alternative to the looping argmax."""

    if not ranked_target_token_ids:
        raise ValueError("target alternatives are empty")
    looping = ranked_target_token_ids[0]
    evidence = periodic_suffix([*history, looping], **policy.periodic_kwargs())
    if evidence is None:
        raise ValueError("target argmax does not end in a periodic suffix")
    predicate = allowed or (lambda _token: True)
    lookahead = immediately_repeats or (lambda _token: False)
    for token in ranked_target_token_ids[1 : policy.max_candidates]:
        if token in eos_token_ids or not predicate(token) or lookahead(token):
            continue
        return token
    raise ValueError("no admissible target alternative escapes the periodic suffix")


def plan_escape(
    *,
    history: Sequence[int],
    target_approved_round: Sequence[int],
    ranked_target_token_ids: Sequence[int],
    prior_escape_count: int,
    eos_token_ids: frozenset[int],
    allowed: Callable[[int], bool] | None = None,
    immediately_repeats: Callable[[int], bool] | None = None,
    policy: EscapePolicy = DEFAULT_POLICY,
) -> EscapeDecision | None:
    """Plan the minimum mutation at a proven periodic boundary.

    ``ranked_target_token_ids`` must come from the target logits at the returned
    boundary after grammar/logit masks have been applied.  A candidate rejected
    by ``allowed`` or ``immediately_repeats`` is never substituted.  EOS is kept
    solely as a bounded terminal guard; it is not used as an ordinary escape.
    """

    if prior_escape_count < 0:
        raise ValueError("prior_escape_count cannot be negative")
    boundary = first_periodic_boundary(history, target_approved_round, policy=policy)
    if boundary is None:
        return None
    if not ranked_target_token_ids:
        return EscapeDecision(
            "blocked", boundary, None, prior_escape_count, "target candidates are absent"
        )
    if ranked_target_token_ids[0] != boundary.looping_token_id:
        return EscapeDecision(
            "blocked",
            boundary,
            None,
            prior_escape_count,
            "target argmax does not match the verified looping token",
        )

    predicate = allowed or (lambda _token: True)
    lookahead = immediately_repeats or (lambda _token: False)
    candidates = ranked_target_token_ids[: policy.max_candidates]
    if prior_escape_count < policy.max_escapes:
        for token_id in candidates[1:]:
            if token_id in eos_token_ids:
                continue
            if not predicate(token_id) or lookahead(token_id):
                continue
            return EscapeDecision(
                "escape",
                boundary,
                token_id,
                prior_escape_count,
                "highest-ranked admissible non-looping target alternative",
            )

    # Do not turn a normal answer-ending EOS into anything else.  This branch is
    # reached only after a long periodic suffix has already been proved.  The
    # grammar predicate still decides whether termination is legal here.
    for token_id in candidates:
        if token_id in eos_token_ids and predicate(token_id):
            reason = (
                "bounded escape budget exhausted"
                if prior_escape_count >= policy.max_escapes
                else "no admissible non-terminal target alternative"
            )
            return EscapeDecision(
                "terminate", boundary, token_id, prior_escape_count, reason
            )

    reason = (
        "bounded escape budget exhausted and EOS is not grammar-admissible"
        if prior_escape_count >= policy.max_escapes
        else "no admissible target alternative or grammar-admissible EOS"
    )
    return EscapeDecision("blocked", boundary, None, prior_escape_count, reason)
