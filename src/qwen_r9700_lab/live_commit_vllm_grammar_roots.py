"""Independent structured-output FSM roots for live serial validation.

xgrammar matchers are mutable native objects and do not expose a supported clone
operation.  Sharing one matcher lets candidate execution advance the serial
oracle's grammar; using ``deepcopy`` is neither supported nor auditable.  This
module instead keeps one independently constructed matcher per physical root and
rebuilds private roots by resetting and replaying the canonical matcher's exact
accepted-token suffix.

The scheduler/request owner remains responsible for making the selected request
metadata and this grammar root part of its one composite authority pointer.  No
publication operation exists here deliberately.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass


class VllmGrammarRootError(RuntimeError):
    """A structured-output matcher cannot be isolated or reconstructed exactly."""


@dataclass(frozen=True)
class GrammarCheckpoint:
    """Complete address-independent state needed to reconstruct one matcher."""

    accepted_token_ids: tuple[int, ...]
    reasoning_ended: bool | None
    reasoning_end_token_index: int | None
    terminated: bool


def _token_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VllmGrammarRootError(f"{label} are not a token sequence")
    result = tuple(value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in result):
        raise VllmGrammarRootError(f"{label} contain an invalid token")
    return result


def _grammar(request: object) -> object:
    structured = getattr(request, "structured_output_request", None)
    if structured is None:
        raise VllmGrammarRootError("request has no structured-output state")
    grammar = getattr(structured, "grammar", None)
    for method in ("accept_tokens", "reset", "is_terminated"):
        if not callable(getattr(grammar, method, None)):
            raise VllmGrammarRootError(f"grammar omits {method}")
    count = getattr(grammar, "num_processed_tokens", None)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise VllmGrammarRootError("grammar processed-token count is invalid")
    return grammar


def _capture(request: object, grammar: object) -> GrammarCheckpoint:
    current = _grammar(request)
    if current is not grammar:
        raise VllmGrammarRootError("request grammar does not match the active root")
    output = _token_ids(getattr(request, "_output_token_ids", None), "request output IDs")
    count = grammar.num_processed_tokens
    if count > len(output):
        raise VllmGrammarRootError("grammar processed more tokens than request output")
    accepted = output[len(output) - count :] if count else ()
    terminated = grammar.is_terminated()
    if not isinstance(terminated, bool):
        raise VllmGrammarRootError("grammar termination state is invalid")
    structured = request.structured_output_request
    reasoning_ended = getattr(structured, "reasoning_ended", None)
    if reasoning_ended not in (None, False, True):
        raise VllmGrammarRootError("reasoning-ended state is invalid")
    end_index = getattr(structured, "reasoning_end_token_index", None)
    if end_index is not None and (
        isinstance(end_index, bool) or not isinstance(end_index, int) or end_index < 0
    ):
        raise VllmGrammarRootError("reasoning-end token index is invalid")
    all_ids = _token_ids(getattr(request, "_all_token_ids", None), "request token IDs")
    if end_index is not None and end_index >= len(all_ids):
        raise VllmGrammarRootError("reasoning-end token index is outside the request")
    return GrammarCheckpoint(accepted, reasoning_ended, end_index, terminated)


def _restore(
    request: object,
    grammar: object,
    checkpoint: GrammarCheckpoint,
    *,
    request_id: str,
) -> None:
    grammar.reset()
    if checkpoint.accepted_token_ids and not grammar.accept_tokens(
        request_id, list(checkpoint.accepted_token_ids)
    ):
        raise VllmGrammarRootError("grammar rejected its canonical replay")
    if getattr(grammar, "num_processed_tokens", None) != len(
        checkpoint.accepted_token_ids
    ):
        raise VllmGrammarRootError("grammar replay count differs")
    if grammar.is_terminated() is not checkpoint.terminated:
        raise VllmGrammarRootError("grammar replay termination differs")
    structured = request.structured_output_request
    structured.grammar = grammar
    structured.reasoning_ended = checkpoint.reasoning_ended
    structured.reasoning_end_token_index = checkpoint.reasoning_end_token_index
    # Reasoners are request-scoped caches derived from the complete token list.
    # Never share a potentially stateful parser across branch executions.
    structured.reasoner = None


class VllmGrammarRootBank:
    """Three independent FSMs keyed by canonical/candidate/serial root IDs."""

    def __init__(
        self,
        request: object,
        *,
        request_id: str,
        root_ids: Sequence[str],
        make_independent_grammar: Callable[[object], object],
        canonical_root_id: Callable[[], str],
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise VllmGrammarRootError("request ID is invalid")
        ids = tuple(root_ids)
        if len(ids) != 3 or any(not isinstance(root, str) or not root for root in ids):
            raise VllmGrammarRootError("exactly three grammar root IDs are required")
        if len(set(ids)) != 3:
            raise VllmGrammarRootError("grammar root IDs alias")
        if not callable(make_independent_grammar) or not callable(canonical_root_id):
            raise VllmGrammarRootError("grammar root callbacks are invalid")
        self.request = request
        self.request_id = request_id
        self.root_ids = ids
        self._canonical_root_id = canonical_root_id
        canonical_id = self._canonical()
        canonical = _grammar(request)
        self._grammars = {canonical_id: canonical}
        checkpoint = _capture(request, canonical)
        for root_id in ids:
            if root_id == canonical_id:
                continue
            try:
                grammar = make_independent_grammar(canonical)
            except BaseException as error:
                raise VllmGrammarRootError("independent grammar construction failed") from error
            if grammar is canonical or any(
                grammar is existing for existing in self._grammars.values()
            ):
                raise VllmGrammarRootError("independent grammar roots alias")
            self._grammars[root_id] = grammar
            _restore(request, grammar, checkpoint, request_id=request_id)
        _restore(request, canonical, checkpoint, request_id=request_id)
        self._checkpoints = dict.fromkeys(ids, checkpoint)

    def _canonical(self) -> str:
        try:
            root_id = self._canonical_root_id()
        except BaseException as error:
            raise VllmGrammarRootError("canonical grammar root lookup failed") from error
        if root_id not in self.root_ids:
            raise VllmGrammarRootError("canonical grammar root is unknown")
        return root_id

    def prepare_private_roots(self, canonical: str, candidate: str, serial: str) -> None:
        # Root rotation changes which two IDs are spares; accept either spare
        # order but reject aliases or roots outside this bank.
        if (
            canonical != self._canonical()
            or {candidate, serial} != set(self.root_ids) - {canonical}
            or candidate == serial
        ):
            raise VllmGrammarRootError("grammar private-root topology differs")
        canonical_grammar = self._grammars[canonical]
        checkpoint = _capture(self.request, canonical_grammar)
        self._checkpoints[canonical] = checkpoint
        for root_id in (candidate, serial):
            _restore(
                self.request,
                self._grammars[root_id],
                checkpoint,
                request_id=self.request_id,
            )
            self._checkpoints[root_id] = checkpoint
        _restore(
            self.request,
            canonical_grammar,
            checkpoint,
            request_id=self.request_id,
        )

    @contextmanager
    def activate_private(self, root_id: str) -> Iterator[None]:
        canonical = self._canonical()
        if root_id == canonical or root_id not in self._grammars:
            raise VllmGrammarRootError("private grammar root is invalid")
        canonical_checkpoint = self._checkpoints[canonical]
        grammar = self._grammars[root_id]
        _restore(
            self.request,
            grammar,
            self._checkpoints[root_id],
            request_id=self.request_id,
        )
        try:
            yield
            self._checkpoints[root_id] = _capture(self.request, grammar)
        finally:
            _restore(
                self.request,
                self._grammars[canonical],
                canonical_checkpoint,
                request_id=self.request_id,
            )

    def compare(self, left: str, right: str) -> None:
        try:
            left_state = self._checkpoints[left]
            right_state = self._checkpoints[right]
        except KeyError as error:
            raise VllmGrammarRootError("grammar comparison root is absent") from error
        if left_state != right_state:
            raise VllmGrammarRootError("structured-output grammar state differs")

    def prepare_external_publication(self, expected: str, selected: str) -> None:
        if self._canonical() != expected:
            raise VllmGrammarRootError("canonical grammar root changed before publication")
        try:
            checkpoint = self._checkpoints[selected]
            grammar = self._grammars[selected]
        except KeyError as error:
            raise VllmGrammarRootError("selected grammar root is absent") from error
        _restore(self.request, grammar, checkpoint, request_id=self.request_id)
