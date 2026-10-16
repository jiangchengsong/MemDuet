# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# Adapted for MemDuet inference and sampling.
"""Asynchronous fixed-window memory inference with verified updates."""

from __future__ import annotations
import asyncio
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, List, Optional, Protocol
from enum import Enum

KEEP_CONTROL = "<KEEP>"
UPDATE_CONTROL = "<UPDATE>"
KEEP_GATE_TOKEN = "k"
UPDATE_GATE_TOKEN = "u"
TARGET_VERIFICATION_MAX_NEW = 1
DEFAULT_ACCEPTANCE_SEED = 42
_PROMPTS = Path(__file__).parent / "prompts"
TEMPLATE = (_PROMPTS / "memory_update.txt").read_text(encoding="utf-8")
TEMPLATE_FINAL = (
    (_PROMPTS / "final_answer.txt")
    .read_text(encoding="utf-8")
    .replace(r"\boxed{}", r"\boxed{{}}")
)
FORWARD_GATE_TEMPLATE = (_PROMPTS / "gate.txt").read_text(encoding="utf-8")
STREAMING_BARRIER_TEMPLATE = FORWARD_GATE_TEMPLATE


class DecisionKind(Enum):
    KEEP = "keep"
    UPDATE = "update"
    INVALID = "invalid"


@dataclass(frozen=True)
class SpeculativeDecision:
    kind: DecisionKind
    memory: Optional[str]
    raw_text: str
    invalid_reason: Optional[str] = None


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    usage: Optional[Mapping[str, Any]] = None


class ChatClient(Protocol):
    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionResult: ...


class ControlContinuation(Protocol):
    """One streaming control branch whose selected request can be continued."""

    text: str
    usage: Optional[Mapping[str, Any]]

    async def read_control(self) -> str:
        """Read through one complete ``<KEEP>`` or ``<UPDATE>`` tag."""
        ...

    async def finish(self) -> ChatCompletionResult:
        """Continue this same request through its final memory output."""
        ...

    async def close(self) -> Optional[Mapping[str, Any]]:
        """Cancel/release this branch and return its latest cumulative usage."""
        ...


class StreamingChatClient(ChatClient, Protocol):
    async def open_control_stream(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ControlContinuation: ...


class ChatCompletionError(RuntimeError):
    """A chat-completion request failed or returned an unusable response."""


@dataclass
class UsageTotals:
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    responses_with_usage: int = 0
    responses_total: int = 0
    responses_missing_output_tokens: int = 0

    def add(self, usage: Optional[Mapping[str, Any]]) -> None:
        self.responses_total += 1
        if isinstance(usage, Mapping):
            self.responses_with_usage += 1
        else:
            usage = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                value = None
                if name == "completion_tokens":
                    self.responses_missing_output_tokens += 1
            setattr(self, name, _sum_usage(getattr(self, name), value))

    def to_json(self) -> dict[str, Optional[int]]:
        return asdict(self)


def _sum_usage(*values: Optional[int]) -> Optional[int]:
    return None if any(value is None for value in values) else sum(values)


def _usage_snapshot(usage: UsageTotals) -> dict[str, Optional[int]]:
    return usage.to_json()


def _usage_delta(
    before: Mapping[str, Optional[int]], usage: UsageTotals
) -> dict[str, Optional[int]]:
    after = _usage_snapshot(usage)
    return {
        name: None
        if after[name] is None or before.get(name, 0) is None
        else after[name] - before.get(name, 0)
        for name in after
    }


@dataclass
class RunStats:
    full_update_calls: int = 0
    committed_update_outputs: int = 0
    fallback_update_calls: int = 0
    gate_requests: int = 0
    skipped_chunks: int = 0
    discarded_outputs: int = 0
    discarded_update_outputs: int = 0
    control_tokens: int = 0
    continued_update_streams: int = 0
    cancelled_streams: int = 0
    speculative_rounds: int = 0
    issued_window_sum: int = 0
    min_issued_window: int = 0
    max_issued_window: int = 0
    window_strategy: str = "fixed"
    initial_window: int = 0
    final_window: int = 0
    controller_state: dict[str, Any] = field(default_factory=dict)

    def record_window(self, issued_window: int) -> None:
        self.speculative_rounds += 1
        self.issued_window_sum += issued_window
        if self.min_issued_window == 0:
            self.min_issued_window = issued_window
        else:
            self.min_issued_window = min(self.min_issued_window, issued_window)
        self.max_issued_window = max(self.max_issued_window, issued_window)


@dataclass
class SpeculativeBlockTrace:
    cursor_before: int
    chunk_indices: List[int]
    raw_outputs: List[str]
    parsed_decisions: List[dict[str, Optional[str]]]
    accepted_keep_prefix: int
    committed_update_index: Optional[int]
    used_fallback: bool
    cursor_after: int
    memory_before: str
    memory_after: str
    target_window_before: int = 0
    issued_window: int = 0
    target_window_after: int = 0
    block_wall_seconds: float = 0.0
    controller_state: dict[str, Any] = field(default_factory=dict)
    fallback_chunk_index: Optional[int] = None
    fallback_chunk_text: Optional[str] = None
    fallback_reason: Optional[str] = None
    fallback_raw_output: Optional[str] = None
    fallback_full_update_wall_seconds: Optional[float] = None
    fallback_usage: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class FallbackEvent:
    """Record of a full-update fallback."""

    chunk_index: int
    current_chunk: str
    trigger_reason: str
    raw_abnormal_output: str
    memory_before: str
    memory_after: str


@dataclass
class SpeculativeStepResult:
    memory: str
    cursor: int
    trace: SpeculativeBlockTrace
    fallback_event: Optional[FallbackEvent] = None


@dataclass
class _ControlBranch:
    stream: Optional[ControlContinuation]
    control: str
    raw_text: str


def parse_speculative_response(text: str) -> SpeculativeDecision:
    """Parse the two-format protocol without accepting approximate matches."""
    raw_text = text if isinstance(text, str) else str(text)
    stripped = raw_text.strip()
    if stripped == "<KEEP>":
        return SpeculativeDecision(DecisionKind.KEEP, None, raw_text)
    if stripped.count("<UPDATE>") != 1 or stripped.count("</UPDATE>") != 1:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "invalid_update_tag_count"
        )
    match = re.fullmatch("<UPDATE>(.*?)</UPDATE>", stripped, flags=re.DOTALL)
    if match is None:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "protocol_format_mismatch"
        )
    memory = match.group(1).strip()
    prefix = "Updated memory:"
    if memory.startswith(prefix):
        memory = memory[len(prefix) :].strip()
    if not memory:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "empty_update_memory"
        )
    return SpeculativeDecision(DecisionKind.UPDATE, memory, raw_text)


def parse_streaming_barrier_response(text: str) -> SpeculativeDecision:
    """Parse a completed selected ``<UPDATE>`` branch."""
    raw_text = text if isinstance(text, str) else str(text)
    stripped = raw_text.strip()
    if stripped == KEEP_CONTROL:
        return SpeculativeDecision(DecisionKind.KEEP, None, raw_text)
    if stripped.startswith(UPDATE_CONTROL):
        memory = stripped
    else:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "invalid_streaming_control_prefix"
        )
    if not memory:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "empty_streaming_update_memory"
        )
    if memory.startswith("<UPDATE>"):
        parsed = parse_speculative_response(memory)
        if parsed.kind is not DecisionKind.UPDATE:
            return SpeculativeDecision(
                DecisionKind.INVALID,
                None,
                raw_text,
                parsed.invalid_reason or "invalid_streaming_update_protocol",
            )
        memory = parsed.memory or ""
    prefix = "Updated memory:"
    if memory.startswith(prefix):
        memory = memory[len(prefix) :].strip()
    if not memory:
        return SpeculativeDecision(
            DecisionKind.INVALID, None, raw_text, "empty_streaming_update_memory"
        )
    return SpeculativeDecision(DecisionKind.UPDATE, memory, raw_text)


async def full_memory_update(
    client: ChatClient,
    problem: str,
    memory: str,
    chunk: str,
    max_new: int,
    usage: UsageTotals,
) -> str:
    prompt = TEMPLATE.format(prompt=problem, memory=memory, chunk=chunk)
    result = await client.complete(prompt, max_new, purpose="full_update")
    usage.add(result.usage)
    if not result.text.strip():
        raise ChatCompletionError("full memory update returned empty content")
    return result.text


async def generate_final_answer(
    client: ChatClient, problem: str, memory: str, max_new: int, usage: UsageTotals
) -> str:
    prompt = TEMPLATE_FINAL.format(prompt=problem, memory=memory)
    result = await client.complete(prompt, max_new, purpose="final_answer")
    usage.add(result.usage)
    if not result.text.strip():
        raise ChatCompletionError("final answer returned empty content")
    return result.text


async def _open_control_branch(
    client: StreamingChatClient, problem: str, memory: str, chunk: str, max_new: int
) -> _ControlBranch:
    prompt = STREAMING_BARRIER_TEMPLATE.format(
        prompt=problem, memory=memory, chunk=chunk
    )
    stream: Optional[ControlContinuation] = None
    try:
        stream = await client.open_control_stream(
            prompt,
            max_new + 32,
            purpose="streaming_barrier_control",
            temperature=0.0,
            top_p=1.0,
            problem=problem,
            memory=memory,
            chunk=chunk,
        )
        control = await stream.read_control()
        raw_text = stream.text
        return _ControlBranch(stream, control, raw_text)
    except Exception as exc:
        raw_text = f"<request failed: {type(exc).__name__}: {exc}>"
        return _ControlBranch(stream, "", raw_text)


async def _close_control_branches(
    branches: Sequence[_ControlBranch], usage: UsageTotals
) -> None:

    async def close_one(branch: _ControlBranch) -> None:
        if branch.stream is None:
            return
        try:
            usage.add(await branch.stream.close())
            branch.raw_text = branch.stream.text or branch.raw_text
        except Exception as exc:
            branch.raw_text += f" <close failed: {type(exc).__name__}: {exc}>"

    await asyncio.gather(*(close_one(branch) for branch in branches))


async def streaming_barrier_memory_step(
    client: StreamingChatClient,
    problem: str,
    chunks: Sequence[str],
    memory: str,
    cursor: int,
    lookahead: int,
    max_new: int,
    stats: RunStats,
    usage: UsageTotals,
    verbose: bool = False,
) -> SpeculativeStepResult:
    """Decode control tags in parallel and continue the earliest UPDATE."""
    if lookahead < 1:
        raise ValueError("lookahead must be >= 1")
    if cursor < 0 or cursor >= len(chunks):
        raise ValueError("cursor must identify an unprocessed chunk")
    if not callable(getattr(client, "open_control_stream", None)):
        raise TypeError("fixed-window inference requires open_control_stream support")
    block_started = time.perf_counter()
    indices = list(range(cursor, min(cursor + lookahead, len(chunks))))
    stream_capacity = getattr(client, "max_stream_concurrency", None)
    if (
        isinstance(stream_capacity, int)
        and (not isinstance(stream_capacity, bool))
        and (stream_capacity < len(indices))
    ):
        raise ValueError(
            f"streaming barrier requires one live concurrency slot per issued branch: issued_window={len(indices)}, capacity={stream_capacity}"
        )
    memory_before = memory
    branches = await asyncio.gather(
        *(
            _open_control_branch(client, problem, memory_before, chunks[index], max_new)
            for index in indices
        )
    )
    stats.gate_requests += len(branches)
    stats.control_tokens += sum(
        (branch.control in (KEEP_CONTROL, UPDATE_CONTROL) for branch in branches)
    )
    decisions = [
        SpeculativeDecision(
            DecisionKind.KEEP
            if branch.control == KEEP_CONTROL
            else DecisionKind.UPDATE
            if branch.control == UPDATE_CONTROL
            else DecisionKind.INVALID,
            None,
            branch.raw_text,
            None
            if branch.control in (KEEP_CONTROL, UPDATE_CONTROL)
            else "streaming_control_request_failed"
            if branch.raw_text.lstrip().startswith("<request failed:")
            else "invalid_or_missing_streaming_control_tag",
        )
        for branch in branches
    ]
    accepted_keep_prefix = 0
    committed_update_index: Optional[int] = None
    used_fallback = False
    fallback_event: Optional[FallbackEvent] = None
    fallback_chunk_index: Optional[int] = None
    fallback_chunk_text: Optional[str] = None
    fallback_reason: Optional[str] = None
    fallback_raw_output: Optional[str] = None
    fallback_full_update_wall_seconds: Optional[float] = None
    fallback_usage: Optional[dict[str, int]] = None
    cursor_after = cursor
    memory_after = memory_before
    barrier_offset: Optional[int] = None
    for offset, (index, decision) in enumerate(zip(indices, decisions)):
        if decision.kind is DecisionKind.KEEP:
            accepted_keep_prefix += 1
            stats.skipped_chunks += 1
            cursor_after = index + 1
            continue
        barrier_offset = offset
        break
    if barrier_offset is None:
        await _close_control_branches(branches, usage)
        cursor_after = indices[-1] + 1
    else:
        barrier_index = indices[barrier_offset]
        selected = decisions[barrier_offset]
        fallback_reason = (
            "streaming_control_request_failed"
            if selected.raw_text.lstrip().startswith("<request failed:")
            else "invalid_or_missing_streaming_control_tag"
        )
        barrier_usage_recorded = False
        stale_branches = branches[barrier_offset + 1 :]
        stats.discarded_outputs += len(stale_branches)
        stats.discarded_update_outputs += sum(
            (branch.control == UPDATE_CONTROL for branch in stale_branches)
        )
        stats.cancelled_streams += sum(
            (branch.stream is not None for branch in stale_branches)
        )
        await _close_control_branches(
            [*branches[:barrier_offset], *stale_branches], usage
        )
        if selected.kind is DecisionKind.UPDATE:
            selected_stream = branches[barrier_offset].stream
            if selected_stream is None:
                raise AssertionError(
                    "UPDATE control must have a live continuation stream"
                )
            stats.continued_update_streams += 1
            try:
                completion = await selected_stream.finish()
                usage.add(completion.usage)
                barrier_usage_recorded = True
                branches[barrier_offset].raw_text = completion.text
                selected = parse_streaming_barrier_response(completion.text)
                if selected.kind is not DecisionKind.UPDATE:
                    fallback_reason = "malformed_streaming_update_response"
            except Exception as exc:
                usage.add(await selected_stream.close())
                barrier_usage_recorded = True
                branches[barrier_offset].raw_text = (
                    selected_stream.text
                    or f"<continuation failed: {type(exc).__name__}: {exc}>"
                )
                selected = SpeculativeDecision(
                    DecisionKind.INVALID,
                    None,
                    branches[barrier_offset].raw_text,
                    "streaming_update_continuation_failed",
                )
                fallback_reason = "streaming_update_continuation_failed"
            decisions[barrier_offset] = selected
        if selected.kind is DecisionKind.UPDATE:
            if selected.memory is None:
                raise AssertionError("parsed streaming UPDATE must contain memory")
            memory_after = selected.memory
            committed_update_index = barrier_index
            stats.committed_update_outputs += 1
        else:
            if branches[barrier_offset].stream is not None and (
                not barrier_usage_recorded
            ):
                usage.add(await branches[barrier_offset].stream.close())
            used_fallback = True
            stats.fallback_update_calls += 1
            fallback_chunk_index = barrier_index
            fallback_chunk_text = chunks[barrier_index]
            fallback_raw_output = branches[barrier_offset].raw_text
            usage_before_fallback = _usage_snapshot(usage)
            fallback_started = time.perf_counter()
            memory_after = await full_memory_update(
                client, problem, memory_before, chunks[barrier_index], max_new, usage
            )
            fallback_full_update_wall_seconds = time.perf_counter() - fallback_started
            fallback_usage = _usage_delta(usage_before_fallback, usage)
            fallback_event = FallbackEvent(
                chunk_index=barrier_index,
                current_chunk=chunks[barrier_index],
                trigger_reason=fallback_reason,
                raw_abnormal_output=branches[barrier_offset].raw_text,
                memory_before=memory_before,
                memory_after=memory_after,
            )
        cursor_after = barrier_index + 1
    for decision_index, (branch, decision) in enumerate(zip(branches, decisions)):
        if branch.raw_text != decision.raw_text:
            decisions[decision_index] = SpeculativeDecision(
                decision.kind, decision.memory, branch.raw_text, decision.invalid_reason
            )
    trace = SpeculativeBlockTrace(
        cursor_before=cursor,
        chunk_indices=indices,
        raw_outputs=[branch.raw_text for branch in branches],
        parsed_decisions=[
            {
                "kind": decision.kind.value,
                "memory": decision.memory,
                "invalid_reason": decision.invalid_reason,
            }
            for decision in decisions
        ],
        accepted_keep_prefix=accepted_keep_prefix,
        committed_update_index=committed_update_index,
        used_fallback=used_fallback,
        cursor_after=cursor_after,
        memory_before=memory_before,
        memory_after=memory_after,
        target_window_before=lookahead,
        issued_window=len(indices),
        target_window_after=lookahead,
        block_wall_seconds=time.perf_counter() - block_started,
        fallback_chunk_index=fallback_chunk_index,
        fallback_chunk_text=fallback_chunk_text,
        fallback_reason=fallback_reason if used_fallback else None,
        fallback_raw_output=fallback_raw_output,
        fallback_full_update_wall_seconds=fallback_full_update_wall_seconds,
        fallback_usage=fallback_usage,
    )
    if verbose:
        labels = [decision.kind.value.upper() for decision in decisions]
        discarded_indices = (
            [] if barrier_offset is None else indices[barrier_offset + 1 :]
        )
        print(f"streaming barrier block chunks={indices}")
        print(f"  controls={[branch.control or 'INVALID' for branch in branches]}")
        print(f"  decisions={labels}")
        print(f"  accepted_contiguous_keep={accepted_keep_prefix}")
        print(f"  committed_update_index={committed_update_index}")
        print(f"  cancelled_chunk_streams={discarded_indices}")
        print(f"  fallback={used_fallback}")
        print(f"  memory_length={len(memory_before)} -> {len(memory_after)}")
        if memory_before != memory_after:
            print(f"  memory={memory_after[:200]}")
    return SpeculativeStepResult(memory_after, cursor_after, trace, fallback_event)


@dataclass
class RoleRoutingStats:
    gate_requests: int = 0
    forward_gate_requests: int = 0
    replay_gate_requests: int = 0
    invalid_gate_outputs: int = 0
    gate_request_failures: int = 0
    system2_requests: int = 0
    authoritative_update_attempts: int = 0
    authoritative_update_successes: int = 0
    authoritative_forward_update_attempts: int = 0
    authoritative_forward_update_successes: int = 0
    authoritative_replay_update_attempts: int = 0
    authoritative_replay_update_successes: int = 0
    system2_purpose_counts: dict[str, int] = field(default_factory=dict)


def _merge_usage(target: UsageTotals, source: UsageTotals) -> None:
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        setattr(target, name, _sum_usage(getattr(target, name), getattr(source, name)))
    for name in (
        "responses_with_usage",
        "responses_total",
        "responses_missing_output_tokens",
    ):
        setattr(target, name, getattr(target, name) + getattr(source, name))


@dataclass(frozen=True)
class KUProbabilityResult(ChatCompletionResult):
    """A strict k/u decision plus probabilities normalized over {k, u}."""

    keep_probability: float = 0.0
    update_probability: float = 0.0
    keep_logprob: float = float("-inf")
    update_logprob: float = float("-inf")


class KUProbabilityClient(Protocol):
    model: str

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> KUProbabilityResult: ...


def _validate_probability_result(result: KUProbabilityResult, role: str) -> None:
    if result.text not in (KEEP_GATE_TOKEN, UPDATE_GATE_TOKEN):
        raise ChatCompletionError(f"{role} returned invalid decision {result.text!r}")
    keep_p = result.keep_probability
    update_p = result.update_probability
    if not all(
        (math.isfinite(value) and 0.0 <= value <= 1.0 for value in (keep_p, update_p))
    ):
        raise ChatCompletionError(f"{role} returned invalid k/u probabilities")
    if not math.isclose(keep_p + update_p, 1.0, rel_tol=1e-06, abs_tol=1e-06):
        raise ChatCompletionError(
            f"{role} k/u probabilities do not sum to one: k={keep_p}, u={update_p}"
        )


@dataclass
class RoutingStats(RoleRoutingStats):
    small_keep_decisions: int = 0
    small_update_proposals: int = 0
    target_verification_attempts: int = 0
    target_verification_successes: int = 0
    target_verification_failures: int = 0
    target_keep_argmax_outputs: int = 0
    target_update_argmax_outputs: int = 0
    accepted_update_proposals: int = 0
    rejected_update_proposals: int = 0
    sampling_rejections: int = 0
    verifier_error_rejections: int = 0
    memory_generation_blocked_by_rejection: int = 0
    accepted_updates_cancelled_as_stale: int = 0


class UAcceptanceContinuation:
    """Hard barrier around one system1 proposal."""

    def __init__(
        self,
        gate_result: KUProbabilityResult,
        router: "UOnlySpeculativeAcceptanceClient",
        sequence: int,
        route: str,
        problem: str,
        memory: str,
        chunk: str,
    ) -> None:
        self._gate_result = gate_result
        self._router = router
        self._sequence = sequence
        self._route = route
        self._problem = problem
        self._memory = memory
        self._chunk = chunk
        self._control: Optional[str] = None
        self._accepted = False
        self._finished = False
        self._closed = False
        self._gate_usage_deferred = False
        self._event: Optional[dict[str, Any]] = None
        self.text = gate_result.text
        self.usage = gate_result.usage

    async def read_control(self) -> str:
        if self._control is not None:
            return self._control
        try:
            _validate_probability_result(self._gate_result, "small Gate")
        except ChatCompletionError:
            self._router.stats.invalid_gate_outputs += 1
            self._control = ""
            self._event = self._router.record_event(
                self._sequence,
                self._route,
                self._gate_result.text or "INVALID",
                outcome="invalid_small_gate",
            )
            return self._control
        if self._gate_result.text == KEEP_GATE_TOKEN:
            self._router.stats.small_keep_decisions += 1
            self._control = KEEP_CONTROL
            self._event = self._router.record_event(
                self._sequence,
                self._route,
                KEEP_GATE_TOKEN,
                q_small_u=self._gate_result.update_probability,
                outcome="direct_keep",
                target_requested=False,
                memory_generation="not_requested",
            )
            return self._control
        self._router.stats.small_update_proposals += 1
        try:
            target_result = await self._router.verify_update(
                self._problem, self._memory, self._chunk
            )
            _validate_probability_result(target_result, "large verifier")
        except Exception as exc:
            self._router.stats.target_verification_failures += 1
            self._router.stats.rejected_update_proposals += 1
            self._router.stats.verifier_error_rejections += 1
            self._router.stats.memory_generation_blocked_by_rejection += 1
            self._control = KEEP_CONTROL
            self._event = self._router.record_event(
                self._sequence,
                self._route,
                UPDATE_GATE_TOKEN,
                q_small_u=self._gate_result.update_probability,
                outcome="rejected_verifier_error",
                target_requested=True,
                verifier_error=f"{type(exc).__name__}: {exc}",
                memory_generation="blocked",
            )
            return self._control
        q_update = self._gate_result.update_probability
        p_update = target_result.update_probability
        if q_update <= 0.0:
            raise ChatCompletionError(
                "small Gate proposed u with zero update probability"
            )
        acceptance_probability = min(1.0, p_update / q_update)
        uniform_sample = self._router._acceptance_draws[self._sequence]
        self._accepted = uniform_sample < acceptance_probability
        if self._accepted:
            self._router.stats.accepted_update_proposals += 1
            self._control = UPDATE_CONTROL
            outcome = "accepted"
            memory_generation = "pending_document_order_commit"
        else:
            self._router.stats.rejected_update_proposals += 1
            self._router.stats.sampling_rejections += 1
            self._router.stats.memory_generation_blocked_by_rejection += 1
            self._control = KEEP_CONTROL
            outcome = "rejected_sampling"
            memory_generation = "blocked"
        self._event = self._router.record_event(
            self._sequence,
            self._route,
            UPDATE_GATE_TOKEN,
            q_small_u=q_update,
            p_large_u=p_update,
            target_argmax=target_result.text,
            acceptance_probability=acceptance_probability,
            uniform_sample=uniform_sample,
            accepted=self._accepted,
            outcome=outcome,
            target_requested=True,
            memory_generation=memory_generation,
        )
        return self._control

    async def finish(self) -> ChatCompletionResult:
        if self._control != UPDATE_CONTROL or not self._accepted:
            raise ChatCompletionError(
                "memory generation requested without an accepted u"
            )
        result = await self._router.authoritative_update(
            self._route, self._problem, self._memory, self._chunk
        )
        self._finished = True
        self.text = f"<UPDATE>\n{result.text.strip()}\n</UPDATE>"
        self.usage = result.usage
        if not self._gate_usage_deferred:
            self._router.defer_usage(self._gate_result.usage)
            self._gate_usage_deferred = True
        if self._event is not None:
            self._event["memory_generation"] = "completed"
        return ChatCompletionResult(self.text, result.usage)

    async def close(self) -> Optional[Mapping[str, Any]]:
        if self._closed:
            return None
        self._closed = True
        if self._accepted and (not self._finished):
            self._router.stats.accepted_updates_cancelled_as_stale += 1
            if self._event is not None:
                self._event["memory_generation"] = "cancelled_stale"
        return None if self._gate_usage_deferred else self._gate_result.usage


class UOnlySpeculativeAcceptanceClient:
    """Route k directly and hard-gate every proposed u before memory writing."""

    def __init__(
        self,
        gate_client: KUProbabilityClient,
        target_verifier_client: KUProbabilityClient,
        system2_client: StreamingChatClient,
        update_max_tokens: int,
        acceptance_seed: int = DEFAULT_ACCEPTANCE_SEED,
        record_events: bool = False,
        selection_mode: str = "sample",
        decision_delta: float = -3.0,
    ) -> None:
        self.gate_client = gate_client
        self.target_verifier_client = target_verifier_client
        self.system2_client = system2_client
        self.selection_mode = selection_mode
        self.decision_delta = decision_delta
        self._proposal_rng = random.Random(acceptance_seed)
        self._acceptance_rng = random.Random(acceptance_seed ^ 0x9E3779B97F4A7C15)
        self._acceptance_draws = {}
        self.update_max_tokens = update_max_tokens
        self.acceptance_seed = int(acceptance_seed)
        self.record_events = record_events
        self.max_stream_concurrency = getattr(
            gate_client, "max_stream_concurrency", None
        )
        self.gate_usage = UsageTotals()
        self.system2_usage = UsageTotals()
        self.generation_usage = UsageTotals()
        self.target_verifier_usage = UsageTotals()
        self.deferred_usage = UsageTotals()
        self.stats = RoutingStats()
        self.events: list[dict[str, Any]] = []
        self._next_sequence = 0

    def _record_system2_request(self, purpose: str) -> None:
        self.stats.system2_requests += 1
        self.stats.system2_purpose_counts[purpose] = (
            self.stats.system2_purpose_counts.get(purpose, 0) + 1
        )

    def defer_usage(self, usage: Optional[Mapping[str, Any]]) -> None:
        self.deferred_usage.add(usage)

    def record_event(
        self, sequence: int, route: str, draft_decision: str, **fields: Any
    ) -> dict[str, Any]:
        event = {
            "sequence": sequence,
            "route": route,
            "draft_decision": draft_decision,
            **fields,
        }
        if self.record_events:
            self.events.append(event)
        return event

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionResult:
        self._record_system2_request(purpose)
        try:
            result = await self.system2_client.complete(
                prompt, max_tokens, purpose, temperature, top_p
            )
        except Exception:
            self.system2_usage.add(None)
            self.generation_usage.add(None)
            raise
        self.system2_usage.add(result.usage)
        self.generation_usage.add(result.usage)
        return result

    async def open_control_stream(
        self,
        prompt,
        max_tokens,
        purpose,
        temperature=None,
        top_p=None,
        *,
        problem,
        memory,
        chunk,
    ):
        if purpose != "streaming_barrier_control":
            raise ChatCompletionError(f"unsupported gate purpose: {purpose}")
        sequence = self._next_sequence
        self._next_sequence += 1
        proposal_uniform = self._proposal_rng.random()
        self._acceptance_draws[sequence] = self._acceptance_rng.random()
        self.stats.gate_requests += 1
        self.stats.forward_gate_requests += 1
        try:
            result = await self.gate_client.complete(prompt, 1, purpose=purpose)
        except Exception:
            self.stats.gate_request_failures += 1
            self.gate_usage.add(None)
            raise
        self.gate_usage.add(result.usage)
        try:
            _validate_probability_result(result, "system1")
        except ChatCompletionError:
            self.stats.invalid_gate_outputs += 1
            raise
        action = (
            proposal_uniform < result.update_probability
            if self.selection_mode == "sample"
            else result.update_logprob - result.keep_logprob >= self.decision_delta
        )
        result = replace(result, text="u" if action else "k")
        return UAcceptanceContinuation(
            result, self, sequence, "forward", problem, memory, chunk
        )

    async def verify_update(
        self, problem: str, memory: str, chunk: str
    ) -> KUProbabilityResult:
        self.stats.target_verification_attempts += 1
        self._record_system2_request("target_update_verification")
        prompt = FORWARD_GATE_TEMPLATE.format(
            prompt=problem, memory=memory, chunk=chunk
        )
        try:
            result = await self.target_verifier_client.complete(
                prompt,
                TARGET_VERIFICATION_MAX_NEW,
                purpose="target_update_verification",
                temperature=0.0,
                top_p=1.0,
            )
        except Exception:
            self.target_verifier_usage.add(None)
            self.system2_usage.add(None)
            self.defer_usage(None)
            raise
        self.target_verifier_usage.add(result.usage)
        self.system2_usage.add(result.usage)
        self.defer_usage(result.usage)
        _validate_probability_result(result, "large verifier")
        self.stats.target_verification_successes += 1
        if result.text == KEEP_GATE_TOKEN:
            self.stats.target_keep_argmax_outputs += 1
        else:
            self.stats.target_update_argmax_outputs += 1
        return result

    async def authoritative_update(
        self, route: str, problem: str, memory: str, chunk: str
    ) -> ChatCompletionResult:
        if route == "forward":
            template = TEMPLATE
            purpose = "authoritative_forward_update"
            self.stats.authoritative_forward_update_attempts += 1
        elif route == "replay":
            template = TEMPLATE
            purpose = "authoritative_replay_update"
            self.stats.authoritative_replay_update_attempts += 1
        else:
            raise ChatCompletionError(f"unsupported authoritative route: {route}")
        self.stats.authoritative_update_attempts += 1
        self._record_system2_request(purpose)
        bound = template.replace("{max_memory_tokens}", str(self.update_max_tokens))
        try:
            result = await self.system2_client.complete(
                bound.format(prompt=problem, memory=memory, chunk=chunk),
                self.update_max_tokens,
                purpose=purpose,
            )
        except Exception:
            self.system2_usage.add(None)
            self.generation_usage.add(None)
            raise
        self.system2_usage.add(result.usage)
        self.generation_usage.add(result.usage)
        if not result.text.strip():
            raise ChatCompletionError(f"{purpose} returned an empty replacement memory")
        self.stats.authoritative_update_successes += 1
        if route == "forward":
            self.stats.authoritative_forward_update_successes += 1
        else:
            self.stats.authoritative_replay_update_successes += 1
        return result

    def merge_deferred_usage(self, usage: UsageTotals) -> None:
        _merge_usage(usage, self.deferred_usage)


async def run_inference(
    system2_client,
    gate_client,
    target_verifier_client,
    problem,
    chunks,
    *,
    seed=42,
    selection_mode="sample",
    decision_delta=-3.0,
    lookahead=4,
    memory_tokens=1024,
    answer_tokens=1024,
    method="memduet",
):
    """Bootstrap, verify a fixed window, commit its earliest update, and answer."""
    if selection_mode not in {"sample", "threshold"}:
        raise ValueError("selection_mode must be sample or threshold")
    if not math.isfinite(decision_delta):
        raise ValueError("decision_delta must be finite")
    if min(lookahead, memory_tokens, answer_tokens) < 1:
        raise ValueError("window and output limits must be positive")
    if not chunks or method not in {"memduet", "all_chunks"}:
        raise ValueError("nonempty chunks and a supported memory method are required")
    started = time.perf_counter()
    router = UOnlySpeculativeAcceptanceClient(
        gate_client,
        target_verifier_client,
        system2_client,
        memory_tokens,
        acceptance_seed=seed,
        record_events=True,
        selection_mode=selection_mode,
        decision_delta=decision_delta,
    )
    usage = UsageTotals()
    stats = RunStats(
        window_strategy="fixed", initial_window=lookahead, final_window=lookahead
    )
    traces, fallbacks = [], []
    stats.full_update_calls += 1
    memory = await full_memory_update(
        router, problem, "No previous memory", chunks[0], memory_tokens, usage
    )
    cursor = 1
    while cursor < len(chunks):
        if method == "all_chunks":
            memory = await full_memory_update(
                router, problem, memory, chunks[cursor], memory_tokens, usage
            )
            stats.full_update_calls += 1
            cursor += 1
            continue
        step = await streaming_barrier_memory_step(
            router,
            problem,
            chunks,
            memory,
            cursor,
            lookahead,
            memory_tokens,
            stats,
            usage,
        )
        stats.record_window(step.trace.issued_window)
        memory, cursor = step.memory, step.cursor
        traces.append(asdict(step.trace))
        if step.fallback_event is not None:
            fallbacks.append(asdict(step.fallback_event))
    accounted = (
        stats.full_update_calls
        + stats.skipped_chunks
        + stats.committed_update_outputs
        + stats.fallback_update_calls
    )
    if accounted != len(chunks):
        raise AssertionError("forward transition count differs from chunk count")
    answer = await generate_final_answer(router, problem, memory, answer_tokens, usage)
    router.merge_deferred_usage(usage)
    return {
        "final_answer": answer,
        "final_memory": memory,
        "chunk_count": len(chunks),
        "latency_seconds": time.perf_counter() - started,
        "traces": traces,
        "fallback_events": fallbacks,
        "stats": asdict(stats),
        "routing_stats": asdict(router.stats),
        "routing_events": sorted(router.events, key=lambda e: e["sequence"]),
        "usage": {
            "system1_input_tokens": router.gate_usage.prompt_tokens,
            "system1_output_tokens": router.gate_usage.completion_tokens,
            "system2_input_tokens": router.system2_usage.prompt_tokens,
            "system2_output_tokens": router.system2_usage.completion_tokens,
            "system2_verification_output_tokens": router.target_verifier_usage.completion_tokens,
            "system2_generation_output_tokens": router.generation_usage.completion_tokens,
            "total_output_tokens": _sum_usage(
                router.gate_usage.completion_tokens,
                router.system2_usage.completion_tokens,
            ),
        },
        "usage_responses": {
            "system1": router.gate_usage.to_json(),
            "system2": router.system2_usage.to_json(),
            "verifier": router.target_verifier_usage.to_json(),
        },
    }
