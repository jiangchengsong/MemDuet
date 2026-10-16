"""Differentiable fixed-window trajectories for policy training.

system1 proposals and system2 verification share the same prompt and memory.
"""

from __future__ import annotations

import math
import random
from time import perf_counter
from .prompting import render_answer, render_gate, render_memory
from .types import Problem, Trajectory, RoundEvent, SelectionEvent, VerificationEvent


def _scores(backend, prompts):
    scores = (
        backend.score_many(prompts)
        if hasattr(backend, "score_many")
        else [backend.score(p) for p in prompts]
    )
    if len(scores) != len(prompts):
        raise ValueError("backend score count differs from requested prompt count")
    for prompt, score in zip(prompts, scores):
        if score.prompt != prompt:
            raise ValueError("backend score prompt differs from requested prompt")
        if (
            not math.isfinite(score.update_probability)
            or not 0 <= score.update_probability <= 1
        ):
            raise ValueError("gate probability must be finite and between zero and one")
        if not isinstance(score.output_tokens, int) or score.output_tokens < 1:
            raise ValueError("gate output_tokens must be a positive integer")
    return scores


def run_trajectory(
    problem: Problem,
    system1,
    system2,
    rollout: dict,
    *,
    seed: int,
    selection_mode: str = "sample",
) -> Trajectory:
    if selection_mode not in {"sample", "threshold"}:
        raise ValueError("selection_mode must be sample or threshold")
    options = dict(
        window_size=4,
        decision_delta=-3.0,
        memory_tokens=1024,
        answer_tokens=1024,
        temperature=1.0,
        top_p=1.0,
    )
    options.update(rollout)
    for name in ("window_size", "memory_tokens", "answer_tokens"):
        if (
            isinstance(options[name], bool)
            or not isinstance(options[name], int)
            or options[name] <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("decision_delta", "temperature", "top_p"):
        if not math.isfinite(options[name]):
            raise ValueError(f"{name} must be finite")
    if options["temperature"] < 0 or not 0 < options["top_p"] <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0, 1]")
    if not problem.chunks:
        raise ValueError("trajectory requires nonempty chunks")
    rng = random.Random(seed)
    started = perf_counter()
    rounds, generations = [], []
    usage = dict(
        system1_output_tokens=0,
        system2_verification_output_tokens=0,
        system2_generation_output_tokens=0,
    )

    def generate(prompt, kind):
        event = system2.generate(
            prompt,
            kind=kind,
            max_new_tokens=options[
                "answer_tokens" if kind == "answer" else "memory_tokens"
            ],
            temperature=options["temperature"],
            top_p=options["top_p"],
        )
        if event.kind != kind or event.prompt != prompt:
            raise ValueError(
                "backend generation does not match requested kind and prompt"
            )
        generations.append(event)
        usage["system2_generation_output_tokens"] += len(event.token_ids)
        return event.text

    memory = generate(
        render_memory(problem.question, "No previous memory", problem.chunks[0]),
        "bootstrap",
    )
    start = 1
    while start < len(problem.chunks):
        end = min(start + options["window_size"], len(problem.chunks))
        prompts = [
            render_gate(problem.question, memory, problem.chunks[i])
            for i in range(start, end)
        ]
        scores = _scores(system1, prompts)
        selections = []
        for index, score in zip(range(start, end), scores):
            q = score.update_probability
            if selection_mode == "sample":
                action = rng.random() < q
            else:
                log_odds = (
                    -math.inf
                    if q == 0
                    else math.inf
                    if q == 1
                    else math.log(q) - math.log1p(-q)
                )
                action = log_odds >= options["decision_delta"]
            selections.append(
                SelectionEvent(index, score.prompt, score.prompt_ids, q, action)
            )
            usage["system1_output_tokens"] += score.output_tokens
        candidates = [s for s in selections if s.action]
        verification_scores = (
            _scores(system2, [s.prompt for s in candidates]) if candidates else []
        )
        verifications = []
        committed = None
        log_probability = 0.0
        for candidate, score in zip(candidates, verification_scores):
            if candidate.update_probability <= 0:
                raise ValueError("zero-probability update candidate is impossible")
            alpha = min(1.0, score.update_probability / candidate.update_probability)
            uniform = rng.random()
            accepted = uniform < alpha
            verifications.append(
                VerificationEvent(
                    candidate.chunk_index,
                    score.prompt,
                    score.prompt_ids,
                    score.update_probability,
                    acceptance_probability=alpha,
                    uniform_sample=uniform,
                    accepted=accepted,
                )
            )
            usage["system2_verification_output_tokens"] += score.output_tokens
            # Marginalize ALL suffix outcomes; they cannot affect commitment.
            if committed is None:
                if accepted:
                    committed = candidate.chunk_index
                    log_probability += math.log(alpha)
                else:
                    log_probability += math.log1p(-alpha)
        rounds.append(
            RoundEvent(
                memory, start, selections, verifications, committed, log_probability
            )
        )
        if committed is None:
            start = end
        else:
            memory = generate(
                render_memory(problem.question, memory, problem.chunks[committed]),
                "memory",
            )
            start = committed + 1
    answer = generate(render_answer(problem.question, memory), "answer")
    usage["system2_output_tokens"] = (
        usage["system2_verification_output_tokens"]
        + usage["system2_generation_output_tokens"]
    )
    usage["total_output_tokens"] = (
        usage["system1_output_tokens"] + usage["system2_output_tokens"]
    )
    return Trajectory(
        problem.id,
        list(problem.answers),
        len(problem.chunks),
        rounds,
        generations,
        answer,
        usage=usage,
        latency_seconds=perf_counter() - started,
        system1_version=str(getattr(system1, "version", "unversioned")),
        system2_version=str(getattr(system2, "version", "unversioned")),
    )
