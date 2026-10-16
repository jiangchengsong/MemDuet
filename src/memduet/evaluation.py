"""Evaluation records, cohort alignment, and sum-based comparisons."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .metrics import choice_match, exact_match, extract_answer, substring_match


def length_bin(count: int) -> str:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("source_context_tokens must be a nonnegative integer")
    if count < 32768:
        return "short"
    if count < 131072:
        return "medium"
    return "long" if count <= 262144 else "ultra"


def _index(records):
    result = {}
    for row in records:
        identifier = row["id"]
        if identifier in result:
            raise ValueError(f"duplicate question id: {identifier}")
        result[identifier] = row
    return result


def _tokens(row):
    if (
        "total_output_tokens" in row.get("usage", {})
        and row["usage"]["total_output_tokens"] is None
    ):
        return None
    values = [
        row.get("usage", {}).get(name)
        for name in ("system1_output_tokens", "system2_output_tokens")
    ]
    if any(value is None for value in values):
        return None
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError(
            "output token counts must be nonnegative integers or unavailable"
        )
    return sum(values)


def trajectory_failures(details):
    """Identify request failures separately from sampled algorithmic rejections."""
    failures = list(details.get("evaluation_errors", []))
    if details.get("evaluation_valid") is False and not failures:
        failures.append("trajectory marked invalid")
    routing = details.get("routing_stats", {})
    for name in (
        "gate_request_failures",
        "invalid_gate_outputs",
        "target_verification_failures",
        "verifier_error_rejections",
    ):
        if routing.get(name, 0):
            failures.append(name)
    if routing.get("authoritative_update_attempts", 0) > routing.get(
        "authoritative_update_successes", 0
    ):
        failures.append("failed memory generation")
    if details.get("stats", {}).get("fallback_update_calls", 0) or details.get(
        "fallback_events"
    ):
        failures.append("fallback update")
    for trace in details.get("traces", []):
        if any(
            decision.get("kind") == "invalid"
            for decision in trace.get("parsed_decisions", [])
        ):
            failures.append("invalid speculative request")
            break
    return sorted(set(failures))


def _record_failures(row):
    return trajectory_failures(row) + trajectory_failures(row.get("trajectory", {}))


def _summary(rows):
    rows = list(rows)
    latency = [r["latency_seconds"] for r in rows]
    if any(
        not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
        for x in latency
    ):
        raise ValueError("invalid measured latency")
    accuracy = [r["accuracy"] for r in rows]
    if any(x not in (0, 1) for x in accuracy):
        raise ValueError("accuracy must be binary")
    tokens = [_tokens(r) for r in rows]
    return {
        "count": len(rows),
        "accuracy": sum(accuracy) / len(rows) if rows else None,
        "accuracy_percent": 100 * sum(accuracy) / len(rows) if rows else None,
        "latency_seconds": sum(latency),
        "output_tokens": sum(tokens) if all(x is not None for x in tokens) else None,
    }


def aggregate(
    records, baseline_records=None, *, expected_ids=None, strict=False
) -> dict:
    """Compare matched IDs only. Unknown provider tokens propagate as null."""
    current = _index(records)
    baseline = _index(baseline_records) if baseline_records is not None else None
    invalid_current = {key for key, row in current.items() if _record_failures(row)}
    invalid_baseline = {
        key for key, row in (baseline or {}).items() if _record_failures(row)
    }
    if strict and (invalid_current or invalid_baseline):
        raise ValueError(
            "invalid evaluation trajectories; rerun failed questions: "
            f"method={sorted(invalid_current)}, baseline={sorted(invalid_baseline)}"
        )
    expected = None if expected_ids is None else list(expected_ids)
    if expected is not None and len(set(expected)) != len(expected):
        raise ValueError("duplicate expected cohort id")
    matched = set(current) if baseline is None else set(current) & set(baseline)
    matched -= invalid_current | invalid_baseline
    expected_set = set(expected) if expected is not None else None
    complete = (
        not (invalid_current or invalid_baseline)
        and (expected_set is None or set(current) == expected_set)
        and (baseline is None or set(current) == set(baseline))
    )
    if strict and not complete:
        raise ValueError("incomplete or mismatched paper cohort")
    selected = [current[key] for key in sorted(matched)]
    if baseline is not None:
        for row in selected:
            reference = baseline[row["id"]]
            for key in ("source_context_tokens", "source_tokenizer"):
                if (
                    row.get(key) is not None
                    and reference.get(key) is not None
                    and row[key] != reference[key]
                ):
                    raise ValueError(
                        "matched question source context provenance differs"
                    )
            if (
                "answers" in row
                and "answers" in reference
                and row["answers"] != reference["answers"]
            ):
                raise ValueError("matched question reference answers differ")
    report = {
        "matched_count": len(matched),
        "cohort_complete": complete,
        "invalid_method_ids": sorted(invalid_current),
        "invalid_baseline_ids": sorted(invalid_baseline),
        "unmatched_method_ids": sorted(set(current) - matched),
        "unmatched_baseline_ids": []
        if baseline is None
        else sorted(set(baseline) - matched),
        "missing_expected_ids": []
        if expected_set is None
        else sorted(expected_set - set(current)),
        "method": _summary(selected),
        "length_bins": {},
        "unavailable_length_ids": [],
    }
    for row in selected:
        provenance = row.get("source_tokenizer")
        identifier = (
            provenance.get("name") if isinstance(provenance, dict) else provenance
        )
        count = row.get("source_context_tokens")
        if (
            count is None
            or not isinstance(identifier, str)
            or identifier not in {"Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"}
        ):
            report["unavailable_length_ids"].append(row["id"])
            continue
        bin_name = length_bin(count)
        report["length_bins"].setdefault(bin_name, []).append(row)
    binned = report["length_bins"]
    report["length_bins"] = {name: _summary(rows) for name, rows in binned.items()}
    if baseline is not None:
        report["baseline"] = _summary(baseline[key] for key in sorted(matched))
        denominator = report["method"]["latency_seconds"]
        report["speedup"] = (
            report["baseline"]["latency_seconds"] / denominator if denominator else None
        )
        baseline_tokens, method_tokens = (
            report["baseline"]["output_tokens"],
            report["method"]["output_tokens"],
        )
        report["token_ratio"] = (
            method_tokens / baseline_tokens
            if baseline_tokens and method_tokens is not None
            else None
        )
        report["token_reduction"] = (
            1 - report["token_ratio"] if report["token_ratio"] is not None else None
        )
        report["token_reduction_percent"] = (
            100 * report["token_reduction"]
            if report["token_reduction"] is not None
            else None
        )
        for name, rows in binned.items():
            summary = report["length_bins"][name]
            reference = _summary(baseline[row["id"]] for row in rows)
            summary["baseline"] = reference
            summary["speedup"] = (
                reference["latency_seconds"] / summary["latency_seconds"]
                if summary["latency_seconds"]
                else None
            )
            summary["token_ratio"] = (
                summary["output_tokens"] / reference["output_tokens"]
                if reference["output_tokens"] and summary["output_tokens"] is not None
                else None
            )
            summary["token_reduction"] = (
                1 - summary["token_ratio"]
                if summary["token_ratio"] is not None
                else None
            )
            summary["token_reduction_percent"] = (
                100 * summary["token_reduction"]
                if summary["token_reduction"] is not None
                else None
            )
    return report


def _native_trajectory(*args, **kwargs):
    from .controller import run_trajectory

    return run_trajectory(*args, **kwargs)


def _prompt(name, **values):
    text = (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(
        encoding="utf-8"
    )
    # One-pass replacement protects braces and any placeholder-looking source text.
    import re

    return re.sub(
        r"\{(prompt|memory|chunk)\}", lambda match: values[match.group(1)], text
    )


def _baseline(problem, system2, rollout, method):
    from .types import Trajectory

    generations = []
    if method == "all_chunks":
        memory = "No previous memory"
        for index, chunk in enumerate(problem.chunks):
            event = system2.generate(
                _prompt(
                    "memory_update", prompt=problem.question, memory=memory, chunk=chunk
                ),
                kind="bootstrap" if index == 0 else "memory",
                max_new_tokens=rollout.get("memory_tokens", 1024),
                temperature=rollout.get("temperature", 1.0),
                top_p=rollout.get("top_p", 1.0),
            )
            generations.append(event)
            memory = event.text
    else:
        memory = (
            problem.source_context
            if problem.source_context is not None
            else "".join(problem.chunks)
        )
    answer = system2.generate(
        _prompt("final_answer", prompt=problem.question, memory=memory),
        kind="answer",
        max_new_tokens=rollout.get("answer_tokens", 1024),
        temperature=rollout.get("temperature", 1.0),
        top_p=rollout.get("top_p", 1.0),
    )
    generations.append(answer)
    return Trajectory(
        problem.id,
        problem.answers,
        len(problem.chunks),
        [],
        generations,
        answer.text,
        usage={
            "system1_output_tokens": 0,
            "system2_output_tokens": sum(len(e.token_ids) for e in generations),
        },
    )


def _question_seed(seed, identifier):
    payload = int.from_bytes(f"{seed}\0{identifier}".encode(), "big")
    return random.Random(payload).randrange(2**63)


@contextmanager
def _seeded(seed, enabled):
    """Use an already-loaded Torch only; restore caller CPU/CUDA/Python RNG."""
    if not enabled:
        yield
        return
    python_state = random.getstate()
    torch = sys.modules.get("torch")
    cpu_state = torch.get_rng_state() if torch is not None else None
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )
    try:
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        if torch is not None:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def evaluate(
    problems,
    system1,
    system2,
    rollout,
    *,
    output_path,
    seed=42,
    method="memduet",
    metric="em",
    memory_tokenizer=None,
    trajectory_runner=None,
    error_policy="strict",
) -> dict:
    """Evaluate preloaded backends; resume atomic JSON report on matching provenance."""
    if method not in {"memduet", "all_chunks", "full_context", "api_transfer"}:
        raise ValueError("unknown evaluation method")
    if error_policy not in {"strict", "tolerant"}:
        raise ValueError("error_policy must be strict or tolerant")
    rollout = dict(rollout)
    if method == "memduet":
        rollout.setdefault("selection_mode", "sample")
        if rollout["selection_mode"] not in {"sample", "threshold"}:
            raise ValueError("selection_mode must be sample or threshold")
    metrics = {"em": exact_match, "sub_em": substring_match, "choice": choice_match}
    if metric not in metrics:
        raise ValueError("unknown answer metric")
    problems = list(problems)
    if not problems:
        raise ValueError("empty evaluation cohort")
    _index({"id": problem.id} for problem in problems)
    config = {
        "seed": seed,
        "method": method,
        "metric": metric,
        "rollout": rollout,
        "system1": system1.provenance()
        if hasattr(system1, "provenance")
        else {"version": getattr(system1, "version", "unused")},
        "system2": system2.provenance()
        if hasattr(system2, "provenance")
        else {"version": system2.version},
    }
    if method == "api_transfer":
        if memory_tokenizer is None or not callable(
            getattr(memory_tokenizer, "encode", None)
        ):
            raise ValueError("memory_tokenizer is required for API memory accounting")
        vocabulary = getattr(memory_tokenizer, "get_vocab", None)
        serializer = getattr(
            getattr(memory_tokenizer, "backend_tokenizer", None), "to_str", None
        )
        config["memory_tokenizer"] = {
            "type": type(memory_tokenizer).__module__
            + "."
            + type(memory_tokenizer).__qualname__,
            "name": getattr(memory_tokenizer, "name_or_path", None),
            "vocabulary": vocabulary() if callable(vocabulary) else None,
            "backend": serializer() if callable(serializer) else None,
        }
    inputs = [
        {
            "id": p.id,
            "question": p.question,
            "chunks": p.chunks,
            "answers": p.answers,
            "metadata": p.metadata,
            "source_context": p.source_context,
        }
        for p in problems
    ]
    snapshot = {"config": config, "inputs": inputs}
    path = Path(output_path)
    snapshot_path = path.with_name(path.name + ".inputs.json")
    summary_config = dict(config)
    if "memory_tokenizer" in config:
        summary_config["memory_tokenizer"] = {
            key: config["memory_tokenizer"][key] for key in ("type", "name")
        }
    provenance = {
        "config": summary_config,
        "input_snapshot": snapshot_path.name,
        "question_ids": [p.id for p in problems],
    }
    report = {
        "provenance": provenance,
        "records": [],
    }
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("provenance") != provenance:
            raise ValueError("resume provenance differs from inputs/config")
        if not snapshot_path.is_file():
            raise ValueError("resume input snapshot missing")
        if json.loads(snapshot_path.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("resume provenance differs from input/tokenizer snapshot")
    existing = _index(report["records"])
    if not set(existing) <= {p.id for p in problems}:
        raise ValueError("resume records outside requested cohort")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = snapshot_path.with_name(snapshot_path.name + ".tmp")
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, snapshot_path)
    # Resolve imports before wall timing; preloaded backends exclude model loading.
    api_runner = None
    if method == "api_transfer":
        from .api import run_api_trajectory

        api_runner = run_api_trajectory
    elif method == "memduet" and trajectory_runner is None:
        # Resolve controller dependencies before measured inference starts.
        from . import controller  # noqa: F401
    for problem in problems:
        if problem.id in existing and not _record_failures(existing[problem.id]):
            continue
        report["records"] = [
            row for row in report["records"] if row["id"] != problem.id
        ]
        question_seed = _question_seed(seed, problem.id)
        with _seeded(question_seed, method != "api_transfer"):
            started = time.perf_counter()
            if trajectory_runner is not None:
                details = trajectory_runner(
                    problem, rollout, seed=question_seed, method=method
                )
                latency = time.perf_counter() - started
            elif method == "memduet":
                trajectory = _native_trajectory(
                    problem,
                    system1,
                    system2,
                    rollout,
                    seed=question_seed,
                    selection_mode=rollout["selection_mode"],
                )
                latency = time.perf_counter() - started
                details = trajectory.to_dict()
            elif method == "api_transfer":
                details = api_runner(
                    problem,
                    system1,
                    system2,
                    rollout,
                    seed=question_seed,
                    memory_tokenizer=memory_tokenizer,
                )
                latency = details["latency_seconds"]
            else:
                trajectory = _baseline(problem, system2, rollout, method)
                latency = time.perf_counter() - started
                details = trajectory.to_dict()
        failures = trajectory_failures(details)
        if failures and error_policy == "strict":
            raise ValueError(
                f"invalid evaluation trajectory for {problem.id}: {', '.join(failures)}; "
                "rerun the same command to retry this question"
            )
        text = details["final_answer"]
        prediction = (
            extract_answer(text, last_chars=max(len(text), 1))
            if r"\boxed" in text
            or (trajectory_runner is not None and metric != "choice")
            else text
        )
        row = {
            "id": problem.id,
            "answers": problem.answers,
            "prediction": prediction,
            "final_answer": text,
            "accuracy": float(metrics[metric](prediction, problem.answers)),
            "latency_seconds": latency,
            "usage": details["usage"],
            "trajectory": details,
            "evaluation_valid": not failures,
            "evaluation_errors": failures,
            "source_context_tokens": problem.metadata.get("source_context_tokens"),
            "source_tokenizer": problem.metadata.get(
                "source_tokenizer", problem.metadata.get("source_context_tokenizer")
            ),
        }
        report["records"].append(row)
        report["summary"] = aggregate(
            report["records"], expected_ids=[p.id for p in problems]
        )
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    return report
