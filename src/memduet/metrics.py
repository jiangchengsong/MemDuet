"""Paper answer metrics; no model loading or external services."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable


def normalize_answer(text: str) -> str:
    """SQuAD-style English case/punctuation/article/whitespace normalization."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _aliases(answers: str | Iterable[str]) -> list[str]:
    return [answers] if isinstance(answers, str) else list(answers)


def exact_match(prediction: str, answers: str | Iterable[str]) -> bool:
    normalized = normalize_answer(prediction)
    return bool(normalized) and any(
        normalized == normalize_answer(a) for a in _aliases(answers)
    )


def substring_match(prediction: str, answers: str | Iterable[str]) -> bool:
    """Appendix A.3: nonempty normalized answers may contain one another."""
    normalized = normalize_answer(prediction)
    return bool(normalized) and any(
        bool(gold := normalize_answer(a)) and (gold in normalized or normalized in gold)
        for a in _aliases(answers)
    )


def extract_answer(text: str, *, last_chars: int = 300) -> str:
    """Last balanced ``\\boxed{...}`` in the tail; missing/malformed => empty.

    The paper's training reward explicitly rejects missing boxes. Literal text
    remains usable through exact_match for evaluation, not through this extractor.
    """
    if not isinstance(last_chars, int) or last_chars <= 0:
        raise ValueError("last_chars must be a positive integer")
    tail = text[-last_chars:]
    position = tail.rfind(r"\boxed")
    if position < 0:
        return ""
    opening = position + len(r"\boxed")
    if opening >= len(tail) or tail[opening] != "{":
        return ""
    depth = 0
    for index in range(opening, len(tail)):
        if tail[index] == "{":
            depth += 1
        elif tail[index] == "}":
            depth -= 1
            if depth == 0:
                return tail[opening + 1 : index]
    return ""


def _choice(text: str) -> str | None:
    """Adapted from JetAstra/PI-Mem 410f0c10 LongBench v2 protocol."""
    text = str(text).replace("*", "").strip()
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    stripped = text.strip(".。,:;，：；").strip().strip("()[]{}").strip()
    if re.fullmatch("[A-D]", stripped):
        return stripped
    match = re.search(r"(?:^|[^A-Za-z])([A-D])(?:[^A-Za-z]|$)", text)
    return match.group(1) if match else None


def choice_match(prediction: str, answers: str | Iterable[str]) -> bool:
    choice = _choice(prediction)
    return choice is not None and any(
        choice == _choice(answer) for answer in _aliases(answers)
    )


def answer_reward(
    text: str, answers: str | Iterable[str], rewrites: int, chunk_count: int
) -> float:
    """Eq.16: binary boxed-answer correctness, shared by both RL stages.

    Rewrite/chunk counts are validated as trace metadata only; they do not
    affect the reward or introduce an efficiency bonus.
    """
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < 1
    ):
        raise ValueError("chunk_count must be positive")
    if (
        isinstance(rewrites, bool)
        or not isinstance(rewrites, int)
        or not 0 <= rewrites <= chunk_count - 1
    ):
        raise ValueError("rewrites must count post-bootstrap rewrites")
    correct = exact_match(extract_answer(text), answers)
    return float(correct)
