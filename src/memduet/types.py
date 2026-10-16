"""Versioned, JSON-safe rollout records shared by training and inference."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class Problem:
    id: str
    question: str
    chunks: list[str]
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_context: str | None = None

    def __post_init__(self):
        if (
            not self.id
            or not self.question
            or not self.chunks
            or any(not isinstance(c, str) or not c for c in self.chunks)
        ):
            raise ValueError("Problem needs an id, question, and nonempty text chunks")
        if not all(isinstance(a, str) for a in self.answers):
            raise ValueError("answers must be strings")


@dataclass
class GateScore:
    prompt: str
    prompt_ids: list[int]
    update_probability: float
    output_tokens: int = 1


@dataclass
class SelectionEvent:
    chunk_index: int
    prompt: str
    prompt_ids: list[int]
    update_probability: float
    action: bool


@dataclass
class VerificationEvent:
    chunk_index: int
    prompt: str
    prompt_ids: list[int]
    update_probability: float
    acceptance_probability: float | None = None
    uniform_sample: float | None = None
    accepted: bool | None = None


@dataclass
class RoundEvent:
    memory_before: str
    start_index: int
    selections: list[SelectionEvent]
    verifications: list[VerificationEvent]
    committed_index: int | None
    old_log_probability: float


@dataclass
class GenerationEvent:
    kind: str
    prompt: str
    prompt_ids: list[int]
    token_ids: list[int]
    old_log_probs: list[float]
    text: str

    def __post_init__(self):
        if self.kind not in {"bootstrap", "memory", "answer"}:
            raise ValueError("unknown generation event kind")
        if len(self.token_ids) != len(self.old_log_probs):
            raise ValueError("every generated token needs its behavior log probability")


@dataclass
class TokenStatistics:
    log_probs: Any
    kl: Any


@dataclass
class Trajectory:
    problem_id: str
    answers: list[str]
    chunk_count: int
    rounds: list[RoundEvent]
    generations: list[GenerationEvent]
    final_answer: str
    reward: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    latency_seconds: float = 0.0
    system1_version: str = "unversioned"
    system2_version: str = "unversioned"
    schema_version: int = SCHEMA_VERSION

    @property
    def rewrite_count(self):
        return sum(g.kind == "memory" for g in self.generations)

    def event_count(self, stage: int):
        if stage not in {2, 3}:
            raise ValueError("RL stage must be 2 or 3")
        return (
            sum(len(g.token_ids) for g in self.generations)
            + sum(bool(r.verifications) for r in self.rounds)
            + (sum(len(r.selections) for r in self.rounds) if stage == 3 else 0)
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported trajectory schema")
        data["rounds"] = [
            RoundEvent(
                **dict(
                    r,
                    selections=[SelectionEvent(**s) for s in r["selections"]],
                    verifications=[VerificationEvent(**v) for v in r["verifications"]],
                )
            )
            for r in data["rounds"]
        ]
        data["generations"] = [GenerationEvent(**g) for g in data["generations"]]
        return cls(**data)
