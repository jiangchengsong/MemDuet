"""Frozen paper prompts and validated assistant-first k/u token interface.

Token-prefix checks are adapted from MemAgent/gate_sft/prompting.py (see NOTICE).
No imports from that original workspace are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from collections.abc import Mapping
import re
from typing import Any


@dataclass(frozen=True)
class TokenProtocol:
    k_id: int
    u_id: int
    pad_token_id: int
    eos_token_id: int | None


def _template(name: str) -> str:
    raw = (
        files("memduet").joinpath("prompts", name + ".txt").read_text(encoding="utf-8")
    )
    return raw.removesuffix("\n").removesuffix("\r")


def _render(name: str, **values: str) -> str:
    for key, value in values.items():
        if not isinstance(value, str) or (key != "memory" and not value):
            raise ValueError(f"{key} must be a nonempty string (memory may be empty)")
    # One substitution pass protects literal boxed braces AND placeholder-like
    # strings inside user-provided content from recursive substitutions.
    return re.sub(
        r"\{(prompt|memory|chunk)\}", lambda match: values[match[1]], _template(name)
    )


def render_gate(question: str, memory: str, chunk: str) -> str:
    return _render("gate", prompt=question, memory=memory, chunk=chunk)


def render_memory(question: str, memory: str, chunk: str) -> str:
    return _render("memory_update", prompt=question, memory=memory, chunk=chunk)


def render_answer(question: str, memory: str) -> str:
    return _render("final_answer", prompt=question, memory=memory)


def _token_list(result: Any) -> list[int]:
    if isinstance(result, Mapping):
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError("expected exactly one tokenized conversation")
        result = result[0]
    ids = list(result)
    if not ids or any(
        isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids
    ):
        raise ValueError("tokenizer must return nonempty nonnegative integer token IDs")
    return ids


def apply_chat_template(
    tokenizer: Any,
    prompt: str,
    *,
    assistant_text: str | None = None,
    add_generation_prompt: bool = True,
) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    try:
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError as exc:
        raise ValueError(
            "chat template must support explicit enable_thinking=False"
        ) from exc
    return _token_list(result)


def validate_token_protocol(tokenizer: Any) -> TokenProtocol:
    ids = [
        _token_list(tokenizer.encode(label, add_special_tokens=False))
        for label in ("k", "u")
    ]
    if any(len(value) != 1 for value in ids):
        raise ValueError("k and u must each be exactly one token")
    k_id, u_id = ids[0][0], ids[1][0]
    if k_id == u_id:
        raise ValueError("k and u must have distinct token IDs")
    for label, token_id in [("k", k_id), ("u", u_id)]:
        if (
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            != label
        ):
            raise ValueError("k/u tokens must decode exactly to their labels")
    probe = render_gate("probe question", "probe memory", "probe section")
    prefix = apply_chat_template(tokenizer, probe)
    for label, token_id in [("k", k_id), ("u", u_id)]:
        full = apply_chat_template(
            tokenizer, probe, assistant_text=label, add_generation_prompt=False
        )
        if full[: len(prefix)] != prefix:
            raise ValueError(
                "assistant conversation does not preserve the generation prompt prefix"
            )
        if len(full) <= len(prefix) or full[len(prefix)] != token_id:
            raise ValueError("k/u must occur at the exact first assistant token")
    pad = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad, int) or pad < 0:
        raise ValueError("tokenizer must define pad_token_id")
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("tokenizer must use right padding")
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer must define a nonempty chat_template")
    return TokenProtocol(
        k_id,
        u_id,
        pad,
        getattr(tokenizer, "eos_token_id", None),
    )
