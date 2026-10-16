# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# Adapted for MemDuet inference and sampling.
"""vLLM generation and exact k/u probability clients."""

from __future__ import annotations
import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional
import aiohttp
from .inference import ChatCompletionResult, ChatCompletionError, KUProbabilityResult

URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
KEEP_GATE_TOKEN = "k"
UPDATE_GATE_TOKEN = "u"
TARGET_VERIFICATION_MAX_NEW = 1
DEFAULT_ACCEPTANCE_SEED = 42


def _short_body(body: str, limit: int = 500) -> str:
    body = body.replace("\n", " ").strip()
    return body if len(body) <= limit else body[:limit] + "...(truncated)"


async def post_chat_completion(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    base_url: str = URL,
    api_key: str = API_KEY,
    max_retries: int = 3,
    seed: int = 42,
) -> ChatCompletionResult:
    """Call the OpenAI-compatible endpoint with bounded conservative retries."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {api_key}", "api-key": api_key}
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        retryable = False
        try:
            async with semaphore:
                async with session.post(
                    url=base_url.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    status = response.status
                    body = await response.text()
            if status == 200:
                try:
                    data = json.loads(body)
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                    raise ChatCompletionError(
                        f"HTTP 200 returned an invalid chat-completion body: {_short_body(body)}"
                    ) from exc
                if not isinstance(content, str) or not content.strip():
                    raise ChatCompletionError(
                        "HTTP 200 returned empty completion content"
                    )
                usage = data.get("usage")
                return ChatCompletionResult(
                    content, usage if isinstance(usage, Mapping) else None
                )
            error = ChatCompletionError(
                f"chat completion failed with HTTP status {status}: {_short_body(body)}"
            )
            if status == 429 or 500 <= status <= 599:
                retryable = True
                last_error = error
            else:
                raise error
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            retryable = True
            last_error = exc
        if not retryable:
            raise AssertionError("unreachable non-retryable request state")
        if attempt >= max_retries:
            if isinstance(last_error, ChatCompletionError):
                raise last_error
            raise ChatCompletionError(
                f"chat completion failed after {max_retries} retries: {last_error}"
            ) from last_error
        await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable retry loop exit")


class OpenAICompatibleChatClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        model: str,
        temperature: float,
        top_p: float,
        base_url: str = URL,
        api_key: str = API_KEY,
        seed: int = 42,
    ) -> None:
        self.session = session
        self.semaphore = semaphore
        self.max_stream_concurrency = int(semaphore._value)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url
        self.api_key = api_key
        self.seed = seed

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionResult:
        del purpose
        return await post_chat_completion(
            session=self.session,
            semaphore=self.semaphore,
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p,
            base_url=self.base_url,
            api_key=self.api_key,
            seed=self.seed,
        )


def _short_response(body: str, limit: int = 500) -> str:
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _probabilities_from_chat_response(
    data: Mapping[str, Any], decision: str
) -> tuple[float, float, float, float]:
    """Read and renormalize the response's first-position k/u logprobs."""
    try:
        token_logprobs = data["choices"][0]["logprobs"]["content"][0]
        top_logprobs = token_logprobs["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionError(
            "k/u probability response omitted first-token top_logprobs"
        ) from exc
    if not isinstance(top_logprobs, list):
        raise ChatCompletionError("k/u top_logprobs must be a list")
    by_token: dict[str, float] = {}
    candidates = [token_logprobs, *top_logprobs]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        token = candidate.get("token")
        logprob = candidate.get("logprob")
        if token in (KEEP_GATE_TOKEN, UPDATE_GATE_TOKEN) and isinstance(
            logprob, (int, float)
        ):
            by_token[str(token)] = float(logprob)
    missing = {
        token for token in (KEEP_GATE_TOKEN, UPDATE_GATE_TOKEN) if token not in by_token
    }
    if missing:
        raise ChatCompletionError(
            "k/u probability response is missing restricted labels: "
            + ",".join(sorted(missing))
        )
    if decision not in (KEEP_GATE_TOKEN, UPDATE_GATE_TOKEN):
        raise ChatCompletionError(f"invalid restricted k/u decision: {decision!r}")
    keep_logprob = by_token[KEEP_GATE_TOKEN]
    update_logprob = by_token[UPDATE_GATE_TOKEN]
    if not math.isfinite(keep_logprob) or not math.isfinite(update_logprob):
        raise ChatCompletionError("k/u logprobs must both be finite")
    maximum = max(keep_logprob, update_logprob)
    keep_weight = math.exp(keep_logprob - maximum)
    update_weight = math.exp(update_logprob - maximum)
    normalizer = keep_weight + update_weight
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ChatCompletionError("k/u probabilities have an invalid normalizer")
    return (
        keep_weight / normalizer,
        update_weight / normalizer,
        keep_logprob,
        update_logprob,
    )


def _completion_token_id(
    token: Any, keep_token_id: int, update_token_id: int
) -> Optional[int]:
    if token == KEEP_GATE_TOKEN:
        return keep_token_id
    if token == UPDATE_GATE_TOKEN:
        return update_token_id
    if isinstance(token, str) and token.startswith("token_id:"):
        try:
            token_id = int(token.split(":", 1)[1])
        except ValueError:
            return None
        if token_id in (keep_token_id, update_token_id):
            return token_id
    return None


def _probabilities_from_completion_response(
    data: Mapping[str, Any], keep_token_id: int, update_token_id: int
) -> tuple[str, float, float, float, float]:
    """Read a vLLM Completion response using stable token-id logprob keys."""
    try:
        logprobs = data["choices"][0]["logprobs"]
        tokens = logprobs["tokens"]
        token_logprobs = logprobs["token_logprobs"]
        top_logprobs = logprobs["top_logprobs"]
        generated_token = tokens[0]
        generated_logprob = token_logprobs[0]
        first_top_logprobs = top_logprobs[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ChatCompletionError(
            "k/u Completion response omitted first-token logprobs"
        ) from exc
    generated_token_id = _completion_token_id(
        generated_token, keep_token_id, update_token_id
    )
    if generated_token_id is None:
        raise ChatCompletionError(
            f"k/u Completion response returned invalid token {generated_token!r}"
        )
    if not isinstance(first_top_logprobs, Mapping):
        raise ChatCompletionError("k/u Completion top_logprobs must be a mapping")
    by_token_id: dict[int, float] = {}
    for token, logprob in first_top_logprobs.items():
        token_id = _completion_token_id(token, keep_token_id, update_token_id)
        if token_id is not None and isinstance(logprob, (int, float)):
            by_token_id[token_id] = float(logprob)
    if isinstance(generated_logprob, (int, float)):
        by_token_id[generated_token_id] = float(generated_logprob)
    missing = {
        token_id
        for token_id in (keep_token_id, update_token_id)
        if token_id not in by_token_id
    }
    if missing:
        raise ChatCompletionError(
            "k/u Completion response is missing restricted token IDs: "
            + ",".join((str(token_id) for token_id in sorted(missing)))
        )
    keep_logprob = by_token_id[keep_token_id]
    update_logprob = by_token_id[update_token_id]
    if not math.isfinite(keep_logprob) or not math.isfinite(update_logprob):
        raise ChatCompletionError("k/u Completion logprobs must both be finite")
    maximum = max(keep_logprob, update_logprob)
    keep_weight = math.exp(keep_logprob - maximum)
    update_weight = math.exp(update_logprob - maximum)
    normalizer = keep_weight + update_weight
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ChatCompletionError(
            "k/u Completion probabilities have an invalid normalizer"
        )
    decision = (
        KEEP_GATE_TOKEN if generated_token_id == keep_token_id else UPDATE_GATE_TOKEN
    )
    return (
        decision,
        keep_weight / normalizer,
        update_weight / normalizer,
        keep_logprob,
        update_logprob,
    )


def _render_chat_prompt_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    """Apply the local system2 chat template for the Completion endpoint."""
    renderer = getattr(tokenizer, "apply_chat_template", None)
    if not callable(renderer):
        raise ValueError("large-model tokenizer must provide apply_chat_template")
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = renderer(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = renderer(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if callable(getattr(rendered, "tolist", None)):
        rendered = rendered.tolist()
    if (
        isinstance(rendered, Sequence)
        and len(rendered) == 1
        and isinstance(rendered[0], Sequence)
        and (not isinstance(rendered[0], (str, bytes)))
    ):
        rendered = rendered[0]
    if (
        not isinstance(rendered, Sequence)
        or isinstance(rendered, (str, bytes))
        or (not rendered)
        or any((not isinstance(token_id, int) for token_id in rendered))
    ):
        raise ValueError(
            "large-model chat template did not return a non-empty token-id list"
        )
    return [int(token_id) for token_id in rendered]


class KUProbabilityOpenAICompatibleChatClient(OpenAICompatibleChatClient):
    """One-token vLLM client returning a restricted k/u distribution."""

    def __init__(
        self,
        *args: Any,
        keep_token_id: int,
        update_token_id: int,
        allowed_purposes: Sequence[str],
        decision_seed: int = DEFAULT_ACCEPTANCE_SEED,
        decision_delta: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if keep_token_id == update_token_id:
            raise ValueError("k/u token IDs must be distinct")
        self.keep_token_id = int(keep_token_id)
        self.update_token_id = int(update_token_id)
        self.allowed_purposes = frozenset(allowed_purposes)
        self.decision_seed = int(decision_seed)
        if decision_delta is not None and (not math.isfinite(decision_delta)):
            raise ValueError("Gate decision delta must be finite")
        self.decision_delta = None if decision_delta is None else float(decision_delta)

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> KUProbabilityResult:
        if purpose not in self.allowed_purposes:
            raise ChatCompletionError(f"unsupported k/u probability purpose: {purpose}")
        if max_tokens != TARGET_VERIFICATION_MAX_NEW:
            raise ChatCompletionError(
                f"k/u probability requests require max_tokens={TARGET_VERIFICATION_MAX_NEW}, got {max_tokens}"
            )
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        if temperature < 0 or not 0 < top_p <= 1:
            raise ChatCompletionError(
                "k/u probability requests require temperature >= 0 and 0 < top_p <= 1"
            )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": TARGET_VERIFICATION_MAX_NEW,
            "allowed_token_ids": [self.keep_token_id, self.update_token_id],
            "logprob_token_ids": [self.keep_token_id, self.update_token_id],
            "logprobs": True,
            "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
            "seed": self.decision_seed,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "api-key": self.api_key}
        last_error: Optional[BaseException] = None
        for attempt in range(4):
            retryable = False
            try:
                async with self.semaphore:
                    async with self.session.post(
                        self.base_url.rstrip("/") + "/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        status = response.status
                        body = await response.text()
                if status == 200:
                    try:
                        data = json.loads(body)
                        content = data["choices"][0]["message"]["content"]
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ChatCompletionError(
                            "invalid k/u probability response: " + _short_response(body)
                        ) from exc
                    if not isinstance(content, str) or not content:
                        raise ChatCompletionError(
                            "k/u probability response returned empty content"
                        )
                    (keep_p, update_p, keep_lp, update_lp) = (
                        _probabilities_from_chat_response(data, content)
                    )
                    decision = content
                    if self.decision_delta is not None:
                        decision = (
                            UPDATE_GATE_TOKEN
                            if update_lp - keep_lp >= self.decision_delta
                            else KEEP_GATE_TOKEN
                        )
                    usage = data.get("usage")
                    return KUProbabilityResult(
                        text=decision,
                        usage=usage if isinstance(usage, Mapping) else None,
                        keep_probability=keep_p,
                        update_probability=update_p,
                        keep_logprob=keep_lp,
                        update_logprob=update_lp,
                    )
                error = ChatCompletionError(
                    f"k/u probability request failed with HTTP {status}: {_short_response(body)}"
                )
                if status == 429 or 500 <= status <= 599:
                    retryable = True
                    last_error = error
                else:
                    raise error
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                retryable = True
                last_error = exc
            if not retryable:
                raise AssertionError("unreachable non-retryable k/u request")
            if attempt == 3:
                if isinstance(last_error, ChatCompletionError):
                    raise last_error
                raise ChatCompletionError(
                    f"k/u probability request failed after 3 retries: {last_error}"
                ) from last_error
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable k/u probability retry loop exit")

    async def open_control_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise ChatCompletionError("k/u decisions use one-token completions")


class KUProbabilityOpenAICompatibleCompletionClient(OpenAICompatibleChatClient):
    """Render the verifier prompt locally and request exact label probabilities."""

    def __init__(
        self,
        *args: Any,
        tokenizer: Any,
        keep_token_id: int,
        update_token_id: int,
        allowed_purposes: Sequence[str],
        decision_seed: int = DEFAULT_ACCEPTANCE_SEED,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if keep_token_id == update_token_id:
            raise ValueError("k/u token IDs must be distinct")
        self.tokenizer = tokenizer
        self.keep_token_id = int(keep_token_id)
        self.update_token_id = int(update_token_id)
        self.allowed_purposes = frozenset(allowed_purposes)
        self.decision_seed = int(decision_seed)

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        purpose: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> KUProbabilityResult:
        if purpose not in self.allowed_purposes:
            raise ChatCompletionError(f"unsupported system2 k/u purpose: {purpose}")
        if max_tokens != TARGET_VERIFICATION_MAX_NEW:
            raise ChatCompletionError(
                f"system2 k/u requests require max_tokens={TARGET_VERIFICATION_MAX_NEW}, got {max_tokens}"
            )
        if temperature not in (None, 0, 0.0) or top_p not in (None, 1, 1.0):
            raise ChatCompletionError(
                "system2 k/u requests require temperature=0 and top_p=1"
            )
        prompt_token_ids = _render_chat_prompt_token_ids(self.tokenizer, prompt)
        payload = {
            "model": self.model,
            "prompt": prompt_token_ids,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": TARGET_VERIFICATION_MAX_NEW,
            "allowed_token_ids": [self.keep_token_id, self.update_token_id],
            "logprob_token_ids": [self.keep_token_id, self.update_token_id],
            "logprobs": 2,
            "add_special_tokens": False,
            "return_tokens_as_token_ids": True,
            "seed": self.decision_seed,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "api-key": self.api_key}
        last_error: Optional[BaseException] = None
        for attempt in range(4):
            retryable = False
            try:
                async with self.semaphore:
                    async with self.session.post(
                        self.base_url.rstrip("/") + "/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        status = response.status
                        body = await response.text()
                if status == 200:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise ChatCompletionError(
                            "invalid system2 k/u Completion response: "
                            + _short_response(body)
                        ) from exc
                    if not isinstance(data, Mapping):
                        raise ChatCompletionError(
                            "system2 k/u Completion response must be an object"
                        )
                    (decision, keep_p, update_p, keep_lp, update_lp) = (
                        _probabilities_from_completion_response(
                            data, self.keep_token_id, self.update_token_id
                        )
                    )
                    usage = data.get("usage")
                    return KUProbabilityResult(
                        text=decision,
                        usage=usage if isinstance(usage, Mapping) else None,
                        keep_probability=keep_p,
                        update_probability=update_p,
                        keep_logprob=keep_lp,
                        update_logprob=update_lp,
                    )
                error = ChatCompletionError(
                    f"system2 k/u Completion request failed with HTTP {status}: {_short_response(body)}"
                )
                if status == 429 or 500 <= status <= 599:
                    retryable = True
                    last_error = error
                else:
                    raise error
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                retryable = True
                last_error = exc
            if not retryable:
                raise AssertionError("unreachable non-retryable system2 k/u request")
            if attempt == 3:
                if isinstance(last_error, ChatCompletionError):
                    raise last_error
                raise ChatCompletionError(
                    f"system2 k/u Completion request failed after 3 retries: {last_error}"
                ) from last_error
            await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable system2 k/u retry loop exit")

    async def open_control_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise ChatCompletionError("system2 verification is non-streaming")


def resolve_ku_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve exact single-token k/u IDs for the served large-model tokenizer."""

    def encode(label: str) -> list[int]:
        try:
            encoded = tokenizer.encode(label, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(label)
        return [int(token_id) for token_id in encoded]

    keep_ids = encode(KEEP_GATE_TOKEN)
    update_ids = encode(UPDATE_GATE_TOKEN)
    if len(keep_ids) != 1 or len(update_ids) != 1:
        raise ValueError(
            f"large-model tokenizer must encode k/u as one token each: k={keep_ids}, u={update_ids}"
        )
    if keep_ids[0] == update_ids[0]:
        raise ValueError("large-model tokenizer maps k/u to the same token ID")
    if callable(getattr(tokenizer, "decode", None)):
        for label, token_id in (
            (KEEP_GATE_TOKEN, keep_ids[0]),
            (UPDATE_GATE_TOKEN, update_ids[0]),
        ):
            try:
                decoded = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded = tokenizer.decode([token_id])
            if decoded != label:
                raise ValueError(
                    f"large-model token {token_id} decodes to {decoded!r}, expected {label!r}"
                )
    return (keep_ids[0], update_ids[0])
