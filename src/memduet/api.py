"""Training-free Appendix B.5 transfer through explicitly configured APIs."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


class APIError(RuntimeError):
    """A failed request or invalid provider response; never substitutes an answer."""


@dataclass
class APICompletion:
    text: str
    usage: dict[str, int] | None = None


def _disabled_thinking_parameters(value):
    """Require an explicit wire control; provider/model names imply no setting."""
    if not isinstance(value, dict) or not (
        value == {"thinking": {"type": "disabled"}}
        or value == {"reasoning_effort": "none"}
        or (set(value) == {"enable_thinking"} and value["enable_thinking"] is False)
    ):
        raise ValueError(
            "thinking_parameters must explicitly disable thinking with "
            "thinking.type=disabled, reasoning_effort=none, or enable_thinking=false"
        )
    return json.loads(json.dumps(value))


def _usage(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise APIError("invalid provider usage object")
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    result = {}
    for name in fields:
        if name in value:
            count = value[name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise APIError("invalid provider usage token count")
            result[name] = count
    if not result:
        return None
    if (
        all(name in result for name in fields)
        and result["total_tokens"]
        != result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise APIError("inconsistent provider usage totals")
    return result


class OpenAICompatibleClient:
    """urllib chat-completions client with environment-only credentials.

    ``base_url`` is the provider API root (for example an explicitly supplied
    /v1 URL). No endpoint, model, credential, or price is assumed. Independent
    calls are thread-safe; attempt-log writes/snapshots are locked. Only
    transport/HTTP 429/5xx failures are retryable.
    """

    def __init__(
        self,
        base_url,
        model,
        api_key_env,
        *,
        thinking_parameters=None,
        timeout=60,
        max_retries=0,
        prices=None,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an explicit HTTP(S) API root without credentials/query/fragment"
            )
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(api_key_env, str)
            or not api_key_env
        ):
            raise ValueError("model and api_key_env must be explicitly provided")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("invalid timeout or max_retries")
        if prices is not None:
            if set(prices) != {"input_per_million", "output_per_million"} or any(
                isinstance(x, bool)
                or not isinstance(x, (float, int))
                or not math.isfinite(x)
                or x < 0
                for x in prices.values()
            ):
                raise ValueError(
                    "prices require known nonnegative input/output USD per million tokens"
                )
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.max_retries = timeout, max_retries
        self.prices = dict(prices) if prices is not None else None
        self.thinking_parameters = _disabled_thinking_parameters(thinking_parameters)
        self.version = f"openai-compatible:{model}"
        self.calls = []
        self._calls_lock = threading.Lock()

    def snapshot_calls(self):
        """Return a consistent attempt-log snapshot without exposing credentials."""
        with self._calls_lock:
            return list(self.calls)

    def provenance(self):
        # Environment variable names are configuration, never credential values.
        return {
            "client": "urllib-chat-completions",
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "prices": self.prices,
            "thinking_parameters": _disabled_thinking_parameters(
                self.thinking_parameters
            ),
        }

    def complete(
        self, prompt, *, max_tokens, temperature=0.0, top_p=1.0
    ) -> APICompletion:
        if (
            not isinstance(prompt, str)
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("prompt and positive max_tokens required")
        if (
            not math.isfinite(temperature)
            or temperature < 0
            or not math.isfinite(top_p)
            or not 0 < top_p <= 1
        ):
            raise ValueError("invalid generation sampling settings")
        key = os.environ.get(self.api_key_env)
        if not key:
            raise APIError(
                f"credential environment variable is unset: {self.api_key_env}"
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                **_disabled_thinking_parameters(self.thinking_parameters),
            }
        ).encode()
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            entry = {"attempt": attempt + 1, "model": self.model}
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                try:
                    body = json.loads(raw)
                    choices = body["choices"]
                    text = choices[0]["message"]["content"]
                    if not isinstance(choices, list) or not isinstance(text, str):
                        raise ValueError("invalid content")
                    usage = _usage(body.get("usage"))
                except (ValueError, KeyError, IndexError, TypeError):
                    raise APIError(
                        "invalid chat-completions response structure"
                    ) from None
                entry.update(status="ok", usage=usage)
                if (
                    self.prices is not None
                    and usage is not None
                    and all(
                        name in usage for name in ("prompt_tokens", "completion_tokens")
                    )
                ):
                    entry["cost_usd"] = (
                        usage["prompt_tokens"] * self.prices["input_per_million"]
                        + usage["completion_tokens"] * self.prices["output_per_million"]
                    ) / 1_000_000
                return APICompletion(text, usage)
            except urllib.error.HTTPError as error:
                # Provider bodies and exception strings can contain request secrets.
                entry.update(status="http_error", http_status=error.code)
                retry = error.code == 429 or 500 <= error.code <= 599
                if not retry or attempt == self.max_retries:
                    raise APIError(
                        f"HTTP request failed (status {error.code})"
                    ) from None
            except (urllib.error.URLError, TimeoutError, OSError):
                entry["status"] = "transport_error"
                if attempt == self.max_retries:
                    raise APIError("API transport request failed") from None
            except APIError:
                entry["status"] = "invalid_response"
                raise
            finally:
                entry["latency_seconds"] = time.perf_counter() - started
                with self._calls_lock:
                    self.calls.append(entry)
        raise APIError("API retries exhausted")


def _literal_gate(completion):
    if completion.text.strip() not in {"k", "u"}:
        raise APIError("gate response must be literal k or u")
    return completion.text.strip()


def run_api_trajectory(
    problem, system1, system2, rollout, *, seed=42, memory_tokenizer=None
) -> dict:
    """B.5 text gates, all-candidate verification, earliest accepted commitment.

    The record intentionally is not an RL Trajectory: it has no probabilities,
    token IDs, or fabricated behavior log-probabilities. Provider usage is optional.
    ``seed`` is recorded for provenance, not claimed as provider determinism.
    """
    from .evaluation import _prompt

    width = rollout.get("window_size", 4)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("window_size must be a positive integer")
    concurrency = rollout.get("api_concurrency", 1)
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency < 1
    ):
        raise ValueError("api_concurrency must be a positive integer")
    if memory_tokenizer is None or not callable(
        getattr(memory_tokenizer, "encode", None)
    ):
        raise ValueError("memory_tokenizer is required to enforce the memory budget")
    settings = {
        name: rollout.get(name, default)
        for name, default in (
            ("system1_output_tokens", 8192),
            ("system2_output_tokens", 32768),
            ("memory_tokens", 1024),
        )
    }
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if "answer_tokens" in rollout:
        raise ValueError(
            "API output caps use system2_output_tokens; remove answer_tokens"
        )
    calls, rounds, generations = [], [], []
    attempts = {"system1": [], "system2": []}
    temperature, top_p = rollout.get("temperature", 1.0), rollout.get("top_p", 1.0)
    started = time.perf_counter()

    def snapshot(client):
        return (
            client.snapshot_calls()
            if hasattr(client, "snapshot_calls")
            else list(getattr(client, "calls", []))
        )

    def complete_many(client, role, kind, specs):
        # Workers only make requests; ordered records and usage are accumulated
        # by this calling thread after the entire phase finishes.
        offset = len(snapshot(client))

        def request(spec):
            prompt, max_tokens, _ = spec
            response = client.complete(
                prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p
            )
            if not isinstance(response, APICompletion) or not isinstance(
                response.text, str
            ):
                raise APIError(
                    "client must return APICompletion with text and optional usage"
                )
            _usage(response.usage)
            if kind in {"selection", "verification"}:
                _literal_gate(response)
            return response

        try:
            if concurrency > 1 and len(specs) > 1:
                # map preserves input order; context shutdown drains all in-flight
                # work before any request exception is propagated.
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    responses = list(executor.map(request, specs))
            else:
                responses = [request(spec) for spec in specs]
        finally:
            # Slice once per phase, not once per racing request. This counts each
            # provider attempt exactly once, even if both roles use one client.
            attempts[role].extend(snapshot(client)[offset:])
        for response, (_, max_tokens, chunk_index) in zip(responses, specs):
            row = {
                "role": role,
                "kind": kind,
                "chunk_index": chunk_index,
                "usage": _usage(response.usage),
                "max_tokens": max_tokens,
            }
            calls.append(row)
            if kind in {"bootstrap", "memory", "answer"}:
                generations.append(dict(row, text=response.text))
        return responses

    def complete(client, role, kind, prompt, max_tokens, chunk_index=None):
        return complete_many(client, role, kind, [(prompt, max_tokens, chunk_index)])[0]

    def checked_memory(response):
        tokens = memory_tokenizer.encode(response.text, add_special_tokens=False)
        count = len(tokens)
        generations[-1]["memory_tokens"] = count
        if count > settings["memory_tokens"]:
            raise APIError(
                f"generated memory exceeds the memory budget "
                f"({count} > {settings['memory_tokens']}); no truncation policy is assumed"
            )
        return response.text

    bootstrap = complete(
        system2,
        "system2",
        "bootstrap",
        _prompt(
            "memory_update",
            prompt=problem.question,
            memory="No previous memory",
            chunk=problem.chunks[0],
        ),
        settings["system2_output_tokens"],
        0,
    )
    memory = checked_memory(bootstrap)
    start = 1
    while start < len(problem.chunks):
        before = memory
        stop = min(start + width, len(problem.chunks))
        selections, verifications = [], []
        indices = list(range(start, stop))
        specs = [
            (
                _prompt(
                    "gate",
                    prompt=problem.question,
                    memory=before,
                    chunk=problem.chunks[index],
                ),
                settings["system1_output_tokens"],
                index,
            )
            for index in indices
        ]
        responses = complete_many(system1, "system1", "selection", specs)
        for index, response in zip(indices, responses):
            selections.append({"chunk_index": index, "label": _literal_gate(response)})
        # Verify every candidate against the unchanged memory, including stale suffix.
        candidates = [
            event["chunk_index"] for event in selections if event["label"] == "u"
        ]
        specs = [
            (
                _prompt(
                    "gate",
                    prompt=problem.question,
                    memory=before,
                    chunk=problem.chunks[index],
                ),
                settings["system2_output_tokens"],
                index,
            )
            for index in candidates
        ]
        responses = complete_many(system2, "system2", "verification", specs)
        for index, response in zip(candidates, responses):
            verifications.append(
                {"chunk_index": index, "label": _literal_gate(response)}
            )
        accepted = [
            event["chunk_index"] for event in verifications if event["label"] == "u"
        ]
        committed = accepted[0] if accepted else None
        rounds.append(
            {
                "memory_before": before,
                "start_index": start,
                "selections": selections,
                "verifications": verifications,
                "committed_index": committed,
            }
        )
        if committed is not None:
            response = complete(
                system2,
                "system2",
                "memory",
                _prompt(
                    "memory_update",
                    prompt=problem.question,
                    memory=before,
                    chunk=problem.chunks[committed],
                ),
                settings["system2_output_tokens"],
                committed,
            )
            memory = checked_memory(response)
            start = committed + 1
        else:
            start = stop
    answer = complete(
        system2,
        "system2",
        "answer",
        _prompt("final_answer", prompt=problem.question, memory=memory),
        settings["system2_output_tokens"],
    )
    usage = {
        "system1_calls": sum(call["role"] == "system1" for call in calls),
        "system2_calls": sum(call["role"] == "system2" for call in calls),
        "verification_calls": sum(call["kind"] == "verification" for call in calls),
    }
    for role in ("system1", "system2"):
        selected = [call for call in calls if call["role"] == role]
        counts = [
            call["usage"].get("completion_tokens")
            if call["usage"] is not None
            else None
            for call in selected
        ]
        input_counts = [
            call["usage"].get("prompt_tokens") if call["usage"] is not None else None
            for call in selected
        ]
        usage[role + "_output_tokens"] = (
            sum(counts) if all(value is not None for value in counts) else None
        )
        usage[role + "_input_tokens"] = (
            sum(input_counts)
            if all(value is not None for value in input_counts)
            else None
        )

    def output_count(selected):
        counts = [
            call["usage"].get("completion_tokens")
            if call["usage"] is not None
            else None
            for call in selected
        ]
        return sum(counts) if all(value is not None for value in counts) else None

    usage["system2_verification_output_tokens"] = output_count(
        call for call in calls if call["kind"] == "verification"
    )
    usage["system2_generation_output_tokens"] = output_count(
        call for call in calls if call["kind"] in {"bootstrap", "memory", "answer"}
    )
    usage["total_output_tokens"] = (
        usage["system1_output_tokens"] + usage["system2_output_tokens"]
        if all(
            usage[role + "_output_tokens"] is not None
            for role in ("system1", "system2")
        )
        else None
    )
    all_attempts = [attempt for role in attempts.values() for attempt in role]
    attempt_usage_complete = bool(all_attempts) and all(
        attempt.get("usage") is not None for attempt in all_attempts
    )
    costs = [attempt.get("cost_usd") for attempt in all_attempts]
    return {
        "problem_id": problem.id,
        "answers": problem.answers,
        "chunk_count": len(problem.chunks),
        "rounds": rounds,
        "generations": generations,
        "final_answer": answer.text,
        "usage": usage,
        "calls": calls,
        "provider_attempts": attempts,
        "provider_attempt_usage_complete": attempt_usage_complete,
        "cost_usd": sum(costs)
        if costs and all(cost is not None for cost in costs)
        else None,
        "latency_seconds": time.perf_counter() - started,
        "seed": seed,
        "request_settings": dict(settings, memory_overflow_policy="error"),
        "protocol": "Appendix B.5 training-free literal-text k/u; provider seed determinism not assumed",
        "system1_version": system1.version,
        "system2_version": system2.version,
    }
