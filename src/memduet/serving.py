"""Checkpoint export, vLLM serving, and evaluation entry points."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def export_models(config):
    """Merge the selected adapters into standalone serving checkpoints."""
    from .backend import HFBackend

    outputs = {}
    for role in ("system1", "system2"):
        destination = Path(config["inference"][role]["model_path"])
        if destination.exists():
            raise ValueError(f"export destination already exists: {destination}")
        spec = dict(config[role], device="cpu", device_map=None)
        backend = HFBackend.from_config(spec, trainable=False)
        model = backend.model
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload(safe_merge=True)
        destination.mkdir(parents=True)
        model.save_pretrained(destination, safe_serialization=True)
        backend.tokenizer.save_pretrained(destination)
        outputs[role] = str(destination)
        del model, backend
        gc.collect()
    return outputs


def serve(config, role, *, host=None, port=None):
    settings = config["inference"]
    spec = settings[role]
    endpoint = urlsplit(spec["base_url"])
    if not Path(spec["model_path"]).is_dir():
        raise ValueError(
            "serving model is missing; run export-inference or set inference model_path"
        )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec["model_path"],
        "--served-model-name",
        spec["served_model_name"],
        "--host",
        host or endpoint.hostname or "127.0.0.1",
        "--port",
        str(port or endpoint.port or 8000),
        "--dtype",
        "bfloat16",
        "--seed",
        str(config.get("seed", 42)),
        "--tensor-parallel-size",
        str(settings["tensor_parallel_size"]),
        "--max-model-len",
        str(spec["max_model_len"]),
        "--max-num-seqs",
        str(settings["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(spec["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    if settings.get("enforce_eager", False):
        command.append("--enforce-eager")
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", str(settings["gpu_device"]))
    return subprocess.call(command, env=environment)


def load_inference_problems(path, tokenizer, *, chunk_tokens=5000):
    """Read canonical JSONL or benchmark JSON; preserve all context chunks."""
    from .types import Problem

    if (
        isinstance(chunk_tokens, bool)
        or not isinstance(chunk_tokens, int)
        or chunk_tokens < 1
    ):
        raise ValueError("chunk_tokens must be a positive integer")
    text = Path(path).read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("records", [rows])
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("input must contain a nonempty list of records")
    result, seen = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("each input record must be an object")
        question = row.get("question", row.get("input"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        identity = str(row.get("id", row.get("_id", index)))
        if identity in seen:
            raise ValueError(f"duplicate id: {identity}")
        seen.add(identity)
        context = row.get("context")
        if context is not None and not isinstance(context, str):
            raise ValueError(f"context must be text for {identity}")
        metadata = dict(row.get("metadata", {}))
        if "chunks" in row:
            chunks = row["chunks"]
            if (
                not isinstance(chunks, list)
                or not chunks
                or any(not isinstance(c, str) or not c for c in chunks)
            ):
                raise ValueError(f"chunks must be a nonempty text list for {identity}")
        else:
            if not isinstance(context, str) or not context.strip():
                raise ValueError(f"missing context for {identity}")
            ids = tokenizer.encode(context.strip())
            chunks = [
                tokenizer.decode(ids[i : i + chunk_tokens])
                for i in range(0, len(ids), chunk_tokens)
            ]
            metadata.setdefault("source_context_tokens", len(ids))
        answers = row.get("answers", row.get("answer"))
        if isinstance(answers, str):
            answers = [answers]
        if (
            not isinstance(answers, list)
            or not answers
            or any(not isinstance(a, str) or not a.strip() for a in answers)
        ):
            raise ValueError(f"missing answers for {identity}")
        result.append(
            Problem(
                identity,
                question,
                chunks,
                answers,
                metadata,
                context,
            )
        )
    return result


class ServedModel:
    def __init__(self, settings):
        self.settings = dict(settings)

    def provenance(self):
        return {"engine": "vllm", **self.settings}


def evaluate_vllm(
    config,
    *,
    input_path,
    output_path,
    seed=42,
    method=None,
    metric="sub_em",
    temperature=0.0,
    top_p=1.0,
    selection_mode=None,
    decision_delta=None,
    error_policy="strict",
):
    import aiohttp
    from transformers import AutoTokenizer
    from .evaluation import evaluate, _prompt
    from .inference import UsageTotals, run_inference
    from .vllm_client import (
        OpenAICompatibleChatClient,
        KUProbabilityOpenAICompatibleChatClient,
        KUProbabilityOpenAICompatibleCompletionClient,
        resolve_ku_token_ids,
    )

    settings = config.get("inference", {})
    method = method or settings.get("method", "memduet")
    if settings.get("engine") != "vllm":
        raise ValueError("local evaluation requires inference.engine=vllm")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be nonnegative and top_p in (0,1]")
    roles = ("system1", "system2") if method == "memduet" else ("system2",)
    tokenizers = {
        role: AutoTokenizer.from_pretrained(
            settings[role]["model_path"], trust_remote_code=False
        )
        for role in roles
    }
    problems = load_inference_problems(
        input_path, tokenizers["system2"], chunk_tokens=config["data"]["chunk_tokens"]
    )
    rollout = dict(
        config["rollout"], temperature=temperature, top_p=top_p, engine="vllm"
    )
    rollout["selection_mode"] = selection_mode or rollout.get(
        "selection_mode", "sample"
    )
    if decision_delta is not None:
        rollout["decision_delta"] = decision_delta
    loop = asyncio.new_event_loop()

    async def make_clients():
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings["timeout_seconds"])
        )
        try:
            common = {}
            for role in roles:
                spec = settings[role]
                common[role] = dict(
                    session=session,
                    semaphore=asyncio.Semaphore(max(rollout["window_size"], 1)),
                    model=spec["served_model_name"],
                    base_url=spec["base_url"],
                    api_key=os.environ.get(settings["api_key_env"], "EMPTY"),
                    temperature=0.0,
                    top_p=1.0,
                )
                async with session.get(
                    spec["base_url"].rstrip("/") + "/models",
                    headers={"Authorization": "Bearer " + common[role]["api_key"]},
                ) as response:
                    response.raise_for_status()
                    available = {m["id"] for m in (await response.json())["data"]}
                    if spec["served_model_name"] not in available:
                        raise ValueError(f"served model unavailable for {role}")
            writer = OpenAICompatibleChatClient(
                **dict(common["system2"], temperature=temperature, top_p=top_p),
                seed=seed,
            )
            gate = verifier = None
            if method == "memduet":
                k, u = resolve_ku_token_ids(tokenizers["system1"])
                gate = KUProbabilityOpenAICompatibleChatClient(
                    **common["system1"],
                    keep_token_id=k,
                    update_token_id=u,
                    allowed_purposes=["streaming_barrier_control"],
                    decision_seed=seed,
                )
                k, u = resolve_ku_token_ids(tokenizers["system2"])
                verifier = KUProbabilityOpenAICompatibleCompletionClient(
                    **common["system2"],
                    tokenizer=tokenizers["system2"],
                    keep_token_id=k,
                    update_token_id=u,
                    allowed_purposes=["target_update_verification"],
                    decision_seed=seed,
                )
            return session, writer, gate, verifier
        except BaseException:
            await session.close()
            raise

    async def trajectory(problem, rollout, *, seed, method):
        writer.seed = seed
        if gate is not None:
            gate.decision_seed = verifier.decision_seed = seed
        if method == "full_context":
            response = await writer.complete(
                _prompt(
                    "final_answer",
                    prompt=problem.question,
                    memory=problem.source_context
                    if problem.source_context is not None
                    else "".join(problem.chunks),
                ),
                rollout["answer_tokens"],
                "final_answer",
            )
            usage = UsageTotals()
            usage.add(response.usage)
            return {
                "final_answer": response.text,
                "usage": {
                    "system1_output_tokens": 0,
                    "system2_output_tokens": usage.completion_tokens,
                    "total_output_tokens": usage.completion_tokens,
                },
                "usage_responses": {"system2": usage.to_json()},
            }
        return await run_inference(
            writer,
            gate,
            verifier,
            problem.question,
            problem.chunks,
            seed=seed,
            method=method,
            selection_mode=rollout["selection_mode"],
            decision_delta=rollout["decision_delta"],
            lookahead=rollout["window_size"],
            memory_tokens=rollout["memory_tokens"],
            answer_tokens=rollout["answer_tokens"],
        )

    session = None
    try:
        session, writer, gate, verifier = loop.run_until_complete(make_clients())

        def runner(*args, **kwargs):
            return loop.run_until_complete(trajectory(*args, **kwargs))

        return evaluate(
            problems,
            ServedModel(settings["system1"] if method == "memduet" else {}),
            ServedModel(settings["system2"]),
            rollout,
            output_path=output_path,
            seed=seed,
            method=method,
            metric=metric,
            trajectory_runner=runner,
            error_policy=error_policy,
        )
    finally:
        if session is not None:
            loop.run_until_complete(session.close())
        loop.close()
