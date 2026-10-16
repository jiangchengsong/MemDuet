"""Portable JSON configuration, inheritance, and preflight checks (no model I/O)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError("configuration inheritance cycle")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    parent = config.pop("extends", None)
    if parent is not None:
        config = _merge(load_config(path.parent / parent, _seen + (path,)), config)
    return config


def set_override(config: dict, expression: str) -> None:
    """Set existing dotted keys only; JSON values preserve booleans/numbers/null."""
    key, separator, value = expression.partition("=")
    if not separator:
        raise ValueError("override must have dotted.key=value form")
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(f"unknown configuration key: {key}")
        node = node[part]
    if parts[-1] not in node:
        raise ValueError(f"unknown configuration key: {key}")
    try:
        node[parts[-1]] = json.loads(value)
    except json.JSONDecodeError:
        node[parts[-1]] = value


def _positive(value, name, *, integer=False, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite numeric")
    if (value < 0 if zero else value <= 0) or (integer and not isinstance(value, int)):
        raise ValueError(f"invalid {name}: {value}")


def validate_config(config: dict, *, check_paths: bool = False) -> dict:
    """Check inference settings without loading models or making requests."""
    from urllib.parse import urlsplit

    settings = config.get("inference", {})
    if settings.get("engine") != "vllm":
        raise ValueError("inference.engine must be vllm")
    if settings.get("method") not in {"memduet", "all_chunks", "full_context"}:
        raise ValueError("invalid inference.method")
    _positive(config.get("seed"), "seed", integer=True, zero=True)
    _positive(config.get("data", {}).get("chunk_tokens"), "data.chunk_tokens", integer=True)
    rollout = config.get("rollout", {})
    for key in ("window_size", "memory_tokens", "answer_tokens"):
        _positive(rollout.get(key), f"rollout.{key}", integer=True)
    if rollout.get("selection_mode") not in {"sample", "threshold"}:
        raise ValueError("selection_mode must be sample or threshold")
    delta = rollout.get("decision_delta")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta):
        raise ValueError("decision_delta must be finite numeric")
    for key in ("tensor_parallel_size", "max_num_seqs"):
        _positive(settings.get(key), f"inference.{key}", integer=True)
    _positive(settings.get("timeout_seconds"), "inference.timeout_seconds")
    roles = ("system1", "system2") if settings["method"] == "memduet" else ("system2",)
    for role in roles:
        spec = settings.get(role, {})
        for key in ("model_path", "served_model_name"):
            if not isinstance(spec.get(key), str) or not spec[key]:
                raise ValueError(f"inference.{role}.{key} is required")
        endpoint = urlsplit(spec.get("base_url", ""))
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError(f"invalid inference.{role}.base_url")
        _positive(spec.get("max_model_len"), f"inference.{role}.max_model_len", integer=True)
        budget = spec.get("gpu_memory_utilization")
        _positive(budget, f"inference.{role}.gpu_memory_utilization")
        if budget >= 1:
            raise ValueError("gpu_memory_utilization must be less than 1")
        if check_paths and not Path(spec["model_path"]).is_dir():
            raise ValueError(f"missing {role} serving model: {spec['model_path']}")
    return {"valid": True, "paths_checked": check_paths, "models_loaded": False}
