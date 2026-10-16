"""Inference entry points; importing the CLI never loads models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _configured(args):
    from .config import load_config, set_override, validate_config

    config = load_config(args.config)
    for expression in args.set:
        set_override(config, expression)
    validate_config(config)
    return config


def _parser():
    parser = argparse.ArgumentParser(description="MemDuet dual-process memory inference")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "export-inference", "serve", "evaluate"):
        child = commands.add_parser(name)
        child.add_argument("--config", default="configs/inference.json")
        child.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name == "validate":
            child.add_argument("--check-paths", action="store_true")
        elif name == "serve":
            child.add_argument("--role", required=True, choices=["system1", "system2"])
            child.add_argument("--host")
            child.add_argument("--port", type=int)
        elif name == "evaluate":
            child.add_argument("--input", required=True)
            child.add_argument("--output", required=True)
            child.add_argument("--metric", choices=["em", "sub_em", "choice"], default="sub_em")
            child.add_argument("--seed", type=int, help="override the configuration seed")
            child.add_argument("--method", choices=["memduet", "all_chunks", "full_context"])
            child.add_argument("--error-policy", choices=["strict", "tolerant"], default="strict")
            child.add_argument("--decision-delta", type=float)
            child.add_argument("--temperature", type=float, help="override rollout.temperature (default 0)")
            child.add_argument("--top-p", type=float, help="override rollout.top_p (default 1)")
            child.add_argument("--selection-mode", choices=["sample", "threshold"])
    summary = commands.add_parser("summarize", help="compare matched question cohorts")
    summary.add_argument("--results", required=True)
    summary.add_argument("--baseline")
    summary.add_argument("--expected-ids", help="JSON list of question IDs")
    summary.add_argument("--strict", action="store_true")
    summary.add_argument("--output")
    return parser


def _read_records(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        result = json.loads(text)
        return result["records"] if isinstance(result, dict) else result
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            from .config import validate_config

            _json(validate_config(_configured(args), check_paths=args.check_paths))
        elif args.command == "export-inference":
            from .serving import export_models

            _json(export_models(_configured(args)))
        elif args.command == "serve":
            from .serving import serve

            return serve(_configured(args), args.role, host=args.host, port=args.port)
        elif args.command == "evaluate":
            from .serving import evaluate_vllm

            config = _configured(args)
            rollout = config["rollout"]
            result = evaluate_vllm(
                config, input_path=args.input, output_path=args.output,
                seed=config["seed"] if args.seed is None else args.seed,
                method=args.method, metric=args.metric,
                temperature=rollout.get("temperature", 0) if args.temperature is None else args.temperature,
                top_p=rollout.get("top_p", 1) if args.top_p is None else args.top_p,
                selection_mode=args.selection_mode, decision_delta=args.decision_delta,
                error_policy=args.error_policy,
            )
            _json({"output": args.output, "records": len(result["records"]), "summary": result.get("summary")})
        elif args.command == "summarize":
            from .evaluation import aggregate

            expected = None if args.expected_ids is None else json.loads(Path(args.expected_ids).read_text())
            result = aggregate(
                _read_records(args.results),
                None if args.baseline is None else _read_records(args.baseline),
                expected_ids=expected, strict=args.strict,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            _json(result)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"memduet: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
