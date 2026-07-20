"""Package raw source files into a FlashInfer solution JSON.

Usage:
  python scripts/src_to_solution.py path/to/src_dir path/to/solution.json \
      --definition <task-definition-name> [--language cuda] [--dep cutlass] ...

The inverse of solution_to_src.py: a user authors a kernel as plain source
files, then mints the backend-native Solution that task/baseline/ (or
task/target/) expects — including the BuildSpec that, for a baseline, becomes
the run's frozen build spec. No JSON authored by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default host entry file by language (the kernel's symbol is `run`).
ENTRY_FILE_BY_LANGUAGE = {"cuda": "main.cpp", "cpp": "main.cpp"}
ENTRY_SYMBOL = "run"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package raw source files into a FlashInfer solution JSON."
    )
    parser.add_argument("source_dir", type=Path, help="Directory of source files to package.")
    parser.add_argument("output_json", type=Path, help="Path to write the solution JSON to.")
    parser.add_argument("--definition", required=True, help="Task definition name the solution targets.")
    parser.add_argument("--language", default="cuda", help="Source language (default: cuda).")
    parser.add_argument(
        "--entry",
        help="Entry point 'file::symbol' (default: main.cpp::run for cuda/cpp, else main.py::run).",
    )
    parser.add_argument(
        "--dep", action="append", default=[], dest="deps", metavar="NAME",
        help="A build dependency; repeatable.",
    )
    parser.add_argument(
        "--dps", action="store_true",
        help="Destination-passing style: the kernel writes outputs into out-parameters "
        "instead of returning them (default: value-returning).",
    )
    parser.add_argument("--binding", default="torch", choices=["torch", "tvm-ffi"], help="Binding (default: torch).")
    parser.add_argument("--author", default="user", help="Solution author (default: user).")
    parser.add_argument("--name", help="Solution name (default: '<definition>_baseline').")
    parser.add_argument(
        "--hardware", action="append", default=[], metavar="NAME",
        help="Target hardware label; repeatable. Defaults to the local CUDA device.",
    )
    return parser.parse_args()


def _read_sources(source_dir: Path) -> list[tuple[str, str]]:
    if not source_dir.is_dir():
        raise ValueError(f"{source_dir} is not a directory.")
    files = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(source_dir.iterdir()) if p.is_file()]
    if not files:
        raise ValueError(f"{source_dir} has no source files.")
    return files


def _resolve_hardware(hardware: list[str]) -> list[str]:
    if hardware:
        return hardware
    import torch

    return [torch.cuda.get_device_name(0).replace(" ", "_")]


def _build_solution(args: argparse.Namespace, sources: list[tuple[str, str]]):
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedBindings

    entry = args.entry or f"{ENTRY_FILE_BY_LANGUAGE.get(args.language, 'main.py')}::{ENTRY_SYMBOL}"
    return Solution(
        name=args.name or f"{args.definition}_baseline",
        definition=args.definition,
        author=args.author,
        spec=BuildSpec(
            language=args.language,
            target_hardware=_resolve_hardware(args.hardware),
            entry_point=entry,
            binding=SupportedBindings(args.binding),
            dependencies=args.deps,
            destination_passing_style=args.dps,
        ),
        sources=[SourceFile(path=name, content=content or "\n") for name, content in sources],
    )


def main() -> int:
    args = _parse_args()
    try:
        sources = _read_sources(args.source_dir)
        solution = _build_solution(args, sources)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(solution.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote solution {solution.name!r} ({len(sources)} source file(s)) to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
