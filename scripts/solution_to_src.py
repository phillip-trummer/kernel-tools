"""Materialize a FlashInfer solution JSON into source files.

Usage:
  python scripts/solution_to_src.py path/to/solution.json path/to/src_dir

The destination is created if needed. Existing files with the same names are
overwritten; other files are left alone unless --clean is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a FlashInfer solution object's sources into a directory."
    )
    parser.add_argument("solution_json", type=Path, help="Path to the solution JSON object.")
    parser.add_argument("destination_dir", type=Path, help="Directory to receive source files.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing contents of destination_dir before writing sources.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list the files that would be written without changing disk.",
    )
    return parser.parse_args()


def _load_solution(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{path} is not a file.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _validate_source_path(raw_path: Any) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"source path must be a non-empty string, got {raw_path!r}.")
    if "\0" in raw_path:
        raise ValueError(f"source path contains a NUL byte: {raw_path!r}.")

    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"source path must be relative: {raw_path!r}.")
    if any(part in ("", ".", "..") for part in posix_path.parts):
        raise ValueError(f"source path must not contain empty, '.', or '..' parts: {raw_path!r}.")
    return posix_path


def _source_entries(solution: dict[str, Any]) -> list[tuple[PurePosixPath, str]]:
    sources = solution.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("solution must contain a non-empty 'sources' list.")

    entries: list[tuple[PurePosixPath, str]] = []
    seen: set[str] = set()
    for i, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"sources[{i}] must be an object.")
        rel_path = _validate_source_path(source.get("path"))
        content = source.get("content")
        if not isinstance(content, str):
            raise ValueError(f"sources[{i}].content must be a string.")
        normalized = rel_path.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate source path: {normalized!r}.")
        seen.add(normalized)
        entries.append((rel_path, content))
    return entries


def _resolve_output_path(destination_dir: Path, rel_path: PurePosixPath) -> Path:
    dest = destination_dir.resolve(strict=False)
    out = dest.joinpath(*rel_path.parts).resolve(strict=False)
    if dest not in out.parents:
        raise ValueError(f"source path escapes destination: {rel_path.as_posix()!r}.")
    return out


def _clean_destination(destination_dir: Path) -> None:
    dest = destination_dir.resolve(strict=False)
    if dest.parent == dest:
        raise ValueError("refusing to clean filesystem root.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    for entry in destination_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def materialize_solution(solution_path: Path, destination_dir: Path, *, clean: bool, dry_run: bool) -> list[Path]:
    solution = _load_solution(solution_path)
    entries = _source_entries(solution)

    outputs = [_resolve_output_path(destination_dir, rel_path) for rel_path, _ in entries]
    if dry_run:
        return outputs

    if clean:
        _clean_destination(destination_dir)
    else:
        destination_dir.mkdir(parents=True, exist_ok=True)

    for (rel_path, content), out_path in zip(entries, outputs):
        # Re-resolve after cleaning/creating directories in case symlinks changed.
        out_path = _resolve_output_path(destination_dir, rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
    return outputs


def main() -> int:
    args = _parse_args()
    try:
        solution = _load_solution(args.solution_json)
        entries = _source_entries(solution)
        outputs = [
            _resolve_output_path(args.destination_dir, rel_path)
            for rel_path, _ in entries
        ]

        if not args.dry_run:
            if args.clean:
                _clean_destination(args.destination_dir)
            else:
                args.destination_dir.mkdir(parents=True, exist_ok=True)
            for (rel_path, content), _ in zip(entries, outputs):
                out_path = _resolve_output_path(args.destination_dir, rel_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(content, encoding="utf-8")

        action = "would write" if args.dry_run else "wrote"
        name = solution.get("name", args.solution_json.name)
        print(f"{action} {len(outputs)} source file(s) from {name!r} to {args.destination_dir}:")
        dest = args.destination_dir.resolve(strict=False)
        for out_path in outputs:
            try:
                display = out_path.relative_to(dest)
            except ValueError:
                display = out_path
            print(f"  {display}")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
