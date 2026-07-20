"""Seed <workspace>/task/ from a local flashinfer-trace dataset copy.

Stages ONE definition's fixtures into the workspace in the layout the adapter
expects: definition.json, workloads.jsonl, the input blobs those workloads
reference, and optionally a baseline/target solution. Fetch the dataset first with
scripts/download_data.py.

The dataset stores one definition per file under definitions/<family>/<name>.json,
its workloads at workloads/<family>/<name>.jsonl, its candidate solutions under
solutions/<family>/<name>/, and safetensors inputs under blob/. A workload record
references a blob by a path relative to the dataset root ("./blob/..."), and task/
is that same root at runtime — so blobs are copied preserving their relative path.
Only the blobs the chosen definition's workloads name are copied.

    python scripts/seed_task.py --list
    python scripts/seed_task.py mla_paged_decode_h16_ckv512_kpe64_ps1 --list
    python scripts/seed_task.py mla_paged_decode_h16_ckv512_kpe64_ps1 \
        --baseline gpt-5_cuda --target flashinfer_wrapper
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "flashinfer-trace"


def _default_workspace() -> Path:
    """The workspace setup_workspace.py will use, from config.toml."""
    config = tomllib.loads((REPO_ROOT / "config.toml").read_text())
    path = Path(config.get("task", {}).get("workspace_path", ".agent_workspace"))
    return path if path.is_absolute() else REPO_ROOT / path


def _definitions(root: Path) -> dict[str, Path]:
    """Every available definition name -> its file."""
    return {p.stem: p for p in sorted(root.glob("definitions/*/*.json"))}


def _blob_paths(workloads: Path) -> list[str]:
    """Dataset-relative paths of every safetensors input these workloads reference."""
    paths: set[str] = set()
    for line in workloads.read_text().splitlines():
        if not line.strip():
            continue
        for spec in (json.loads(line)["workload"]["inputs"] or {}).values():
            if spec.get("type") == "safetensors":
                path = spec["path"]
                paths.add(path[2:] if path.startswith("./") else path)
    return sorted(paths)


def _resolve_solution(root: Path, family: str, definition: str, name: str) -> Path:
    """A solution for this definition, by exact stem or unique substring."""
    solution_dir = root / "solutions" / family / definition
    available = sorted(solution_dir.glob("*.json")) if solution_dir.is_dir() else []
    if not available:
        raise SystemExit(f"Error: the dataset ships no solutions for {definition!r}.")
    hits = [p for p in available if p.stem == name] or [p for p in available if name in p.stem]
    if not hits:
        names = "\n  ".join(p.stem for p in available)
        raise SystemExit(f"Error: no solution matching {name!r}. Available:\n  {names}")
    if len(hits) > 1:
        names = ", ".join(p.stem for p in hits)
        raise SystemExit(f"Error: {name!r} matches several solutions: {names}")
    return hits[0]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("definition", nargs="?", help="Definition name to stage (omit with --list to see all).")
    parser.add_argument("--list", action="store_true",
                        help="List available definitions, or a definition's solutions when one is named.")
    parser.add_argument("--baseline", help="Solution to stage as the starting kernel (exact stem or unique substring).")
    parser.add_argument("--target", help="Solution to stage as the comparison target.")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root (default: [task] workspace_path).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help=f"Local dataset copy (default: {DATA_DIR.relative_to(REPO_ROOT)}).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing task/ fixture set.")
    args = parser.parse_args(argv)

    root = args.data_dir
    if not (root / "definitions").is_dir():
        raise SystemExit(f"Error: no dataset at {root} — run scripts/download_data.py first.")
    definitions = _definitions(root)

    # Listing modes: all definitions, or one definition's solutions.
    if args.list and not args.definition:
        for path in definitions.values():
            print(f"{path.parent.name:12} {path.stem}")
        return 0
    if not args.definition:
        raise SystemExit("Error: name a definition (or pass --list to see the available ones).")
    if args.definition not in definitions:
        raise SystemExit(f"Error: definition {args.definition!r} not found. Use --list to see available definitions.")

    definition_path = definitions[args.definition]
    family = definition_path.parent.name
    if args.list:
        solution_dir = root / "solutions" / family / args.definition
        for path in sorted(solution_dir.glob("*.json")) if solution_dir.is_dir() else []:
            print(path.stem)
        return 0

    workloads_path = root / "workloads" / family / f"{args.definition}.jsonl"
    if not workloads_path.is_file():
        raise SystemExit(f"Error: the dataset ships no workloads for {args.definition!r}.")

    # Resolve solutions before writing anything, so a bad name fails cleanly.
    baseline = _resolve_solution(root, family, args.definition, args.baseline) if args.baseline else None
    target = _resolve_solution(root, family, args.definition, args.target) if args.target else None

    task_dir = (args.workspace or _default_workspace()) / "task"
    if (task_dir / "definition.json").exists() and not args.force:
        raise SystemExit(f"Error: {task_dir}/definition.json already exists — pass --force to replace the fixtures.")

    # Copy only the blobs these workloads name, preserving the dataset-relative
    # layout so the "./blob/..." paths still resolve against task/.
    blobs = _blob_paths(workloads_path)
    missing = [b for b in blobs if not (root / b).is_file()]
    if missing:
        raise SystemExit(
            f"Error: {len(missing)} blob(s) this definition needs are not in {root} "
            f"(e.g. {missing[0]}) — re-run scripts/download_data.py without --metadata-only."
        )

    _copy(definition_path, task_dir / "definition.json")
    _copy(workloads_path, task_dir / "workloads.jsonl")
    for blob in blobs:
        _copy(root / blob, task_dir / blob)
    print(f"[Task] staged {args.definition} ({family}) into {task_dir}: "
          f"definition.json, workloads.jsonl, {len(blobs)} blob(s).")

    for solution, sub in ((baseline, "baseline"), (target, "target")):
        if solution is None:
            continue
        _copy(solution, task_dir / sub / solution.name)
        print(f"[Task] staged {sub}: task/{sub}/{solution.name}")

    # The fixtures are in place; config still points setup at them by path.
    print("\nNext: set the paths in config.toml [task], then run scripts/setup_workspace.py")
    if baseline:
        print(f'  baseline = "task/baseline/{baseline.name}"')
    if target:
        print(f'  [task.target] path = "task/target/{target.name}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
