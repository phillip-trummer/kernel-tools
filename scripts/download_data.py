"""Download the flashinfer-trace dataset into data/.

The counterpart to SOL-ExecBench's scripts/download_data.sh: fetches the
HuggingFace dataset `flashinfer-ai/flashinfer-trace` — definitions, workloads,
candidate solutions, and the safetensors input blobs — into data/flashinfer-trace/.
Seed a workspace task from that local copy with scripts/seed_task.py.

The full dataset is ~13 GB, almost entirely blobs. `--metadata-only` fetches just
the definitions/workloads/solutions (a few MB), enough to browse the corpus and to
seed a task whose workloads need no safetensors inputs.

`--revision` defaults to `main` (newest). Pin it only to reproduce an older fixture:
the reference implementation is the correctness oracle and the speedup denominator,
and it does change across revisions (tag `1.0`'s MLA decode reference returns a dict
where `main` returns a tuple).

    python scripts/download_data.py
    python scripts/download_data.py --metadata-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "flashinfer-ai/flashinfer-trace"
REVISION = "main"
DEST = REPO_ROOT / "data" / "flashinfer-trace"

# Everything except blob/, which is the bulk of the dataset.
METADATA_PATTERNS = ["definitions/**", "workloads/**", "solutions/**", "*.md"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEST, help=f"Destination directory (default: {DEST.relative_to(REPO_ROOT)}).")
    parser.add_argument("--repo", default=REPO_ID, help=f"HF dataset repo (default: {REPO_ID}).")
    parser.add_argument("--revision", default=REVISION, help=f"Dataset revision, branch or tag (default: {REVISION}).")
    parser.add_argument("--metadata-only", action="store_true", help="Skip blob/ (~13 GB); fetch definitions, workloads, and solutions only.")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Error: huggingface_hub is required (pip install huggingface_hub).")

    scope = "metadata (definitions, workloads, solutions)" if args.metadata_only else "full dataset (~13 GB)"
    print(f"[Data] downloading {args.repo}@{args.revision} — {scope} → {args.dest}")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.dest,
        allow_patterns=METADATA_PATTERNS if args.metadata_only else None,
    )

    counts = {
        name: sum(1 for p in (args.dest / name).rglob("*") if p.is_file())
        for name in ("definitions", "workloads", "solutions", "blob")
        if (args.dest / name).is_dir()
    }
    summary = ", ".join(f"{n} {name}" for name, n in counts.items())
    print(f"[Data] done: {summary or 'no files'}.")
    print(f"\nNext: seed a task from it, e.g.\n  python scripts/seed_task.py --list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
