"""Diff a logged experiment against the current working kernel.

Lets the agent see what it has changed in the working kernel relative to a
logged experiment, without checking anything out. To compare two logged
experiments, check one out and diff against the other. Optional filename
narrows the diff to a single file; otherwise all files on either side are
diffed.
"""
import difflib
from pathlib import Path
from typing import Optional

from tools.registry import registry
from tools._workspace import read_src_files, resolve_experiment_dir

SCHEMA = {
    "name": "diff_experiment",
    "description": (
        "Show a unified diff between a logged experiment and the current "
        "working kernel. Useful for seeing what you have changed since an "
        "experiment, or which code change moved performance. To compare two "
        "logged experiments, check one out first, then diff against the other."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment id to diff against, e.g. 'v2_tiled'.",
            },
            "filename": {
                "type": "string",
                "description": "Restrict diff to a single file. Diffs all files if omitted.",
            },
        },
        "required": ["experiment"],
    },
}


@registry.register(SCHEMA)
def diff_experiment(
    experiment: str,
    filename: Optional[str] = None,
) -> str:
    # Resolve the experiment snapshot (path-traversal safe; clear error if missing).
    exp_dir = resolve_experiment_dir(experiment)
    if not isinstance(exp_dir, Path):
        return f"Error: {exp_dir}"

    exp_files = {p.name: p.read_text() for p in exp_dir.iterdir() if p.is_file()}
    working_files = dict(read_src_files())

    # Pick which files to diff.
    if filename is not None:
        if filename not in exp_files and filename not in working_files:
            return f"Error: neither {experiment!r} nor the working kernel contains {filename!r}."
        names = [filename]
    else:
        names = sorted(exp_files.keys() | working_files.keys())

    # Unified diff per file; a file absent on one side diffs against empty,
    # and identical files yield nothing and are skipped.
    chunks = []
    for name in names:
        diff = difflib.unified_diff(
            exp_files.get(name, "").splitlines(keepends=True),
            working_files.get(name, "").splitlines(keepends=True),
            fromfile=f"{experiment}/{name}",
            tofile=f"working_kernel/{name}",
        )
        text = "".join(diff)
        if text:
            chunks.append(text)

    if not chunks:
        scope = f" for {filename!r}" if filename else ""
        return f"No differences between {experiment} and the working kernel{scope}."
    return "\n".join(chunks)
