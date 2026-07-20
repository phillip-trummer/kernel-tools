"""Record the current src/ tree as an optimization experiment.

Looks up a benchmark result for the current src/ in .state/benchmark_cache.json
(written by benchmark_kernel, keyed by src hash); any previously-benchmarked
snapshot qualifies, so reverting to an earlier-benchmarked kernel can be logged
without re-running. Snapshots src/ to experiments/vN_<slug>/, appends a node to .state/tree.json,
advances head, updates current_best / best_by_representative_workload if the new result is
better, and re-renders optimization_journal.md.
"""
from __future__ import annotations

import re
from typing import Optional

from tools.registry import registry
from tools._workspace import (
    EXPERIMENTS_DIR,
    BENCHMARK_CACHE_PATH,
    BenchmarkCache,
    read_src_files,
    solution_name_from_src_files,
)
from tools import _tree


SCHEMA = {
    "name": "log_experiment",
    "description": (
        "Record the current working kernel as a new experiment. The "
        "measured correctness and performance evaluation is recorded automatically. "
        "Requires a previous benchmark_kernel call with scope='full' on this exact "
        "source; smoke tests do not qualify. The new "
        "experiment becomes a child of head; head then advances to the "
        "new experiment. The journal is updated with its results and any "
        "provided notes/tags."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Short identifier for the experiment (lowercase letters, "
                    "digits, underscores only). Do not add a version number prefix — it is automatically assigned."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Objective implementation description. Observations and "
                    "follow-up ideas belong in 'notes', not here."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optimization ingredients / idea lineage tags, e.g. "
                    "['tiling', 'vectorized']."
                ),
            },
            "notes": {
                "type": "string",
                "description": (
                    "Optional initial note: interpretation, caveats, follow-up "
                    "ideas. Do not restate performance or representative workload numbers — the "
                    "measured evaluation is recorded automatically. More notes "
                    "can be appended later via annotate_journal."
                ),
            },
        },
        "required": ["slug", "description"],
    },
}

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@registry.register(SCHEMA)
def log_experiment(
    slug: str,
    description: str,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
) -> str:
    # Validate slug.
    if not _SLUG_RE.fullmatch(slug):
        return (
            f"Error: slug {slug!r} must match {_SLUG_RE.pattern} "
            "(lowercase letters, digits, underscores only)."
        )

    # Hash src/ into a candidate solution name.
    files = read_src_files()
    if not files:
        return "Error: the working kernel has no source files."
    solution_name = solution_name_from_src_files(files)

    # Pull the cached evaluation for the current src hash. Any snapshot tested
    # this run qualifies, so reverting to an earlier-tested kernel still logs.
    cache = BenchmarkCache.load(BENCHMARK_CACHE_PATH)
    evaluation = cache.entries.get(solution_name)
    if evaluation is None:
        return (
            f"Error: no benchmark result for the current working kernel "
            f"(hashes to {solution_name!r}) — run benchmark_kernel with "
            f"scope='full' first."
        )

    # Load tree.
    tree = _tree.load_tree()

    # Reject if this exact snapshot is already logged.
    existing = _tree.find_node_by_solution(tree, solution_name)
    if existing is not None:
        return f"Error: this source snapshot is already logged as {existing}."

    # Assign the next experiment id. Strip any leading vN_ the agent copied from
    # the versioned ids it sees in the journal, so we don't double-prefix
    # (e.g. slug 'v3_ldg_cache' -> 'v3_v3_ldg_cache').
    bare_slug = re.sub(r"^v\d+_", "", slug)
    new_id = f"v{_tree.next_version(tree, EXPERIMENTS_DIR)}_{bare_slug}"

    # Snapshot src/ into experiments/<new_id>/.
    parent = _tree.get_head(tree)
    snapshot_dir = EXPERIMENTS_DIR / new_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files:
        (snapshot_dir / name).write_text(content)

    # Append the node, advance head, and recompute bests.
    evaluation_dict = evaluation.model_dump()
    _tree.add_node(
        tree,
        node_id=new_id,
        parent=parent,
        solution=solution_name,
        description=description,
        tags=tags or [],
        evaluation=evaluation_dict,
        initial_note=notes,
    )
    _tree.set_head(tree, new_id)
    _tree.update_bests(tree, new_id, evaluation_dict)

    # Persist and return the live journal header + the just-logged node.
    _tree.save_tree(tree)
    return _tree.render_commit(tree, new_id)
