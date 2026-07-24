"""Record a fully benchmarked working kernel in the active structural branch."""
from __future__ import annotations

import re
from typing import Optional

from tools.registry import registry
from tools._workspace import (
    BENCHMARK_CACHE_PATH,
    EXPERIMENTS_DIR,
    BenchmarkCache,
    read_src_files,
    solution_name_from_src_files,
)
from tools import _tree


SCHEMA = {
    "name": "log_experiment",
    "description": (
        "Record the fully benchmarked working kernel as an experiment in the "
        "active structural branch. Log every full-benchmark result that informs "
        "the branch, including regressions and correctness failures. The initial "
        "workspace baseline creates the root branch automatically; after that, "
        "create_handoff must assign a structural branch before experiments can "
        "be logged. Requires benchmark_kernel(scope='full') on this exact source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Short experiment identifier using lowercase letters, digits, "
                    "and underscores. Do not include the automatically assigned eN prefix."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Objective implementation description. Interpretation and "
                    "follow-up evidence belong in notes."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optimization ingredients used by this experiment.",
            },
            "notes": {
                "type": "string",
                "description": (
                    "Optional interpretation, caveat, or failure diagnosis. Do not "
                    "restate measurements recorded automatically."
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
    # Validate identifier
    if not _SLUG_RE.fullmatch(slug):
        return (
            f"Error: slug {slug!r} must match {_SLUG_RE.pattern} "
            "(lowercase letters, digits, underscores only)."
        )

    # Identify working kernel
    files = read_src_files()
    if not files:
        return "Error: the working kernel has no source files."
    solution_name = solution_name_from_src_files(files)

    # Load full evaluation
    cache = BenchmarkCache.load(BENCHMARK_CACHE_PATH)
    evaluation = cache.entries.get(solution_name)
    if evaluation is None:
        return (
            "Error: no full benchmark result for the current working kernel "
            f"(hashes to {solution_name!r}) — run benchmark_kernel with "
            "scope='full' first."
        )

    # Load optimization memory
    memory = _tree.load_memory()
    existing = _tree.find_experiment_by_solution(memory, solution_name)
    if existing is not None:
        return f"Error: this source snapshot is already logged as {existing}."

    # Resolve active branch
    branch_id = memory["active_branch"]
    is_baseline = not memory["experiments"] and branch_id is None
    if not is_baseline and branch_id is None:
        return (
            "Error: no active structural branch — call create_handoff with the "
            "next structure and its base experiment before logging more experiments."
        )

    parent = _tree.get_head(memory)
    if not is_baseline:
        branch = memory["branches"][branch_id]
        allowed_parents = {branch["base_experiment"], *branch["experiments"]}
        if parent not in allowed_parents:
            return (
                f"Error: head {parent!r} is outside active branch {branch_id!r}; "
                "use checkout_experiment within the active branch or create_handoff "
                "to start another structure."
            )

    # Allocate experiment
    bare_slug = re.sub(r"^e\d+_", "", slug)
    experiment_id = (
        f"e{_tree.next_experiment_number(memory, EXPERIMENTS_DIR)}_{bare_slug}"
    )

    # Snapshot sources
    snapshot_dir = EXPERIMENTS_DIR / experiment_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files:
        (snapshot_dir / name).write_text(content)

    # Record experiment
    evaluation_dict = evaluation.model_dump()
    if is_baseline:
        branch_id = f"b{_tree.next_branch_number(memory)}_baseline"
        _tree.add_baseline_branch(
            memory,
            branch_id=branch_id,
            experiment_id=experiment_id,
            solution=solution_name,
            description=description,
            tags=tags or [],
            evaluation=evaluation_dict,
            initial_note=notes,
        )
    else:
        _tree.add_experiment(
            memory,
            experiment_id=experiment_id,
            branch_id=branch_id,
            parent=parent,
            solution=solution_name,
            description=description,
            tags=tags or [],
            evaluation=evaluation_dict,
            initial_note=notes,
        )

    # Advance global state
    _tree.set_head(memory, experiment_id)
    _tree.update_bests(memory, experiment_id, evaluation_dict)

    # Persist memory
    _tree.save_memory(memory)
    return _tree.render_commit(memory, experiment_id)
