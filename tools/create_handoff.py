"""Finish one structural branch and assign the next optimization session."""
from __future__ import annotations

import re
from typing import Optional

from tools.registry import registry
from tools._workspace import restore_experiment
from tools import _tree


SCHEMA = {
    "name": "create_handoff",
    "description": (
        "Create the first structural assignment when none exists, or finish the "
        "active branch and create the next planned branch. "
        "One optimization session owns one branch: implement its structure, fully "
        "benchmark and log every informative tuning attempt, then call this tool "
        "as the terminal action. The only nonterminal call is the initial bootstrap "
        "after setup. The tool restores continue_from so the assigned session starts "
        "mechanically from the selected experiment. It requires a clean logged "
        "working kernel and never discards branch history."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Short structural branch identifier using lowercase letters, "
                    "digits, and underscores. Do not include the assigned bN prefix."
                ),
            },
            "strategy": {
                "type": "string",
                "description": (
                    "Concrete new structure the next session must implement and "
                    "then tune fully. Include its rationale in the same text when useful."
                ),
            },
            "continue_from": {
                "type": "string",
                "description": (
                    "Logged experiment to restore as the new branch's code base. "
                    "Defaults to the current head."
                ),
            },
            "conclusion": {
                "type": "string",
                "description": (
                    "Durable conclusion from the branch being finished. Required "
                    "when an active branch exists; omitted when assigning the first branch."
                ),
            },
            "keep_current_on_frontier": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Keep the finished structure available to future sessions. "
                    "Set false only when evidence refutes it or another structure "
                    "dominates it."
                ),
            },
        },
        "required": ["slug", "strategy"],
    },
}

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@registry.register(SCHEMA)
def create_handoff(
    slug: str,
    strategy: str,
    continue_from: Optional[str] = None,
    conclusion: Optional[str] = None,
    keep_current_on_frontier: bool = True,
) -> str:
    # Validate inputs
    if not _SLUG_RE.fullmatch(slug):
        return (
            f"Error: slug {slug!r} must match {_SLUG_RE.pattern} "
            "(lowercase letters, digits, underscores only)."
        )
    strategy = strategy.strip()
    if not strategy:
        return "Error: strategy must be non-empty."

    # Load current assignment
    memory = _tree.load_memory()
    _tree.refresh_head_state(memory)
    active_id = memory["active_branch"]
    active = memory["branches"].get(active_id) if active_id else None
    is_bootstrap = active is None

    # Require completed evidence
    if active:
        if not active["experiments"]:
            return (
                f"Error: active branch {active_id!r} has no logged experiment — "
                "implement, fully benchmark, and log its assigned structure first."
            )
        if memory["head_state"] != "clean":
            return (
                "Error: the working kernel is not logged at head — fully benchmark "
                "and log the current attempt, including a failure or regression, "
                "or restore an experiment from the active branch first."
            )
        conclusion = (conclusion or "").strip()
        if not conclusion:
            return "Error: conclusion is required when finishing an active branch."

    # Resolve next base
    base_experiment = continue_from or memory["head"]
    if base_experiment is not None and not _tree.has_experiment(memory, base_experiment):
        return (
            f"Error: experiment {base_experiment!r} not found. "
            f"Available: {_tree.list_experiment_ids(memory)}"
        )

    # Restore next base
    file_count = 0
    if base_experiment is not None:
        restored = restore_experiment(base_experiment)
        if isinstance(restored, str):
            return f"Error: {restored}"
        file_count = restored

    # Finish current branch
    if active_id:
        _tree.complete_branch(
            memory,
            active_id,
            conclusion,
            keep_current_on_frontier,
        )

    # Create next branch
    bare_slug = re.sub(r"^b\d+_", "", slug)
    branch_id = f"b{_tree.next_branch_number(memory)}_{bare_slug}"
    _tree.add_branch(
        memory,
        branch_id=branch_id,
        base_experiment=base_experiment,
        strategy=strategy,
    )
    if base_experiment is not None:
        _tree.set_head(memory, base_experiment)

    # Persist handoff
    _tree.save_memory(memory)
    return _tree.render_handoff(memory, branch_id, file_count, is_bootstrap)
