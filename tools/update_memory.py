"""Update structured annotations in the optimization memory."""
from __future__ import annotations

from typing import Optional

from tools.registry import registry
from tools import _tree


_ACTIONS = ("add", "replace", "remove")


SCHEMA = {
    "name": "update_memory",
    "description": (
        "Revise the optimization memory's structured annotations. 'add' "
        "(default) appends a new entry; 'replace' swaps an existing entry's "
        "text; 'remove' deletes one. Replace and remove identify the entry by "
        "old_text (a substring unique to it). Use 'note' or "
        "'profiling_observation' to attach to a specific experiment (requires "
        "experiment_id); use 'open_hypothesis', 'global_fact', or 'hazard' for "
        "cross-experiment annotations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": list(_tree.PER_EXPERIMENT_SCOPES + _tree.TOP_LEVEL_SCOPES),
                "description": "Which memory slot the annotation belongs to.",
            },
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "add a new entry (default), or replace / remove an existing one.",
            },
            "text": {
                "type": "string",
                "description": "The entry text. Required for 'add' and 'replace'.",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Substring uniquely identifying the existing entry to edit. "
                    "Required for 'replace' and 'remove'."
                ),
            },
            "experiment_id": {
                "type": "string",
                "description": "Required for 'note' and 'profiling_observation'; ignored otherwise.",
            },
        },
        "required": ["scope"],
    },
}


@registry.register(SCHEMA)
def update_memory(
    scope: str,
    action: str = "add",
    text: Optional[str] = None,
    old_text: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> str:
    # Validate request
    if scope not in _tree.PER_EXPERIMENT_SCOPES and scope not in _tree.TOP_LEVEL_SCOPES:
        valid = ", ".join(_tree.PER_EXPERIMENT_SCOPES + _tree.TOP_LEVEL_SCOPES)
        return f"Error: unknown scope {scope!r}; expected one of: {valid}."
    if action not in _ACTIONS:
        return f"Error: unknown action {action!r}; expected one of: {', '.join(_ACTIONS)}."

    # Validate content
    if action in ("add", "replace"):
        text = (text or "").strip()
        if not text:
            return f"Error: action {action!r} requires non-empty text."
    if action in ("replace", "remove"):
        old_text = (old_text or "").strip()
        if not old_text:
            return f"Error: action {action!r} requires old_text to identify the entry."

    # Load memory
    memory = _tree.load_memory()
    if not memory["experiments"]:
        return "Error: no experiments logged yet — run benchmark_kernel and log_experiment first."

    # Resolve experiment
    if scope in _tree.PER_EXPERIMENT_SCOPES:
        if not experiment_id:
            return f"Error: scope {scope!r} requires experiment_id."
        if not _tree.has_experiment(memory, experiment_id):
            return (
                f"Error: experiment {experiment_id!r} not found in memory. "
                f"Available: {_tree.list_experiment_ids(memory)}"
            )

    # Apply update
    if action == "add":
        _tree.add_annotation(memory, scope, text, experiment_id)
    else:
        new_text = text if action == "replace" else None
        err = _tree.edit_annotation(memory, scope, old_text, new_text, experiment_id)
        if err:
            return err

    # Persist memory
    _tree.save_memory(memory)
    rendered = _tree.render_annotation(memory, scope, experiment_id)
    if action == "add":
        return rendered
    verb = "Replaced" if action == "replace" else "Removed"
    return f"{verb} 1 entry matching {old_text!r}.\n{rendered}"
