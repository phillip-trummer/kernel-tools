"""Annotate the optimization journal with structured notes.

The journal is rendered from durable structured state, so the agent annotates
it via this tool rather than hand-editing Markdown. The tool routes each
annotation into the right slot (per-experiment note or profiling observation,
or top-level hypothesis / fact / hazard), applies the requested edit (add a new
entry, or replace / remove an existing one), and re-renders the journal.
"""
from __future__ import annotations

from typing import Optional

from tools.registry import registry
from tools import _tree


_ACTIONS = ("add", "replace", "remove")


SCHEMA = {
    "name": "annotate_journal",
    "description": (
        "Revise the optimization journal's structured annotations. 'add' "
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
                "description": "Which journal slot the annotation belongs to.",
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
def annotate_journal(
    scope: str,
    action: str = "add",
    text: Optional[str] = None,
    old_text: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> str:
    # Validate scope and action.
    if scope not in _tree.PER_EXPERIMENT_SCOPES and scope not in _tree.TOP_LEVEL_SCOPES:
        valid = ", ".join(_tree.PER_EXPERIMENT_SCOPES + _tree.TOP_LEVEL_SCOPES)
        return f"Error: unknown scope {scope!r}; expected one of: {valid}."
    if action not in _ACTIONS:
        return f"Error: unknown action {action!r}; expected one of: {', '.join(_ACTIONS)}."

    # add / replace carry new text; replace / remove need a locator.
    if action in ("add", "replace"):
        text = (text or "").strip()
        if not text:
            return f"Error: action {action!r} requires non-empty text."
    if action in ("replace", "remove"):
        old_text = (old_text or "").strip()
        if not old_text:
            return f"Error: action {action!r} requires old_text to identify the entry."

    # Load tree.
    tree = _tree.load_tree()
    if not tree["nodes"]:
        return "Error: no experiments logged yet — run benchmark_kernel and log_experiment first."

    # Per-experiment scopes must name an existing experiment.
    if scope in _tree.PER_EXPERIMENT_SCOPES:
        if not experiment_id:
            return f"Error: scope {scope!r} requires experiment_id."
        if not _tree.has_node(tree, experiment_id):
            return (
                f"Error: experiment {experiment_id!r} not found in tree. "
                f"Available: {_tree.list_node_ids(tree)}"
            )

    # Apply the edit, routing through the tree's annotation helpers.
    if action == "add":
        _tree.add_annotation(tree, scope, text, experiment_id)
    else:
        new_text = text if action == "replace" else None
        err = _tree.edit_annotation(tree, scope, old_text, new_text, experiment_id)
        if err:
            return err

    # Persist and echo the affected slot, so its accumulated contents stay visible.
    _tree.save_tree(tree)
    rendered = _tree.render_annotation(tree, scope, experiment_id)
    if action == "add":
        return rendered
    verb = "Replaced" if action == "replace" else "Removed"
    return f"{verb} 1 entry matching {old_text!r}.\n{rendered}"
