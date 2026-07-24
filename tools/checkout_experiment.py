"""Restore an experiment within the active structural branch."""
from tools.registry import registry
from tools._workspace import restore_experiment
from tools import _tree


SCHEMA = {
    "name": "checkout_experiment",
    "description": (
        "Restore a logged experiment as the working kernel. During an active "
        "structural branch, checkout is limited to that branch's base and its "
        "own experiments, making it a branch-local rollback tool. Use "
        "create_handoff to move across structures. Uncommitted working edits "
        "are discarded."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "string",
                "description": "Experiment id, e.g. 'e0_baseline' or 'e12_tiled'.",
            },
        },
        "required": ["experiment_id"],
    },
}


@registry.register(SCHEMA)
def checkout_experiment(experiment_id: str) -> str:
    # Load optimization memory
    memory = _tree.load_memory()
    if not _tree.has_experiment(memory, experiment_id):
        return (
            f"Error: experiment {experiment_id!r} not found. "
            f"Available: {_tree.list_experiment_ids(memory)}"
        )

    # Enforce branch boundary
    active_id = memory["active_branch"]
    if active_id:
        branch = memory["branches"][active_id]
        allowed = {branch["base_experiment"], *branch["experiments"]}
        if experiment_id not in allowed:
            return (
                f"Error: experiment {experiment_id!r} is outside active branch "
                f"{active_id!r}. Call create_handoff to start another structure "
                "from that experiment."
            )

    # Restore snapshot
    restored = restore_experiment(experiment_id)
    if isinstance(restored, str):
        return f"Error: {restored}"

    # Advance head
    previous_head = _tree.get_head(memory)
    _tree.set_head(memory, experiment_id)
    _tree.save_memory(memory)
    return _tree.render_checkout(memory, experiment_id, previous_head, restored)
