"""Read an agent-sized view of the optimization memory."""
from typing import Optional

from tools.registry import registry
from tools import _tree

SCHEMA = {
    "name": "read_memory",
    "description": (
        "Read the optimization memory. With no branch_id, returns the task, "
        "shared knowledge, active structural assignment, and one representative "
        "experiment for every frontier branch. Pass branch_id to inspect its "
        "strategy, conclusion, and complete experiment history."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch_id": {
                "type": "string",
                "description": (
                    "Optional structural branch to inspect, e.g. "
                    "'b2_eight_stage_pipeline'."
                ),
            },
        },
    },
}


@registry.register(SCHEMA)
def read_memory(branch_id: Optional[str] = None) -> str:
    # Refresh memory
    memory = _tree.load_memory()
    _tree.save_memory(memory)

    # Render frontier
    if branch_id is None:
        return _tree.render_frontier_memory(memory)

    # Render branch
    if not _tree.has_branch(memory, branch_id):
        return (
            f"Error: branch {branch_id!r} not found. "
            f"Available: {_tree.list_branch_ids(memory)}"
        )
    return _tree.render_branch_memory(memory, branch_id)
