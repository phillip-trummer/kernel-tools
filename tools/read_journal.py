"""Read an agent-sized view of the optimization journal.

The journal is the agent's persistent, agent-readable memory across context
resets. The default view contains only branch tips; a selected experiment
expands the complete branch from the root to that node.
"""
from typing import Optional

from tools.registry import registry
from tools import _tree

SCHEMA = {
    "name": "read_journal",
    "description": (
        "Read the optimization journal. With no experiment_id, returns the "
        "task, shared memory, and only the frontier node of each experiment "
        "branch. Pass experiment_id to expand the full branch from its root "
        "through that experiment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "string",
                "description": (
                    "Optional branch tip or other experiment to inspect. "
                    "Returns its complete root-to-experiment lineage."
                ),
            },
        },
    },
}


# Write side effect: read_journal calls save_tree before rendering its view so
# the working-kernel dirty marker on tree.json (and every journal view) reflects
# current src/ rather than the state at the last journal-mutating call. Nothing
# else about the tree changes — save_tree only refreshes head_state.
@registry.register(SCHEMA)
def read_journal(experiment_id: Optional[str] = None) -> str:
    tree = _tree.load_tree()
    _tree.save_tree(tree)
    if experiment_id is None:
        return _tree.render_frontier_journal(tree)
    if not _tree.has_node(tree, experiment_id):
        return (
            f"Error: experiment {experiment_id!r} not found. "
            f"Available: {_tree.list_node_ids(tree)}"
        )
    try:
        return _tree.render_branch_journal(tree, experiment_id)
    except ValueError as exc:
        return f"Error: invalid experiment lineage: {exc}."
