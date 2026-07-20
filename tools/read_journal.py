"""Read the rendered optimization journal.

The journal is the agent's persistent, agent-readable memory across context
resets, so this is the first thing a new agent or subagent should call.
"""
from tools.registry import registry
from tools._tree import JOURNAL_PATH, load_tree, save_tree

SCHEMA = {
    "name": "read_journal",
    "description": (
        "Read the optimization journal: a compact summary of all logged "
        "experiments, their results, and accumulated notes."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


# Write side effect: read_journal calls save_tree before reading so that the
# working-kernel dirty marker on tree.json (and the rendered journal) reflects
# the current src/ rather than the state at the last journal-mutating call.
# Nothing else about the tree changes — save_tree only refreshes head_state.
@registry.register(SCHEMA)
def read_journal() -> str:
    save_tree(load_tree())
    return JOURNAL_PATH.read_text()
