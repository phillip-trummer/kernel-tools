import shutil
from pathlib import Path

from tools.registry import registry
from tools._workspace import SRC_DIR, resolve_experiment_dir
from tools import _tree

SCHEMA = {
    "name": "checkout_experiment",
    "description": (
        "Move head to a previously logged experiment and restore its "
        "source as the current working kernel. Overwrites and removes "
        "files in the working source as needed. Any uncommitted edits to "
        "the working source are discarded — log first if you want to keep "
        "them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "string",
                "description": "Experiment id, e.g. 'v0_baseline' or 'v2_tiled'.",
            },
        },
        "required": ["experiment_id"],
    },
}


@registry.register(SCHEMA)
def checkout_experiment(experiment_id: str) -> str:
    # Resolve the experiment snapshot.
    exp_dir = resolve_experiment_dir(experiment_id)
    if not isinstance(exp_dir, Path):
        return f"Error: {exp_dir}"
    exp_files = [p for p in exp_dir.iterdir() if p.is_file()]
    if not exp_files:
        return f"Error: experiment {experiment_id!r} has no source files; refusing to wipe working tree."

    # Load tree and confirm the journal knows this node.
    tree = _tree.load_tree()
    if not _tree.has_node(tree, experiment_id):
        return (
            f"Error: experiment {experiment_id!r} exists under experiments/ but "
            "is missing from .state/tree.json; refusing to check out a node the "
            "journal does not know about. Investigate the drift before proceeding."
        )

    # Mirror the snapshot into src/.
    src_dir = SRC_DIR.resolve()
    src_dir.mkdir(parents=True, exist_ok=True)
    exp_names = {p.name for p in exp_files}
    for stale in src_dir.iterdir():
        if stale.is_file() and stale.name not in exp_names:
            stale.unlink()
    for p in exp_files:
        shutil.copyfile(p, src_dir / p.name)

    # Advance head.
    prev_head = _tree.get_head(tree)
    _tree.set_head(tree, experiment_id)
    _tree.save_tree(tree)

    return _tree.render_checkout(tree, experiment_id, prev_head, len(exp_files))
