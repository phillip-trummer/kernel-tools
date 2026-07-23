import unittest
from unittest.mock import patch

from tools import _tree
from tools.checkout_experiment import SCHEMA as CHECKOUT_SCHEMA
from tools.read_journal import read_journal


def _evaluation(speedup: float) -> dict:
    return {
        "status": "ALL_PASSED",
        "geomean_latency_ms": 1.0 / speedup,
        "geomean_speedup_factor": speedup,
        "workload_count": 1,
        "passed_workload_count": 1,
        "representative_workloads": {},
        "failure_statuses": {},
    }


def _branched_tree() -> dict:
    tree = _tree.bootstrap_tree(
        task="test_task",
        kernel_description="test kernel",
        hardware="test gpu",
        language="cuda",
    )
    _tree.add_node(
        tree,
        node_id="v0_root",
        parent=None,
        solution="root",
        description="root",
        tags=[],
        evaluation=_evaluation(1.0),
    )
    _tree.add_node(
        tree,
        node_id="v1_left",
        parent="v0_root",
        solution="left",
        description="left branch",
        tags=["left"],
        evaluation=_evaluation(2.0),
    )
    _tree.add_node(
        tree,
        node_id="v2_right",
        parent="v0_root",
        solution="right",
        description="right branch",
        tags=["right"],
        evaluation=_evaluation(3.0),
    )
    tree["head"] = "v2_right"
    tree["head_state"] = "clean"
    tree["current_best"] = "v2_right"
    return tree


class JournalViewTests(unittest.TestCase):
    def test_complete_journal_still_contains_every_node(self):
        rendered = _tree.render_journal(_branched_tree())
        self.assertIn("### `v0_root`", rendered)
        self.assertIn("### `v1_left`", rendered)
        self.assertIn("### `v2_right`", rendered)

    def test_frontier_journal_contains_only_branch_tips(self):
        rendered = _tree.render_frontier_journal(_branched_tree())
        self.assertNotIn("### `v0_root`", rendered)
        self.assertIn("### `v1_left`", rendered)
        self.assertIn("### `v2_right`", rendered)
        self.assertIn("2 branch tips shown from 3 experiments", rendered)

    def test_branch_journal_follows_root_to_selected_node(self):
        rendered = _tree.render_branch_journal(_branched_tree(), "v1_left")
        root_pos = rendered.index("### `v0_root`")
        leaf_pos = rendered.index("### `v1_left`")
        self.assertLess(root_pos, leaf_pos)
        self.assertNotIn("### `v2_right`", rendered)

    @patch("tools.read_journal._tree.save_tree")
    @patch("tools.read_journal._tree.load_tree")
    def test_read_journal_defaults_to_frontier_and_expands_selected_branch(
        self, load_tree, save_tree
    ):
        load_tree.return_value = _branched_tree()

        frontier = read_journal()
        branch = read_journal(experiment_id="v1_left")

        self.assertNotIn("### `v0_root`", frontier)
        self.assertIn("### `v0_root`", branch)
        self.assertIn("### `v1_left`", branch)
        self.assertNotIn("### `v2_right`", branch)
        self.assertEqual(save_tree.call_count, 2)

    def test_checkout_description_explains_branch_creation(self):
        description = CHECKOUT_SCHEMA["description"]
        self.assertIn("create a new branch", description)
        self.assertIn("log_experiment", description)


if __name__ == "__main__":
    unittest.main()
