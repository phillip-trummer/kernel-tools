import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import _tree
from tools.create_handoff import create_handoff
from tools.log_experiment import log_experiment
from tools.read_memory import read_memory
from tools._evaluation import Evaluation
from tools._workspace import BenchmarkCache, solution_name_from_src_files


def _evaluation(speedup: float | None, status: str = "ALL_PASSED") -> dict:
    return {
        "status": status,
        "geomean_latency_ms": 1.0 / speedup if speedup else None,
        "geomean_speedup_factor": speedup,
        "workload_count": 1,
        "passed_workload_count": 1 if speedup else 0,
        "representative_workloads": {},
        "failure_statuses": {} if speedup else {"COMPILE_ERROR": 1},
    }


def _memory() -> dict:
    memory = _tree.bootstrap_memory(
        task="test_task",
        kernel_description="test kernel",
        hardware="test gpu",
        language="cuda",
    )
    _tree.add_baseline_branch(
        memory,
        branch_id="b0_baseline",
        experiment_id="e0_baseline",
        solution="baseline",
        description="Initial baseline.",
        tags=[],
        evaluation=_evaluation(1.0),
    )
    _tree.add_branch(
        memory,
        branch_id="b1_four_stage",
        base_experiment="e0_baseline",
        strategy="Implement a four-stage pipeline.",
    )
    _tree.add_experiment(
        memory,
        experiment_id="e1_four_stage",
        branch_id="b1_four_stage",
        parent="e0_baseline",
        solution="four",
        description="Four-stage structure.",
        tags=[],
        evaluation=_evaluation(2.0),
    )
    _tree.add_experiment(
        memory,
        experiment_id="e2_four_stage_regression",
        branch_id="b1_four_stage",
        parent="e1_four_stage",
        solution="four_bad",
        description="Regressive tuning.",
        tags=[],
        evaluation=_evaluation(1.5),
    )
    _tree.complete_branch(memory, "b1_four_stage", "Four stages tuned.", True)
    _tree.add_branch(
        memory,
        branch_id="b2_eight_stage",
        base_experiment="e0_baseline",
        strategy="Implement an eight-stage pipeline.",
    )
    _tree.add_experiment(
        memory,
        experiment_id="e3_eight_stage",
        branch_id="b2_eight_stage",
        parent="e0_baseline",
        solution="eight",
        description="Eight-stage structure.",
        tags=[],
        evaluation=_evaluation(3.0),
    )
    memory["head"] = "e3_eight_stage"
    memory["head_state"] = "clean"
    memory["current_best"] = "e3_eight_stage"
    return memory


class MemoryViewTests(unittest.TestCase):
    def test_live_tools_reject_legacy_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                Path(".state").mkdir()
                Path(".state/tree.json").write_text("{}")

                with self.assertRaises(FileNotFoundError):
                    _tree.load_memory()
            finally:
                os.chdir(previous)

    def test_complete_memory_contains_every_branch_and_experiment(self):
        rendered = _tree.render_memory(_memory())
        for branch_id in ("b0_baseline", "b1_four_stage", "b2_eight_stage"):
            self.assertIn(f"`{branch_id}`", rendered)
        for experiment_id in (
            "e0_baseline",
            "e1_four_stage",
            "e2_four_stage_regression",
            "e3_eight_stage",
        ):
            self.assertIn(f"`{experiment_id}`", rendered)

    def test_frontier_contains_one_representative_per_structure(self):
        rendered = _tree.render_frontier_memory(_memory())
        self.assertIn("e1_four_stage", rendered)
        self.assertNotIn("e2_four_stage_regression", rendered)
        self.assertIn("e3_eight_stage", rendered)
        self.assertIn("3 frontier structures shown", rendered)

    def test_branch_view_contains_complete_local_history(self):
        rendered = _tree.render_branch_memory(_memory(), "b1_four_stage")
        self.assertIn("e1_four_stage", rendered)
        self.assertIn("e2_four_stage_regression", rendered)
        self.assertNotIn("Eight-stage structure.", rendered)
        self.assertEqual(
            _tree.parent_branch_id(_memory(), "b1_four_stage"),
            "b0_baseline",
        )

    def test_branch_can_nest_inside_structural_variant(self):
        memory = _memory()
        _tree.add_branch(
            memory,
            branch_id="b3_four_stage_split_load",
            base_experiment="e1_four_stage",
            strategy="Split four-stage loads across producer warps.",
        )

        self.assertEqual(
            _tree.parent_branch_id(memory, "b3_four_stage_split_load"),
            "b1_four_stage",
        )

    @patch("tools.read_memory._tree.save_memory")
    @patch("tools.read_memory._tree.load_memory")
    def test_read_memory_defaults_to_frontier_and_expands_branch(
        self,
        load_memory,
        save_memory,
    ):
        load_memory.return_value = _memory()

        frontier = read_memory()
        branch = read_memory(branch_id="b1_four_stage")

        self.assertNotIn("e2_four_stage_regression", frontier)
        self.assertIn("e2_four_stage_regression", branch)
        self.assertEqual(save_memory.call_count, 2)


class HandoffTests(unittest.TestCase):
    def test_handoff_creates_planned_branch_and_restores_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                Path("src").mkdir()
                Path("src/kernel.cu").write_text("baseline\n")
                Path("experiments/e0_baseline").mkdir(parents=True)
                Path("experiments/e0_baseline/kernel.cu").write_text("baseline\n")
                solution = solution_name_from_src_files([("kernel.cu", "baseline\n")])
                memory = _tree.bootstrap_memory(
                    task="test",
                    kernel_description="",
                    hardware="gpu",
                    language="cuda",
                )
                _tree.add_baseline_branch(
                    memory,
                    branch_id="b0_baseline",
                    experiment_id="e0_baseline",
                    solution=solution,
                    description="Baseline.",
                    tags=[],
                    evaluation=_evaluation(1.0),
                )
                memory["head"] = "e0_baseline"
                _tree.save_memory(memory)

                result = create_handoff(
                    slug="eight_stage",
                    strategy="Implement and tune an eight-stage pipeline.",
                )
                updated = _tree.load_memory()

                self.assertIn("Created structural branch `b1_eight_stage`", result)
                self.assertIn("initial assignment", result)
                self.assertEqual(updated["active_branch"], "b1_eight_stage")
                self.assertEqual(
                    updated["branches"]["b1_eight_stage"]["base_experiment"],
                    "e0_baseline",
                )
                self.assertEqual(updated["branches"]["b1_eight_stage"]["state"], "planned")
            finally:
                os.chdir(previous)

    def test_failed_experiment_survives_terminal_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                # Seed baseline
                Path("src").mkdir()
                Path("src/kernel.cu").write_text("baseline\n")
                baseline_solution = solution_name_from_src_files(
                    [("kernel.cu", "baseline\n")]
                )
                cache = BenchmarkCache()
                cache.record(
                    baseline_solution,
                    Evaluation.model_validate(_evaluation(1.0)),
                )
                cache.save(Path(".state/benchmark_cache.json"))
                _tree.save_memory(
                    _tree.bootstrap_memory(
                        task="test",
                        kernel_description="",
                        hardware="gpu",
                        language="cuda",
                    )
                )
                self.assertIn(
                    "e0_baseline",
                    log_experiment(slug="baseline", description="Baseline."),
                )

                # Bootstrap structure
                create_handoff(
                    slug="four_stage",
                    strategy="Implement and tune a four-stage pipeline.",
                )

                # Log failed evidence
                Path("src/kernel.cu").write_text("failed structure\n")
                failed_solution = solution_name_from_src_files(
                    [("kernel.cu", "failed structure\n")]
                )
                cache = BenchmarkCache.load(Path(".state/benchmark_cache.json"))
                cache.record(
                    failed_solution,
                    Evaluation.model_validate(_evaluation(None, "FAILED")),
                )
                cache.save(Path(".state/benchmark_cache.json"))
                self.assertIn(
                    "e1_failed_pipeline",
                    log_experiment(
                        slug="failed_pipeline",
                        description="Four-stage pipeline failed to compile.",
                        notes="Shared-memory layout exceeded the compiler limit.",
                    ),
                )

                # Hand off from baseline
                result = create_handoff(
                    slug="eight_stage",
                    strategy="Implement and tune an eight-stage pipeline.",
                    continue_from="e0_baseline",
                    conclusion="Four-stage layout failed after tuning.",
                    keep_current_on_frontier=False,
                )
                updated = _tree.load_memory()

                self.assertIn("End this optimization session", result)
                self.assertIn("e1_failed_pipeline", updated["experiments"])
                self.assertEqual(
                    updated["branches"]["b1_four_stage"]["experiments"],
                    ["e1_failed_pipeline"],
                )
                self.assertEqual(
                    updated["experiments"]["e1_failed_pipeline"]["notes"],
                    ["Shared-memory layout exceeded the compiler limit."],
                )
                self.assertNotIn("b1_four_stage", updated["frontier"])
                self.assertEqual(updated["active_branch"], "b2_eight_stage")
                self.assertEqual(Path("src/kernel.cu").read_text(), "baseline\n")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
