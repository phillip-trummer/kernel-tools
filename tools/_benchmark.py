"""Portable benchmark-adapter interface — the product integration contract.

The harness talks to benchmark frameworks only through this interface, so tools
never handle a framework's native types. An adapter owns its task fixtures internally and
returns neutral results: a `TaskSpec`, and per-workload `WorkloadResult` leaves
that shared harness code (`aggregate`) folds into the stored `Evaluation` (see
_evaluation.py). No framework imports live here — adding a benchmark means adding
a module under `tools/adapters/` and registering its name in `get_adapter()`,
not touching the tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from tools._evaluation import TaskSpec, WorkloadResult
from tools._workspace import read_benchmark_state


class BenchmarkUnavailable(RuntimeError):
    """The benchmark could not evaluate the kernel, for reasons that are not the
    kernel's fault (no usable timer, a broken toolchain, staging failed).

    Distinct from a kernel that fails: those are `WorkloadResult`s with a failing
    outcome, which the agent should act on. This is the absence of a verdict, so
    an adapter raises it rather than inventing leaves — the message reaches the
    agent, and nothing is scored or logged. Keep the message in agent vocabulary."""


class BenchmarkAdapter(Protocol):
    """What every benchmark adapter exposes to the harness. Any method may raise
    `BenchmarkUnavailable` when the benchmark itself is unusable."""

    def benchmark(self, scope: str) -> list[WorkloadResult]:
        """Build the current working kernel, run it against the workloads ('full'
        suite or 'smoke' representatives), and return one neutral WorkloadResult
        per workload. The adapter does not aggregate — shared harness code folds
        the leaves into the stored Evaluation."""
        ...

    def benchmark_target(self, target_path: Path) -> tuple[list[WorkloadResult], str]:
        """Benchmark the comparison target the user supplied at target_path (in
        this adapter's native format) against the full workload suite; return
        (leaves, target_id). How the target is packaged there is the adapter's
        concern; the harness only gets neutral leaves back."""
        ...

    def representative_axes(self) -> dict[str, dict[str, int]]:
        """Map each representative label to its workload's concrete axes."""
        ...

    def sort_workloads_fixture(self) -> bool:
        """Order the task's workload fixture smallest-to-largest by total work, in
        place, so representative selection is monotonic. Returns True if the order
        changed. The one place an adapter writes back to task/; setup invokes it
        behind a flag, never the runtime tools."""
        ...

    def task_spec(self) -> TaskSpec:
        """The neutral task description seeded onto the tree at setup."""
        ...

    def baseline_files(self, baseline_path: Path) -> list[tuple[str, str]]:
        """The user's starting kernel at baseline_path (in this adapter's native
        format), returned as working-kernel source files (name, content). Freezes
        the baseline's build spec as the run's build spec, so every later rebuild
        of the working kernel uses what the baseline declared."""
        ...

    def reference_baseline_files(self) -> list[tuple[str, str]] | None:
        """The task's reference packaged as working-kernel source files (name,
        content) to use as the starting kernel, or None if the task has no
        runnable reference. Freezes the default build spec. Setup uses this when
        [task] baseline = 'reference'."""
        ...

    def build_contract(self) -> str | None:
        """One agent-facing sentence stating the fixed build contract the working
        kernel must honor (entry symbol, calling convention, available
        dependencies), derived from the frozen build spec. None if the adapter
        has no build contract to surface."""
        ...

    def hardware(self) -> str:
        """The accelerator the benchmark actually runs on, detected at runtime."""
        ...

    def language(self) -> str:
        """The working kernel's source language, read from the frozen build spec
        (which the baseline supplied). Seeded onto the tree at setup."""
        ...

    def strip_build_noise(self, text: str) -> str:
        """Drop this adapter's build-system chatter from externally captured
        output (e.g. a profiler's), so its report dominates. Adapters with no
        build noise return text unchanged."""
        ...

    def prewarm(self) -> None:
        """Build the current working kernel to a reusable on-disk artifact."""
        ...

    def build_profilable(self, label: str) -> tuple[Callable, list]:
        """Build the current kernel and materialize one representative workload's
        inputs. Returns (runnable, inputs); call runnable(*inputs)."""
        ...


def get_adapter(state: dict | None = None) -> BenchmarkAdapter:
    """Construct the workspace's selected benchmark adapter from its run-level
    state (the marker setup wrote, or `state` when given): `adapter` selects the
    implementation. Defaults to flashinfer."""
    state = state or read_benchmark_state() or {}
    name = state.get("adapter", "flashinfer")
    if name == "flashinfer":
        from tools.adapters.flashinfer import FlashInferAdapter

        return FlashInferAdapter()
    if name == "sol":
        from tools.adapters.sol import SOLAdapter

        return SOLAdapter()
    raise ValueError(f"unknown benchmark adapter {name!r}")
