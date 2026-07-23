"""The neutral contracts every benchmark adapter emits and the tree stores.

One shape across benchmark frameworks (flashinfer today, SOL/kernelbench later):
`TaskSpec` describes the task (seeded onto the tree at setup), and `Evaluation`
reports a benchmark run (stored per tree node). Each adapter converts its native
fixtures/results into these, so nothing above the adapter sees native types.

The split that keeps these clean: an adapter returns one `WorkloadResult` per
workload (the leaf), and one shared `aggregate()` folds the leaves into the
stored `Evaluation` — so the scoring rule (geomean, representative pick, failure
histogram) is the harness's, identical across adapters, not re-derived per
adapter.
"""
from __future__ import annotations

import math
from typing import Literal, cast

from pydantic import BaseModel, Field, model_validator

from tools._workloads import representative_items


# --- Normalized per-workload outcome taxonomy ---
# One neutral vocabulary of per-workload outcomes across benchmark frameworks;
# each adapter maps its native status onto these. Modeled on SOL-ExecBench's
# EvaluationStatus (the superset — flashinfer emits a subset, so it maps for
# free), plus OTHER so an unmapped native status never crashes the harness.
Outcome = Literal[
    "PASSED",
    "COMPILE_ERROR",
    "RUNTIME_ERROR",
    "INCORRECT_NUMERICAL",
    "INCORRECT_SHAPE",
    "INCORRECT_DTYPE",
    "TIMEOUT",
    "INVALID_REFERENCE",
    "REWARD_HACK",
    "OTHER",
]
OUTCOMES: tuple[Outcome, ...] = (
    "PASSED",
    "COMPILE_ERROR",
    "RUNTIME_ERROR",
    "INCORRECT_NUMERICAL",
    "INCORRECT_SHAPE",
    "INCORRECT_DTYPE",
    "TIMEOUT",
    "INVALID_REFERENCE",
    "REWARD_HACK",
    "OTHER",
)


def normalize_outcome(native: str) -> Outcome:
    """Map an adapter's native per-workload status onto the neutral taxonomy.
    Unknown labels collapse to OTHER so a new framework never crashes ranking."""
    return cast(Outcome, native) if native in OUTCOMES else "OTHER"


class TensorField(BaseModel):
    """One input/output tensor's signature. The previously-opaque `inputs` /
    `outputs` dict schema, made explicit (shared flashinfer/SOL lineage)."""

    shape: list[str] | None = None  # None = scalar argument
    dtype: str = "?"
    description: str | None = None


class AxisField(BaseModel):
    """One workload axis: a fixed constant, a free variable, or an expression
    derived from other axes."""

    kind: Literal["const", "var", "expr"] = "var"
    value: int | None = None  # const
    expression: str | None = None  # expr
    description: str | None = None


class Tolerance(BaseModel):
    """The numerical bar a candidate must clear to count as correct. Surfaced on
    the task so the agent knows its slack before it fails."""

    max_atol: float | None = None
    max_rtol: float | None = None
    required_matched_ratio: float | None = None


class Correctness(BaseModel):
    """Structured correctness measurement for one workload (vs. the flattened
    diagnostic string). Numbers an adapter can always report; a failing bound is
    then legible without parsing prose."""

    max_abs_error: float | None = None
    max_rel_error: float | None = None
    has_nan: bool = False
    has_inf: bool = False


class TaskSpec(BaseModel):
    """Neutral description of the task, seeded onto the tree at setup. The
    journal header and the agent's prompt read it from the tree — never from a
    native fixture — so the agent can continue from the tree alone. Each adapter
    maps its own task into these fields."""

    name: str
    description: str = ""
    op_type: str | None = None
    axes: dict[str, AxisField] = Field(default_factory=dict)
    inputs: dict[str, TensorField] = Field(default_factory=dict)
    outputs: dict[str, TensorField] = Field(default_factory=dict)
    reference: str = ""
    constraints: list = Field(default_factory=list)
    tolerance: Tolerance | None = None  # the run's correctness bar, if known


class WorkloadResult(BaseModel):
    """One workload's result — the leaf an adapter returns. Latency is the ground
    truth (always present on a pass); `speedup_factor` / `reference_latency_ms`
    are the optional same-run normalizer comparison, None when nothing is timed
    beside the candidate (a latency-only benchmark)."""

    axes: dict[str, int] = Field(default_factory=dict)
    outcome: Outcome
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    speedup_factor: float | None = None
    tolerance: Tolerance | None = None
    correctness: Correctness | None = None
    diagnostic: str | None = None  # human failure detail; None when passed
    # Adapter-set run metadata used only while aggregating. The containing
    # dictionary already carries the representative name, so do not serialize it.
    representative_name: str | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _validate_metrics(self) -> "WorkloadResult":
        for name in ("latency_ms", "reference_latency_ms", "speedup_factor"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.outcome == "PASSED" and self.latency_ms is None:
            raise ValueError("a PASSED workload must report latency_ms")
        if (self.reference_latency_ms is None) != (self.speedup_factor is None):
            raise ValueError(
                "reference_latency_ms and speedup_factor must be present together"
            )
        return self

    @property
    def passed(self) -> bool:
        return self.outcome == "PASSED"


class Evaluation(BaseModel):
    """The aggregate one tree node stores — built by `aggregate()`, never by an
    adapter. `geomean_latency_ms` is the always-present ground truth;
    `geomean_speedup_factor` is present only when a reference was timed same-run.
    Ranking derives its metric: speedup if present, else latency (lower better) —
    no stored primary-metric flag."""

    status: Literal["ALL_PASSED", "FAILED"]
    geomean_latency_ms: float | None = None
    geomean_speedup_factor: float | None = None
    workload_count: int = 0
    passed_workload_count: int = 0
    representative_workloads: dict[str, WorkloadResult] = Field(default_factory=dict)
    failure_statuses: dict[Outcome, int] = Field(default_factory=dict)


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def aggregate(results: list[WorkloadResult]) -> Evaluation:
    """Fold per-workload leaves into the stored aggregate. The one place the
    scoring rule lives, so every adapter (flashinfer, SOL, …) is scored
    identically: geomean over passed workloads (only when all passed), the four
    representatives with detail, and a failure histogram over the taxonomy.

    Geomeans are reported only when every workload passed — a geomean over a
    subset would misrank a candidate that failed the hard shapes. Speedup geomean
    additionally requires every passed workload to carry a same-run speedup."""
    passed = [r for r in results if r.passed]
    all_passed = bool(results) and len(passed) == len(results)

    geomean_latency = geomean_speedup = None
    if all_passed:
        latencies = [r.latency_ms for r in passed if r.latency_ms is not None]
        if len(latencies) == len(passed):
            geomean_latency = round(_geomean(latencies), 6)
        speedups = [r.speedup_factor for r in passed if r.speedup_factor is not None]
        if len(speedups) == len(passed):
            geomean_speedup = round(_geomean(speedups), 4)

    representatives = {label: r for label, r in representative_items(results)}

    failures: dict[str, int] = {}
    for r in results:
        if not r.passed:
            failures[r.outcome] = failures.get(r.outcome, 0) + 1

    return Evaluation(
        status="ALL_PASSED" if all_passed else "FAILED",
        geomean_latency_ms=geomean_latency,
        geomean_speedup_factor=geomean_speedup,
        workload_count=len(results),
        passed_workload_count=len(passed),
        representative_workloads=representatives,
        failure_statuses=failures,
    )
