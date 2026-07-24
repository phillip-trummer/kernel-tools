"""Benchmark the kernel in <workspace>/src/: build it, check correctness, measure speed.

Each call asks the adapter to build the current working kernel and run it
against every workload, or against only the representative workloads when
scope="smoke", returning one WorkloadResult per workload; shared harness code
(`aggregate`) folds those leaves into the standardized Evaluation. Only
full-suite evaluations are cached for downstream tools.
"""

import json

from tools.registry import registry
from tools._workspace import (
    BENCHMARK_CACHE_PATH,
    BenchmarkCache,
    read_src_files,
    solution_name_from_src_files,
)
from tools._evaluation import aggregate
from tools._benchmark import BenchmarkUnavailable, get_adapter

SCHEMA = {
    "name": "benchmark_kernel",
    "description": (
        "Benchmark the current working kernel: builds the source, "
        "checks correctness using the task's oracle, and measures performance. "
        "The score is same-run speedup when a normalizer is available, otherwise "
        "absolute latency (lower is better). Use scope='smoke' after most edits; it runs the "
        "representative workloads (small, medium, large, xlarge) and is the "
        "default fast iteration test. Use scope='full' after a smoke-passing "
        "change before comparing it as a candidate, building further from it, "
        "or moving on to another idea; it runs every workload and is slower. "
        "Returns geomean/counts, "
        "per-representative-workload outcome/performance, a normalized diagnostic "
        "for failed workloads, and a breakdown of failure outcomes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["smoke", "full"],
                "default": "smoke",
                "description": (
                    "smoke runs the four representative workloads for fast "
                    "iteration; full runs the complete benchmark suite."
                ),
            },
        },
    },
}


@registry.register(SCHEMA)
def benchmark_kernel(scope: str = "smoke") -> str:
    if scope not in ("full", "smoke"):
        return "Error: scope must be 'full' or 'smoke'."

    # Load the adapter (parses the task fixtures).
    try:
        adapter = get_adapter()
    except Exception as e:
        return f"Error: failed to load task fixtures: {type(e).__name__}: {e}"

    # Build the current working kernel, run it, and fold the leaves into the aggregate.
    try:
        results = adapter.benchmark(scope)
    except BenchmarkUnavailable as e:
        # Not the kernel's fault: report it as-is, and score nothing.
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    evaluation = aggregate(results)

    if evaluation.workload_count == 0:
        return "No workloads ran (no workloads or all failed before evaluation)."

    # Cache full-suite evaluations for downstream tools, keyed by src hash.
    if scope == "full":
        cache = BenchmarkCache.load(BENCHMARK_CACHE_PATH)
        cache.record(solution_name_from_src_files(read_src_files()), evaluation)
        cache.save(BENCHMARK_CACHE_PATH)

    return json.dumps(_format_for_agent(evaluation, scope), separators=(",", ":"))


def _format_for_agent(evaluation, scope: str) -> dict:
    """Project the lossless Evaluation into a compact, agent-facing payload.

    Concrete representative axes and the run-wide tolerance are already durable
    in memory. Repeating them on every benchmark call costs context without
    adding information, so this view carries performance for passes and only the
    tolerance/correctness/diagnostic detail needed to explain failures. The
    cached Evaluation remains unchanged.
    """
    raw = evaluation.model_dump()
    payload: dict = {
        "status": raw["status"],
        "scope": scope,
        "workloads": {
            "passed": raw["passed_workload_count"],
            "total": raw["workload_count"],
        },
    }

    geomean = _prune_nulls(
        {
            "latency_ms": raw.get("geomean_latency_ms"),
            "speedup_factor": raw.get("geomean_speedup_factor"),
        }
    )
    if geomean:
        payload["geomean"] = geomean

    representatives: dict[str, dict] = {}
    failed_labels: list[str] = []
    tolerances: dict[str, dict] = {}
    correctness: dict[str, dict] = {}
    diagnostics: dict[str, list[str]] = {}

    failure_groups = {
        outcome: {"count": count}
        for outcome, count in raw["failure_statuses"].items()
    }

    for label, result in raw["representative_workloads"].items():
        if result["outcome"] == "PASSED":
            representatives[label] = _prune_nulls(
                {
                    "latency_ms": result.get("latency_ms"),
                    "reference_latency_ms": result.get("reference_latency_ms"),
                    "speedup_factor": result.get("speedup_factor"),
                }
            )
            continue

        failed_labels.append(label)
        group = failure_groups.setdefault(result["outcome"], {"count": 0})
        group.setdefault("representatives", []).append(label)

        tolerance = result.get("tolerance")
        if tolerance:
            rendered = _prune_nulls(tolerance)
            if rendered:
                tolerances[label] = rendered

        measured = result.get("correctness")
        if measured:
            rendered = _prune_correctness(measured)
            if rendered:
                correctness[label] = rendered

        diagnostic = result.get("diagnostic")
        if diagnostic:
            diagnostics.setdefault(diagnostic, []).append(label)

    if representatives:
        payload["representatives"] = representatives
    if failure_groups:
        payload["failures"] = failure_groups

    # A common failure tolerance is a run-level fact. Keep the per-representative
    # form only for adapters whose correctness bars genuinely vary by workload.
    if tolerances:
        unique = {json.dumps(value, sort_keys=True) for value in tolerances.values()}
        if len(tolerances) == len(failed_labels) and len(unique) == 1:
            payload["tolerance"] = next(iter(tolerances.values()))
        else:
            payload["tolerance_by_representative"] = tolerances
    if correctness:
        payload["correctness"] = correctness

    # Preserve every distinct diagnostic, but never repeat identical compiler or
    # runtime output for each workload affected by the same failure.
    if len(diagnostics) == 1:
        payload["diagnostic"] = next(iter(diagnostics))
    elif diagnostics:
        payload["diagnostics"] = [
            {"representatives": labels, "message": message}
            for message, labels in diagnostics.items()
        ]
    return payload


def _prune_nulls(d: dict) -> dict:
    """Drop null fields; compact the nested correctness (drop null metrics and
    default-False flags), dropping it entirely when nothing is left."""
    out: dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if k == "correctness" and isinstance(v, dict):
            v = _prune_correctness(v)
            if not v:
                continue
        out[k] = v
    return out


def _prune_correctness(c: dict) -> dict:
    return {
        k: v
        for k, v in c.items()
        if v is not None and not (isinstance(v, bool) and v is False)
    }
