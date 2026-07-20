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

    return json.dumps(_format_for_agent(evaluation, scope), indent=2)


def _format_for_agent(evaluation, scope: str) -> dict:
    """Return the JSON payload shown to the agent, built from the Evaluation
    alone — the adapter already normalized every diagnostic.

    Every outcome uses one shape: the Evaluation with null/empty fields stripped
    per workload. The representative workloads carry the failure diagnostics
    (they span the shape range, so they are the right failure examples); a
    diagnostic shared across workloads — e.g. one compile error — is shown once,
    on the first workload, and dropped from the rest (their outcome already says
    they failed the same way).
    """
    payload = evaluation.model_dump()

    reps: dict[str, dict] = {}
    seen: set[str] = set()  # diagnostic texts already shown by an earlier workload
    for label, result in payload["representative_workloads"].items():
        # Correctness maxima are only meaningful against the gate that produced
        # them: the max-abs and max-rel points are different elements, and a
        # workload passes when no single point exceeds both tolerances. Printed
        # beside the task's atol/rtol on a pass they read as a catastrophic
        # failure. Show them only where they explain one.
        if result.get("outcome") == "PASSED":
            result.pop("correctness", None)
        diagnostic = result.get("diagnostic")
        if diagnostic is not None and diagnostic in seen:
            result.pop("diagnostic")  # already shown by an earlier workload
        elif diagnostic is not None:
            seen.add(diagnostic)
        reps[label] = _prune_nulls(result)
    payload["representative_workloads"] = reps

    # Drop metric fields that carry nothing for this run (latency-only runs have
    # no speedup; all-passed runs have no failure histogram).
    for key in ("geomean_speedup_factor", "geomean_latency_ms"):
        if payload.get(key) is None:
            payload.pop(key, None)
    if not payload["failure_statuses"]:
        payload.pop("failure_statuses")

    if scope == "smoke":
        payload = _mark_smoke(payload)
    return payload


def _mark_smoke(payload: dict) -> dict:
    payload["scope"] = "smoke"
    for src, dst in (
        ("geomean_speedup_factor", "representative_geomean_speedup_factor"),
        ("geomean_latency_ms", "representative_geomean_latency_ms"),
    ):
        value = payload.pop(src, None)
        if value is not None:
            payload[dst] = value
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
