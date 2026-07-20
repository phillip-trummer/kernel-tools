"""flashinfer-bench benchmark adapter.

The only module that imports `flashinfer_bench.*`. Implements the
BenchmarkAdapter interface (see _benchmark.py) for flashinfer: `FlashInferAdapter`
owns the task fixtures (definition + workloads) so the harness never handles
native flashinfer types (`Definition`, `Workload`, `Trace`, `Solution`,
`EvaluationStatus`) — they stay inside this file, and only neutral results
(WorkloadResult leaves, TaskSpec, runnable+inputs) cross the boundary. Another
benchmark integration is scoped to a sibling adapter.

flashinfer imports are deferred to function bodies so that `import tools`
does not pull torch + flashinfer eagerly.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Callable

from tools.adapters._torch_build_log import denoise
from tools.adapters._torch_build_log import strip_build_noise as _strip_build_noise
from tools._workspace import (
    ARCHIVE_DIR,
    TASK_DIR,
    read_build_spec,
    read_src_files,
    solution_name_from_src_files,
    write_build_spec,
)
from tools._workloads import (
    representative_item_for_label,
    select_representative_workloads,
)
from tools._evaluation import (
    AxisField,
    Correctness,
    TaskSpec,
    TensorField,
    Tolerance,
    WorkloadResult,
    normalize_outcome,
)


# --- Build-spec defaults (Solution construction) ---
# flashinfer entry_point is "<file>::<symbol>". The host entry file follows the
# kernel's language: cuda compiles main.cpp (kernel.cu/.h are included); python
# and triton import main.py. The symbol (run) is the same across languages.
ENTRY_FILE_BY_LANGUAGE = {"cuda": "main.cpp", "python": "main.py", "triton": "main.py"}
ENTRY_SYMBOL = "run"
AUTHOR = "agent"


class FlashInferAdapter:
    """Loads the task fixtures once, then exposes only neutral results — native
    flashinfer types never leave this class. The kernel's source language comes
    from the baseline's frozen build spec, not config."""

    def __init__(self) -> None:
        self.definition, self.workload_traces = _load_task()

    def benchmark(self, scope: str) -> list[WorkloadResult]:
        """Build the current src/ kernel, run it against the workloads (all, or
        the representative four when scope='smoke'), archive the run, and return
        one neutral WorkloadResult per workload. Shared harness code aggregates
        the leaves — this adapter never scores."""
        solution = self._build_solution()
        _append_solution_to_archive(solution)
        workloads = self.workload_traces
        if scope == "smoke":
            workloads, _ = select_representative_workloads(workloads)
        traces = _run_benchmark(self.definition, solution, workloads)
        _append_traces_to_archive(traces)
        return _workload_results(traces)

    def benchmark_target(self, target_path: Path) -> tuple[list[WorkloadResult], str]:
        """Benchmark the target Solution at target_path (a .json file) against the
        full workload suite; return (leaves, target_id). It is benchmarked with
        its OWN build spec, so it may be a different language/runtime than the
        agent's kernel."""
        solution = self._load_solution_file(target_path)
        _append_solution_to_archive(solution)
        traces = _run_benchmark(self.definition, solution, self.workload_traces)
        _append_traces_to_archive(traces)
        return _workload_results(traces), solution.name

    def baseline_files(self, baseline_path: Path) -> list[tuple[str, str]]:
        """Load the baseline Solution at baseline_path (a .json file), freeze its
        build spec as the run's build spec, and return its sources as
        (name, content) for staging into the working kernel."""
        solution = self._load_solution_file(baseline_path)
        write_build_spec(solution.spec.model_dump(mode="json"))
        return [(s.path, s.content) for s in solution.sources]

    def _load_solution_file(self, path: Path):
        """Load the flashinfer Solution at `path` and validate it targets this
        task."""
        from flashinfer_bench.data import Solution

        try:
            solution = Solution.model_validate_json(path.read_text())
        except Exception as e:
            raise ValueError(f"{path.name} is not a valid solution file: {e}")
        if solution.definition != self.definition.name:
            raise ValueError(
                f"solution targets definition {solution.definition!r}, but "
                f"the task definition is {self.definition.name!r}"
            )
        # The flashinfer-bench solution corpus omits destination_passing_style
        # and is uniformly value-returning (so is the reference build), but the
        # BuildSpec default is True — which would wrongly demand out-params and
        # fail signature validation. Honor an explicit flag; default unset to
        # value-returning, matching the corpus and reference convention.
        #
        # An omitted flag is not wrong here, but it is ambiguous: the same file
        # means out-params under flashinfer-bench's and SOL's defaults, where the
        # entry point's return value is discarded. Say so once, at load.
        if "destination_passing_style" not in solution.spec.model_fields_set:
            solution.spec.destination_passing_style = False
            print(
                f"[warn] {path.name} does not declare destination_passing_style; "
                "assuming a value-returning entry point. Benchmark frameworks "
                "default it to out-parameters, so declare it explicitly to keep "
                "the solution portable.",
                file=sys.stderr,
            )
        return solution

    def representative_axes(self) -> dict[str, dict[str, int]]:
        """Map each representative label (small/medium/large/xlarge) to its
        workload's `axes` — the concrete axis-name -> integer shape declared by
        the task fixtures. Labels that collapse onto the same workload are
        deduped, so the result may have fewer than four entries."""
        selected, labels = select_representative_workloads(self.workload_traces)
        return {label: dict(t.workload.axes) for label, t in zip(labels, selected)}

    def sort_workloads_fixture(self) -> bool:
        """Order task/workloads.jsonl smallest-to-largest by total work (the
        product of each workload's variable axes) so representative selection,
        which trusts file order, is monotonic. Rewrites the fixture and reloads
        the in-memory workloads only when the order actually changes; returns
        whether it did."""
        changed = _sort_workloads_file(TASK_DIR / "workloads.jsonl")
        if changed:
            self.definition, self.workload_traces = _load_task()
        return changed

    def hardware(self) -> str:
        """The CUDA device this process runs on."""
        import torch

        return torch.cuda.get_device_name(0)

    def language(self) -> str:
        """The working kernel's source language, read from the frozen build spec.
        Empty string if no spec is frozen (a non-setup workspace)."""
        spec = read_build_spec()
        return spec["language"] if spec else ""

    def reference_baseline_files(self) -> list[tuple[str, str]] | None:
        """Package the task's reference (a torch program defining run()) as the
        working kernel, for an optimize-the-reference baseline, and freeze its
        python build spec. The reference is a torch program, so this is always a
        python baseline; returns None if the task has no reference."""
        import torch
        from flashinfer_bench.data import BuildSpec, SupportedBindings

        reference = (getattr(self.definition, "reference", "") or "").strip()
        if not reference:
            return None
        content = reference.replace("\r\n", "\n").replace("\r", "\n") + "\n"
        entry_file = ENTRY_FILE_BY_LANGUAGE["python"]
        spec = BuildSpec(
            language="python",
            target_hardware=[torch.cuda.get_device_name(0).replace(" ", "_")],
            entry_point=f"{entry_file}::{ENTRY_SYMBOL}",
            binding=SupportedBindings.TORCH,
            destination_passing_style=False,
        )
        write_build_spec(spec.model_dump(mode="json"))
        return [(entry_file, content)]

    def build_contract(self) -> str | None:
        """One agent-facing sentence stating the fixed build contract the working
        kernel must honor — entry symbol, calling convention (value-return vs.
        out-params), and any available build dependencies — derived from the
        frozen build spec. None if no spec is frozen."""
        spec = read_build_spec()
        return _build_contract_text(spec, self.definition) if spec else None

    def task_spec(self) -> TaskSpec:
        """Map the flashinfer Definition into the neutral TaskSpec, including the
        run's correctness bar (flashinfer's tolerance is a single global config,
        so it is a run-level bar, the same for every workload)."""
        d = self.definition.model_dump(mode="json")
        cfg = _bench_config()
        return TaskSpec(
            name=d["name"],
            description=d.get("description") or "",
            op_type=d.get("op_type"),
            axes={k: _axis_field(v) for k, v in (d.get("axes") or {}).items()},
            inputs={k: _tensor_field(v) for k, v in (d.get("inputs") or {}).items()},
            outputs={k: _tensor_field(v) for k, v in (d.get("outputs") or {}).items()},
            reference=d.get("reference") or "",
            constraints=d.get("constraints") or [],
            tolerance=Tolerance(
                max_atol=cfg.atol,
                max_rtol=cfg.rtol,
                required_matched_ratio=getattr(cfg, "required_matched_ratio", None),
            ),
        )

    def strip_build_noise(self, text: str) -> str:
        """Drop this backend's build-system chatter (ninja steps, torch
        cpp_extension, builder banners) from externally captured output — e.g.
        ncu's — so a profiler's report dominates. Other lines pass through."""
        return _strip_build_noise(text)

    def prewarm(self) -> None:
        """Compile the current src/ kernel to its on-disk artifact so a separate
        ncu child process reuses it instead of recompiling under instrumentation."""
        _prewarm_build(self.definition, self._build_solution())

    def build_profilable(self, label: str) -> tuple[Callable, list]:
        """Build the current src/ kernel and materialize one representative
        workload's inputs. Returns (runnable, inputs); call runnable(*inputs)."""
        workload = representative_item_for_label(self.workload_traces, label).workload
        runnable = _build_runnable(self.definition, self._build_solution())
        inputs = _materialize_inputs(self.definition, workload)
        return runnable, inputs

    def _build_solution(self):
        return _build_solution_from_src(self.definition.name)


def _build_contract_text(spec: dict, definition) -> str:
    """Compose the build-contract sentence from a frozen build spec dict: the
    entry symbol, value-return vs. out-param calling convention, and any
    available build dependencies."""
    entry = spec.get("entry_point", "")
    symbol = entry.split("::", 1)[1] if "::" in entry else entry
    outputs = list(getattr(definition, "outputs", {}) or {})
    out_str = ", ".join(outputs) if outputs else "its outputs"
    if spec.get("destination_passing_style", True):
        convention = f"writes {out_str} into caller-provided out-parameters"
    else:
        convention = f"returns ({out_str}) by value"
    parts = [f"The working kernel must define `{symbol}`, which {convention}."]
    deps = spec.get("dependencies") or []
    if deps:
        parts.append(f"Available build dependencies: {', '.join(deps)}.")
    return " ".join(parts)


# --- Task fixtures ---
def _load_task() -> tuple[object, list]:
    """Parse task/definition.json + task/workloads.jsonl from the workspace cwd."""
    from flashinfer_bench.data import Definition, Trace

    def_path = TASK_DIR / "definition.json"
    wl_path = TASK_DIR / "workloads.jsonl"

    definition = Definition.model_validate_json(def_path.read_text())
    workloads = [
        Trace.model_validate_json(line)
        for line in wl_path.read_text().splitlines()
        if line.strip()
    ]
    return definition, workloads


def _workload_size(record: dict) -> int:
    """Total work for a workload record: the product of its variable axes (the
    axis-name -> int map under `workload.axes`). The ordering key for sorting the
    fixture smallest-to-largest."""
    return math.prod((record["workload"]["axes"] or {}).values())


def _sort_workloads_file(path: Path) -> bool:
    """Rewrite the workloads.jsonl at `path` in smallest-to-largest order by
    `_workload_size`, stably. Returns True only if the order changed (so a
    already-sorted fixture is left byte-for-byte untouched)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ordered = sorted(records, key=_workload_size)
    if ordered == records:
        return False
    path.write_text("\n".join(json.dumps(r) for r in ordered) + "\n")
    return True


# --- Solution construction ---
def _build_solution_from_src(definition_name: str):
    """Build a flashinfer Solution from <cwd>/src/ against the run's frozen build
    spec, so every rebuild reuses whatever the baseline declared. Raises if src/
    is empty or no build spec is frozen (setup_workspace.py freezes it)."""
    files = read_src_files()
    if not files:
        raise RuntimeError("the working kernel has no source files")
    spec = read_build_spec()
    if spec is None:
        raise RuntimeError("the working kernel has no build contract")
    return _solution_with_spec(definition_name, files, spec)


def _solution_with_spec(definition_name: str, files: list[tuple[str, str]], spec: dict):
    """Build a flashinfer Solution from (name, content) source pairs and a frozen
    build spec dict (the baseline's own spec, governing the whole run)."""
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile

    return Solution(
        name=solution_name_from_src_files(files),
        definition=definition_name,
        author=AUTHOR,
        spec=BuildSpec.model_validate(spec),
        sources=[SourceFile(path=name, content=content or "\n") for name, content in files],
    )


# --- Benchmark execution ---
def _bench_config():
    """The single benchmark config — its atol/rtol are the correctness gate the
    run scores against and the diagnostic reports, so the two can't drift."""
    from flashinfer_bench.bench import BenchmarkConfig
    return BenchmarkConfig()


def _run_benchmark(definition, solution, workloads: list) -> list:
    """Run flashinfer-bench on `solution` against `workloads`. Returns the
    list of resulting Traces. The TraceSet is rooted at TASK_DIR so relative
    safetensors paths in workloads.jsonl (e.g. "./blob/...") resolve correctly.

    `workloads` is a list of Trace objects (definition+workload populated,
    solution/evaluation null) — this matches the flashinfer schema, where a
    standalone Workload is stored as a Trace and `TraceSet.workloads` is typed
    as `Dict[str, List[Trace]]`."""
    from flashinfer_bench.bench import Benchmark
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet(
        root=TASK_DIR,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
    )
    result_set = Benchmark(trace_set, _bench_config()).run_all(dump_traces=False)
    return result_set.traces.get(definition.name, [])


def _build_runnable(definition, solution):
    """Compile a Solution into a runnable kernel via flashinfer's BuilderRegistry."""
    from flashinfer_bench.compile import BuilderRegistry
    return BuilderRegistry.get_instance().build(definition, solution)


def _prewarm_build(definition, solution) -> None:
    """Compile `solution` into its on-disk build directory so a separate process
    (the profile_kernel ncu child) reuses the artifact instead of recompiling.
    The build dir is keyed by content hash, so the child derives the same path.

    Builds via TorchBuilder directly, not _build_runnable: the registry's
    in-process cache would short-circuit without guaranteeing the on-disk
    artifact exists (e.g. after a benchmark run cleaned the dir)."""
    from flashinfer_bench.compile.builders import TorchBuilder
    TorchBuilder().build(definition, solution)


def _materialize_inputs(definition, workload, device: str = "cuda:0") -> list:
    """Materialize input values for a workload, loading safetensors if needed.

    Returns a positional list in definition.inputs order, matching flashinfer's
    own evaluators — call the runnable as ``runnable(*inputs)``."""
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    safe_tensors = (
        load_safetensors(definition, workload, TASK_DIR)
        if any(d.type == "safetensors" for d in workload.inputs.values())
        else None
    )
    return gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)


# --- Archive I/O (archive/solutions.jsonl + archive/traces.jsonl) ---
def _append_solution_to_archive(solution) -> None:
    """Append `solution` to archive/solutions.jsonl, deduped by name."""
    sol_path = _archive_dir() / "solutions.jsonl"
    seen: set[str] = set()
    if sol_path.exists():
        for line in sol_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                continue
    if solution.name in seen:
        return
    with sol_path.open("a") as f:
        f.write(solution.model_dump_json() + "\n")


def _append_traces_to_archive(traces) -> None:
    """Append `traces` to archive/traces.jsonl. No-op if traces is empty."""
    if not traces:
        return
    with (_archive_dir() / "traces.jsonl").open("a") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")


def _archive_dir() -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR


# --- Neutral leaf mapping (one WorkloadResult per trace) ---
def _workload_results(traces) -> list[WorkloadResult]:
    """Map each flashinfer Trace to a neutral WorkloadResult leaf. Aggregation
    (geomean, representative pick, failure histogram) is shared harness code, not
    this adapter's job."""
    from flashinfer_bench.data import EvaluationStatus

    return [_workload_result(t, EvaluationStatus) for t in traces]


def _workload_result(trace, EvaluationStatus) -> WorkloadResult:
    passed = trace.evaluation.status == EvaluationStatus.PASSED
    perf = trace.evaluation.performance if passed else None
    return WorkloadResult(
        axes=dict(trace.workload.axes),
        outcome=normalize_outcome(trace.evaluation.status.value),
        latency_ms=round(perf.latency_ms, 6) if perf else None,
        reference_latency_ms=(
            round(perf.reference_latency_ms, 6)
            if perf and perf.reference_latency_ms is not None
            else None
        ),
        speedup_factor=(
            round(perf.speedup_factor, 4)
            if perf and perf.speedup_factor is not None
            else None
        ),
        tolerance=_tolerance(),
        correctness=_correctness(trace),
        diagnostic=None if passed else _workload_diagnostic(trace),
    )


def _correctness(trace) -> Correctness | None:
    """Structured correctness from a trace's native record. Non-finite error
    metrics become flags (has_nan / has_inf) with the numeric field left null, so
    the leaf stays JSON-clean."""
    c = getattr(trace.evaluation, "correctness", None)
    if c is None:
        return None
    abs_err, rel_err = c.max_absolute_error, c.max_relative_error
    has_inf = any(isinstance(v, float) and math.isinf(v) for v in (abs_err, rel_err))
    has_nan = any(isinstance(v, float) and math.isnan(v) for v in (abs_err, rel_err))
    return Correctness(
        max_abs_error=_finite_metric(abs_err),
        max_rel_error=_finite_metric(rel_err),
        has_nan=has_nan,
        has_inf=has_inf,
    )


def _tolerance() -> Tolerance:
    cfg = _bench_config()
    return Tolerance(
        max_atol=cfg.atol,
        max_rtol=cfg.rtol,
        required_matched_ratio=getattr(cfg, "required_matched_ratio", None),
    )


def _finite_metric(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return _metric(value)


def _axis_field(a: dict) -> AxisField:
    """Map a flashinfer axis dump (AxisConst / AxisVar / AxisExpr) to AxisField."""
    return AxisField(
        kind=a.get("type", "var"),
        value=a.get("value"),
        expression=a.get("expression"),
        description=a.get("description"),
    )


def _tensor_field(t: dict) -> TensorField:
    """Map a flashinfer TensorSpec dump to the neutral TensorField."""
    return TensorField(
        shape=t.get("shape"),
        dtype=t.get("dtype", "?"),
        description=t.get("description"),
    )


# --- Diagnostics (backend-specific log/correctness normalization) ---
# All of this turns flashinfer's native build logs and correctness records into
# the normalized strings the contract carries, so the harness never parses them.
# The toolchain-level cleanup (ninja/cpp_extension noise, build paths) is shared;
# only what is flashinfer's own lives here.
# Normalize flashinfer's repeated-build wrapper.
_SKIPPED_RE = re.compile(r"^Solution skipped after \d+ failures?\. Last error:\s*")


def _workload_diagnostic(trace) -> str | None:
    # Compose a correctness summary and a filtered log tail into one note.
    parts = []
    correctness = _correctness_text(getattr(trace.evaluation, "correctness", None))
    if correctness:
        parts.append(correctness)
    log = getattr(trace.evaluation, "log", "") or ""
    tail = _diagnostic_tail(log, trace.evaluation.status.value)
    if tail:
        parts.append(tail)
    return "\n".join(parts) or None


def _diagnostic_tail(text: str, status: str) -> str:
    """Unwrap flashinfer's repeated-build wrapper, then hand the log to the shared
    toolchain cleanup."""
    unwrapped = "\n".join(_SKIPPED_RE.sub("", line) for line in text.strip().splitlines())
    return denoise(unwrapped, compile_error=status == "COMPILE_ERROR")


def _correctness_text(correctness) -> str | None:
    if correctness is None:
        return None
    rel = correctness.max_relative_error
    abs_err = correctness.max_absolute_error
    # Non-finite metrics are sentinels for non-finite output.
    for value in (abs_err, rel):
        if isinstance(value, float) and math.isinf(value):
            return "output contains Inf"
        if isinstance(value, float) and math.isnan(value):
            return "output contains NaN"
    # Show each error against its tolerance and the gate: a point fails only
    # when it exceeds both, so the agent can tell a real bug (both exceeded)
    # from benign precision (one within tolerance).
    cfg = _bench_config()
    return (
        f"max relative error {_metric(rel)} (rtol {_metric(cfg.rtol)}), "
        f"max absolute error {_metric(abs_err)} (atol {_metric(cfg.atol)}); "
        f"a point fails only when it exceeds both tolerances"
    )


def _metric(value):
    return round(value, 6) if isinstance(value, float) else value
