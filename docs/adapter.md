# The benchmark adapter contract

This is the one seam a systems engineer implements to point the whole tool suite
at **their own** kernel and benchmark. Everything above it — the working-kernel
tools, the experiment journal, the plots — sees only neutral results and never a
framework's native types. Implement one adapter (build + run + score their
kernel) and the suite works on top.

Governing principle: **every new adapter is a forcing function to re-question the
neutral results.** If a real benchmark can't map onto the types below cleanly,
the neutral vocabulary is wrong — fix the vocabulary, not the adapter.

Two worked examples, deliberately different shapes. `tools/adapters/flashinfer.py`
(`FlashInferAdapter`, wrapping [flashinfer-bench]) runs in-process and is the
example referenced throughout. `tools/adapters/sol.py` (`SOLAdapter`, wrapping
[SOL-ExecBench]) drives a subprocess evaluator, so it shows the harder case:
failures that arrive as text, and a build you must reach into to profile.

## Where it plugs in

- Protocol: `BenchmarkAdapter` in `tools/_benchmark.py`.
- Factory: `get_adapter()` in the same file selects the adapter by name.
- Selection: `[task] benchmark = "<name>"` in `config.toml`; setup records it (and
  the frozen build spec) into `.state/benchmark.json`, the run-level state file.
- Neutral types: `tools/_evaluation.py`.

The registered tools reach the adapter only through `get_adapter()`, so they stay
adapter-neutral. To add an adapter: write a sibling module, return it from
`get_adapter()` for your name, set `[task] benchmark`.

## The neutral result contract

An adapter produces exactly two kinds of neutral value.

### 1. `TaskSpec` — the task description (seeded onto the tree at setup)

```python
class TaskSpec(BaseModel):
    name: str
    description: str = ""
    op_type: str | None = None
    axes: dict[str, AxisField] = {}      # {const|var|expr, value?, expression?, description?}
    inputs: dict[str, TensorField] = {}  # {shape?, dtype, description?}   (shape None = scalar)
    outputs: dict[str, TensorField] = {}
    reference: str = ""                  # the reference implementation, shown in the journal
    constraints: list = []               # optional descriptive bullets
    tolerance: Tolerance | None = None   # the correctness bar: {max_atol?, max_rtol?, required_matched_ratio?}
```

The journal renders the task from this alone, so the agent can continue from the
tree after a context reset. `axes`/`inputs`/`outputs` have an **explicit** schema
(not an opaque dict) — surface expression axes via `AxisField.expression`.

`TaskSpec.tolerance` is the run's shared bar when one exists. Frameworks that set
it per workload (SOL) report it on each `WorkloadResult`; they additionally map it
onto `TaskSpec` only when every workload agrees, rather than picking one.

### 2. `WorkloadResult` — one per workload (the leaf the adapter returns)

`benchmark(scope)` returns `list[WorkloadResult]`, one per workload run. The
adapter does **not** aggregate; shared code does (below).

```python
class WorkloadResult(BaseModel):
    axes: dict[str, int] = {}            # the concrete shape this workload ran at
    outcome: str                         # normalized OUTCOMES label; "PASSED" on success
    latency_ms: float | None = None            # ground truth, present on a pass
    reference_latency_ms: float | None = None  # same-run normalizer, None if none timed
    speedup_factor: float | None = None        # vs the normalizer, None if none timed
    tolerance: Tolerance | None = None         # this workload's correctness bar
    correctness: Correctness | None = None     # {max_abs_error?, max_rel_error?, has_nan, has_inf}
    diagnostic: str | None = None              # human failure detail; None when passed
```

**Latency is the ground truth.** `speedup_factor`/`reference_latency_ms` are the
*optional* same-run normalizer comparison — populate them only if you time a
reference right beside the candidate (it cancels machine noise). A latency-only
benchmark leaves them `None`; the harness then ranks and renders by absolute
latency (lower = better) instead of speedup. There is no metric flag to set — it
is derived.

**`outcome` is normalized and validated.** Map your framework's native status onto the neutral
taxonomy in `_evaluation.py`:

```
PASSED, COMPILE_ERROR, RUNTIME_ERROR, INCORRECT_NUMERICAL, INCORRECT_SHAPE,
INCORRECT_DTYPE, TIMEOUT, INVALID_REFERENCE, REWARD_HACK, OTHER
```

Use `normalize_outcome(native_label)` — anything unrecognized becomes `OTHER`, so
a new framework's exotic status never crashes ranking. (The taxonomy is modeled
on SOL-ExecBench's `EvaluationStatus`; flashinfer emits a subset.)

### When the benchmark itself is broken, raise

Every `outcome` above answers *"what is wrong with the kernel?"*. None answers
*"the benchmark could not evaluate it"* — a missing timer, an unusable driver, a
staging failure. Those are not the kernel's fault, and reporting them as failing
leaves tells the agent to fix a kernel that may be perfect.

```python
raise BenchmarkUnavailable("the benchmark cannot measure the kernel: <reason>")
```

The absence of a verdict is not a verdict: nothing is scored, nothing is logged,
and the message (in agent vocabulary) reaches the agent. Frameworks blur this —
SOL reports a dead timer with the same `RUNTIME_ERROR` status as a kernel that
crashes, and only the log message separates them — so classifying it is the
adapter's job.

### `diagnostic` is agent-facing text

The agent reads it verbatim, so it obeys the agent-vocabulary rule: **no filesystem
paths, no internal identifiers.** A raw compiler error names the build directory it
compiled from (`/home/…/cache/torch/torch_candidate_7c99…/kernel.cu(586)`), which is
neither — the agent knows its sources by bare filename.

If your framework builds through torch's `cpp_extension`, reuse
`tools/adapters/_torch_build_log.py`: `denoise(log, compile_error=...)` drops ninja's
step headers, the compiler command echoed under `FAILED:`, the traceback torch wraps
the error in, and the build path. `strip_build_noise(text)` does the same for captured
profiler output. Locate the real diagnostic first — nvcc writes it to **stdout**;
stderr often holds only the Python wrapper.

Do **not** share the correctness sentence. Report each error against the bound that
judged it *and* name the rule, because the rules differ: flashinfer fails a point only
when it exceeds both atol and rtol, while SOL uses an allclose bound
(`atol + rtol*|ref|`) plus a matched-element ratio. A borrowed sentence is a false one.
Frameworks may also emit metrics with no message at all (SOL does, on numerical
failure), so an empty log is not an empty diagnostic. Return `None`, never `""`.

### Aggregation is shared, not per-adapter

```python
aggregate(list[WorkloadResult]) -> Evaluation
```

One harness function folds the leaves into the stored `Evaluation` (geomean,
representative pick, failure histogram), so the scoring rule is identical across
every adapter. The adapter never constructs an `Evaluation`.

```python
class Evaluation(BaseModel):
    status: str                          # "ALL_PASSED" | "FAILED"
    geomean_latency_ms: float | None     # always the ground truth (when all passed)
    geomean_speedup_factor: float | None # present iff a same-run reference was timed
    workload_count: int
    passed_workload_count: int
    representative_workloads: dict[str, WorkloadResult]  # small/medium/large/xlarge
    failure_statuses: dict[str, int]     # {outcome: count} over failed workloads
```

Geomeans are non-null only when **every** workload passed (a geomean over a
subset would misrank a candidate that failed the hard shapes).

## Design model: correctness oracle vs. speed baselines

Two orthogonal roles keep the types clean:

- **Correctness oracle** — produces the right output; candidates are checked by
  comparing outputs. One per run: a reference impl, golden tensors, differential
  vs. the starting kernel, or none ("just runs / no NaN").
- **Speed baselines** — what you compare latency against: an optional **live
  normalizer** (timed same-run → the `speedup_factor`) and **pinned anchors**
  (the target, the v0 baseline — comparisons the *journal* derives from stored
  numbers, not fields on `Evaluation`).

Consequences: `Evaluation` is **self-contained** (correctness verdict + absolute
latency + optional same-run speedup). "vs target" / "vs v0" are tree-derived
ratios — same-run (via speedup) when a normalizer exists, else a noisier
cross-run latency ratio.

The common case (flashinfer today) is reference = oracle + live normalizer,
target/v0 = anchors. The types support latency-only and oracle-less setups; build
the mode you need, keep the seam.

## The full protocol

Beyond `benchmark`/`benchmark_target` and `task_spec`, an adapter implements:

| method | returns | purpose |
|---|---|---|
| `benchmark(scope)` | `list[WorkloadResult]` | build `src/`, run all (`"full"`) or the 4 representatives (`"smoke"`) |
| `benchmark_target(path)` | `(list[WorkloadResult], target_id)` | benchmark the comparison target in its own build spec |
| `task_spec()` | `TaskSpec` | neutral task description seeded at setup |
| `representative_axes()` | `dict[label, dict[axis, int]]` | concrete shapes for small/medium/large/xlarge |
| `sort_workloads_fixture()` | `bool` | order the fixture cheapest→priciest (setup-only, behind `--sort-workloads`) |
| `baseline_files(path)` | `list[(name, content)]` | the starting kernel's sources; **freezes its build spec** |
| `reference_baseline_files()` | `list[(name, content)] \| None` | package the reference as the starting kernel (for `baseline = "reference"`) |
| `build_contract()` | `str \| None` | one agent-facing sentence: entry symbol, calling convention, deps |
| `hardware()` | `str` | the accelerator, detected at runtime (never configured) |
| `language()` | `str` | source language, read from the frozen build spec |
| `strip_build_noise(text)` | `str` | drop build-system chatter from captured profiler output |
| `prewarm()` / `build_profilable(label)` | — / `(runnable, inputs)` | for `profile_kernel` (ncu): compile once, then hand back the built kernel and one workload's inputs |

`build_profilable` must return a **timing-free** invocation of the kernel. A
profiler and a CUPTI-based timer cannot coexist in one process — the second
subscriber gets `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` and then degrades
*silently*, so a benchmark that times with CUPTI measures nothing while still
exiting 0. Do not profile by running your framework's evaluator; load the compiled
artifact and call it. Even a subprocess-only framework exposes one — SOL's packager
leaves a `benchmark_kernel.so`.

Both are optional. Skip them, and leave your adapter out of `TOOL_ADAPTERS`
(`scripts/setup_workspace.py`), which lists the adapters each restricted tool
supports; setup then aborts if a run exposes `profile_kernel` anyway.

Representative selection is configured as four `name`/`uuid` pairs under
`[[task.representative_workloads]]`. Setup verifies the UUIDs against the task
fixture and freezes the mapping in run state; adapters provide their native UUID
accessor to `tools/_workloads.py`. Positional selection remains a fallback for
non-setup/test workspaces. `src/` is a **flat** set of editable source files —
setup rejects nested source paths.

## Run-level state (`.state/benchmark.json`)

Run constants the agent never needs live here (not on every `Evaluation`, not in
agent-facing `tree.json`): the selected `adapter`, named
`representative_workloads`, the frozen `build_spec`, and — as adapters need them
— `environment` (lib versions), `timing_methodology`, and the
oracle/normalizer/anchors config. Setup writes the adapter and representatives;
the adapter's `baseline_files()` merges in `build_spec` via
`write_build_spec()`. This draws the product boundary: `tree.json` =
agent-facing, `.state/benchmark.json` = internal.

## Checklist

1. Implement `BenchmarkAdapter` in a sibling module; register it in `get_adapter()`; set `[task] benchmark`.
2. `benchmark()` returns one `WorkloadResult` leaf per workload — normalized `outcome`, absolute `latency_ms`, and (if timed) `speedup_factor`. No aggregation.
3. Map native statuses with `normalize_outcome`; populate per-workload `tolerance` and structured `correctness` where cheap.
4. Write `diagnostic` in agent vocabulary — strip build paths, name your own correctness gate, `None` not `""`.
5. Raise `BenchmarkUnavailable` when the benchmark — not the kernel — is the thing that failed.
6. Preserve `candidate_<hash16>` source-snapshot identity (used for dedup + cache keys).
7. Resolve configured representative UUIDs for smoke runs, `representative_axes()`, and profiling, and mark their neutral leaves for aggregation.
8. Run `scripts/setup_workspace.py`, then verify `benchmark_kernel(scope="smoke")`, `benchmark_kernel(scope="full")`, `profile_kernel` on one representative, and `log_experiment` after the full run.

Each place a real benchmark can't map onto these types cleanly is a genuine find —
loop back and fix the neutral vocabulary.

[flashinfer-bench]: https://bench.flashinfer.ai/
[SOL-ExecBench]: https://github.com/NVIDIA/SOL-ExecBench
