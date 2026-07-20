# Kernel Optimizer MCP

An MCP server for correctness-preserving GPU kernel optimization. Claude Code
or Codex edits kernels through a constrained tool surface, benchmarks candidates,
and records measured experiments in a persistent journal.

## Background

Kernel optimization is a non-convex search: promising structural rewrites often
regress before they improve, while long-running agents tend to protect the
current kernel and carry unverified conclusions forward in free-form memory.

This project provides a *git for experiments*. Each logged node contains an
exact source snapshot and full benchmark result; agents can branch, compare, and
restore nodes without losing progress. A structured optimization journal carries
measured results, hypotheses, facts, and hazards across context resets while
keeping qualitative claims distinct from evidence.

Included benchmark adapters:

- [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench) (default)
- [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)

## Quick start

Requirements: Linux, Python 3.11+, a CUDA-capable NVIDIA GPU, a compatible
PyTorch/CUDA toolchain, and either Claude Code or Codex.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/doctor.py
python scripts/setup_workspace.py --skip-baseline-benchmark
```

The repository ships with a complete example under `example-workspace/task/`:
47 paged Multi-head Latent Attention (MLA) decode workloads, their tensor inputs,
and an NVIDIA H100 NVL CUDA scaffold. `config.toml` already points to it; the
FlashInfer comparison target is enabled as a performance yardstick, while
`profile_kernel` is disabled by default.

The flag stages the intentionally empty scaffold without adding its compile
errors to the journal. Setup still benchmarks and records the target; the
agent's first logged kernel becomes `v0`. Omit the flag when the baseline is
runnable and should be measured as `v0_baseline`.

Setup preserves `example-workspace/task/` and `archive/`, but resets the working
source, experiment tree, cache, snapshots, and journal.

### Claude Code

```bash
cd example-workspace
claude
```

Setup writes the project MCP configuration, instructions, and permissions.

### Codex

Run the absolute `codex mcp add ...` command printed by setup, then:

```bash
codex -C /absolute/path/to/example-workspace
```

If the repository or virtual environment moves, replace the registration:

```bash
codex mcp remove kernel-tools
# Run the registration command printed by setup again.
```

## Workflow

- `src/` is the working kernel.
- `benchmark_kernel(scope="smoke")` runs representative workloads.
- `benchmark_kernel(scope="full")` runs the complete suite and is required
  before `log_experiment` accepts the current source.
- `log_experiment` records the exact source, evaluation, and tree position.
- `checkout_experiment` restores a recorded source snapshot.
- `optimization_journal.md` renders the task, build contract, experiment tree,
  current best, hypotheses, facts, and hazards.
- Correctness is mandatory. Performance is same-run speedup when a normalizer is
  available, otherwise absolute latency.

Default tools:

| Tool | Purpose |
| --- | --- |
| `read_source` | Read working or recorded source. |
| `edit_source`, `write_source` | Modify an existing source file. |
| `benchmark_kernel` | Build, validate, and time the kernel. |
| `log_experiment` | Record a full-benchmark result. |
| `read_journal`, `annotate_journal` | Read or update optimization knowledge. |
| `checkout_experiment`, `diff_experiment` | Restore or compare experiments. |
| `profile_kernel` | Profile with Nsight Compute; disabled by default. |

Enable profiling by installing `ncu`, adding `"profile_kernel"` to
`[tools].enabled`, and rerunning setup.

## Configuration

`config.toml` selects the tool surface, adapter, workspace, baseline, and
optional comparison target:

```toml
[tools]
enabled = ["read_source", "edit_source", "write_source", "benchmark_kernel",
           "log_experiment", "read_journal", "annotate_journal",
           "checkout_experiment", "diff_experiment"]

[task]
benchmark = "flashinfer"
workspace_path = "example-workspace"
baseline = "task/baseline/stub.json"

[task.target]
path = "task/target/flashinfer_wrapper_03f7b0.json"
label = "FlashInfer MLA"
description = "FlashInfer paged MLA decode wrapper."
```

Baseline and target paths are relative to the workspace unless absolute. A
baseline may also be `"reference"`, which starts from the task's reference
implementation. Use `--skip-baseline-benchmark` for an incomplete starting
kernel; the comparison target is still measured.

Setup replaces `src/`, `experiments/`, `.state/`, and
`optimization_journal.md`. It does not modify `task/` unless
`--sort-workloads` is passed, and it preserves `archive/`.

## Use another FlashInfer task

Download and inspect the public
[`flashinfer-ai/flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
dataset:

```bash
python scripts/download_data.py --metadata-only
python scripts/seed_task.py --list
python scripts/seed_task.py <definition> --list
```

Replace the bundled task:

```bash
python scripts/seed_task.py <definition> \
  --baseline <solution> \
  --target <solution> \
  --force
```

Omit `--target` if unused. Remove `--metadata-only` when the selected workloads
require saved tensor blobs. To preserve the bundled example, set another
`workspace_path` and pass it through `seed_task.py --workspace`.

Task layout:

```text
<workspace>/task/
├── definition.json
├── workloads.jsonl
├── blob/                 # optional saved inputs
├── baseline/<solution>.json
└── target/<solution>.json
```

Package or extract a FlashInfer Solution:

```bash
python scripts/src_to_solution.py path/to/src path/to/solution.json \
  --definition <definition> --language cuda --entry main.cpp::run

python scripts/solution_to_src.py path/to/solution.json path/to/src
```

## SOL

Install SOL-ExecBench separately and ensure `import sol_execbench` succeeds.
Set `benchmark = "sol"` and provide a SOL-native baseline Solution. Task fixtures
are shared with FlashInfer, but their Solution build specifications differ.

Export a recorded experiment as a SOL problem:

```bash
python scripts/solution_to_sol.py --list
python scripts/solution_to_sol.py <experiment> --out path/to/sol-problem
```

SOL timing uses CUPTI and requires a compatible CUDA driver.

## Extend or verify

The adapter contract is documented in [docs/adapter.md](docs/adapter.md). Add an
implementation under `tools/adapters/` and register it in `tools/_benchmark.py`.

```bash
python -m unittest discover -s tests -v
python -m compileall -q tools scripts
```
