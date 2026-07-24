"""Export a workspace experiment as a SOL-ExecBench problem directory.

SOL-ExecBench is a downstream rework of flashinfer-bench with an incompatible
BuildSpec: `languages` (a list) replaces `language`, and `target_hardware` is
narrowed to the enum {B200, LOCAL}. A solution cannot be native to both, so this
script translates one out. Everything else — definition, workloads, sources —
carries over unchanged.

    python scripts/solution_to_sol.py --list
    python scripts/solution_to_sol.py --out /tmp/sol/mla
    python scripts/solution_to_sol.py e3_tensor_cores --out /tmp/sol/mla \
        --workspace ../mla-experiments/6_journal/run_1/.agent_workspace

Defaults to the memory's current best experiment. SOL never copies safetensors into
its staging directory; it resolves them against FLASHINFER_TRACE_DIR, and our
task/ holds them under the same dataset-relative paths — so that variable points
back at the workspace, and the printed command sets it.

SOL times with CUPTI, which needs a CUDA-13 driver (>= 580). On an older driver a
correct kernel still passes correctness and then ends every workload as
RUNTIME_ERROR "Timing failed" — that message is the pass signal, not a defect.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

try:
    from scripts._experiment_state import experiment_records, load_experiment_state
except ModuleNotFoundError:
    from _experiment_state import experiment_records, load_experiment_state

REPO_ROOT = Path(__file__).resolve().parent.parent

# flashinfer's BuildSpec.language -> SOL's BuildSpec.languages. SOL folds C++ and
# CUDA into one token and calls plain torch "pytorch"; tilelang has no SOL builder.
LANGUAGES = {"cuda": "cuda_cpp", "cpp": "cuda_cpp", "python": "pytorch", "triton": "triton"}

# SOL accepts only these two. LOCAL makes its packager detect the compile machine's
# arch; B200 forces sm_100a, which will not run anywhere else.
HARDWARE = ("LOCAL", "B200")

# SOL's default is 120s, which a real kernel.cu overruns.
COMPILE_TIMEOUT = 1200


def _default_workspace() -> Path:
    """The workspace setup_workspace.py uses, from config.toml."""
    config = tomllib.loads((REPO_ROOT / "config.toml").read_text())
    path = Path(config.get("task", {}).get("workspace_path", ".agent_workspace"))
    return path if path.is_absolute() else REPO_ROOT / path


def _read_state(workspace: Path) -> dict:
    experiment_state, path = load_experiment_state(workspace, workspace=True)
    if path is None:
        raise SystemExit(
            f"Error: no experiment state under {workspace} — is it a set-up workspace?"
        )
    return experiment_state


def _read_solution(workspace: Path, name: str) -> dict:
    """The archived solution an experiment points at."""
    path = workspace / "archive" / "solutions.jsonl"
    for line in path.read_text().splitlines():
        if line.strip() and json.loads(line)["name"] == name:
            return json.loads(line)
    raise SystemExit(f"Error: solution {name!r} is not in {path}.")


def _translate_spec(spec: dict, hardware: str) -> dict:
    """Rewrite a flashinfer BuildSpec into SOL's schema."""
    language = spec["language"]
    if language not in LANGUAGES:
        raise SystemExit(f"Error: SOL has no builder for language {language!r}.")

    translated = {k: v for k, v in spec.items() if k not in ("language", "target_hardware")}
    translated["languages"] = [LANGUAGES[language]]
    translated["target_hardware"] = [hardware]

    # SOL defaults this to True and then calls fn(*inputs, *outputs), discarding the
    # return value — so an unset flag silently fails every workload. Pin it.
    translated.setdefault("destination_passing_style", False)
    return translated


def _workloads(path: Path) -> list[dict]:
    """Unwrap our Trace records into the bare workload objects SOL expects."""
    return [json.loads(line)["workload"] for line in path.read_text().splitlines() if line.strip()]


def _missing_blobs(task_dir: Path, workloads: list[dict]) -> list[str]:
    """Safetensors inputs these workloads name that task/ does not hold."""
    missing = []
    for workload in workloads:
        for spec in (workload["inputs"] or {}).values():
            if spec.get("type") != "safetensors":
                continue
            path = spec["path"].removeprefix("./")
            if not (task_dir / path).is_file():
                missing.append(path)
    return sorted(set(missing))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", nargs="?", help="Experiment to export (default: the current best).")
    parser.add_argument("--list", action="store_true", help="List the experiments available to export.")
    parser.add_argument("--out", type=Path, help="Problem directory to write.")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root (default: [task] workspace_path).")
    parser.add_argument("--hardware", choices=HARDWARE, default="LOCAL", help="SOL target hardware (default: LOCAL).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing problem directory.")
    args = parser.parse_args(argv)

    workspace = (args.workspace or _default_workspace()).resolve()
    experiment_state = _read_state(workspace)
    experiments = experiment_records(experiment_state)

    if args.list:
        for name, experiment_node in experiments.items():
            best = (
                " (current best)"
                if name == experiment_state.get("current_best")
                else ""
            )
            print(f"{name:24} {experiment_node['solution']}{best}")
        return 0

    experiment = args.experiment or experiment_state.get("current_best")
    if experiment not in experiments:
        raise SystemExit(f"Error: experiment {experiment!r} not found. Use --list to see the available ones.")
    if not args.out:
        raise SystemExit("Error: pass --out to name the problem directory to write.")
    if args.out.exists() and not args.force:
        raise SystemExit(f"Error: {args.out} already exists — pass --force to replace it.")

    solution = _read_solution(workspace, experiments[experiment]["solution"])
    solution["spec"] = _translate_spec(solution["spec"], args.hardware)

    task_dir = workspace / "task"
    workloads = _workloads(task_dir / "workloads.jsonl")
    missing = _missing_blobs(task_dir, workloads)
    if missing:
        raise SystemExit(f"Error: {len(missing)} safetensors input(s) are absent from {task_dir}, "
                         f"e.g. {missing[0]} — SOL will not find them either.")

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task_dir / "definition.json", args.out / "definition.json")
    # SOL reads the singular filename, and one bare workload object per line.
    (args.out / "workload.jsonl").write_text("".join(json.dumps(w) + "\n" for w in workloads))
    (args.out / "solution.json").write_text(json.dumps(solution, indent=2) + "\n")

    print(f"[SOL] exported {experiment} ({solution['name']}) to {args.out}: "
          f"definition.json, workload.jsonl ({len(workloads)} workloads), solution.json.")
    print("\nNext, in an environment whose torch matches the driver:")
    print(f"  FLASHINFER_TRACE_DIR={task_dir} \\")
    print(f"    sol-execbench {args.out.resolve()} --compile-timeout {COMPILE_TIMEOUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
