"""Initialize a kernel-optimization workspace for an MCP client.

The user supplies the task fixtures under <workspace>/task/ in the configured
adapter's native format. This wipes prior-run state (src/, experiments/,
.state/, optimization_journal.md) but never task/, then seeds the tree from the
adapter: the neutral task spec, representative shapes, and journal header become
tree state, so the journal renders the task from the tree. Finally it stages the
baseline's sources into src/, freezes the baseline's build spec as the run's
build spec, and optionally benchmarks and logs it as `v0_baseline`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import _tree, registry
from tools._benchmark import BenchmarkUnavailable, get_adapter
from tools._evaluation import aggregate
from tools._workloads import REPRESENTATIVE_WORKLOAD_LABELS
from tools._workspace import write_benchmark_state
from tools.registry import MCP_SERVER_NAME, select_schemas, validate_enabled
from tools.log_experiment import log_experiment
from tools.benchmark_kernel import benchmark_kernel


def _disp(path: Path, root: Path) -> str:
    """Path relative to the repo root for tidy logging; absolute if outside it."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_path(workspace: Path, raw: str) -> Path:
    """Resolve a config path relative to the workspace root, or as-is if absolute."""
    p = Path(raw)
    return p if p.is_absolute() else workspace / p


def _representative_workloads_from_config(
    task_cfg: dict, workloads_path: Path
) -> dict[str, str]:
    """Load named representative UUIDs and verify that the fixture contains them."""
    raw = task_cfg.get("representative_workloads")
    section = "[[task.representative_workloads]]"
    if not isinstance(raw, list):
        raise SystemExit(
            f"Error: {section} must define one name/uuid pair for each of: "
            f"{', '.join(REPRESENTATIVE_WORKLOAD_LABELS)}."
        )

    configured: dict[str, str] = {}
    seen_uuids: set[str] = set()
    for i, entry in enumerate(raw, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"Error: {section} entry {i} must be a table.")
        name = entry.get("name")
        workload_uuid = entry.get("uuid")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"Error: {section} entry {i} has no string name.")
        if not isinstance(workload_uuid, str) or not workload_uuid:
            raise SystemExit(
                f"Error: {section} entry {i} ({name!r}) has no string uuid."
            )
        if name in configured:
            raise SystemExit(
                f"Error: {section} defines representative name {name!r} more than once."
            )
        if workload_uuid in seen_uuids:
            raise SystemExit(
                f"Error: {section} assigns workload {workload_uuid!r} more than once."
            )
        configured[name] = workload_uuid
        seen_uuids.add(workload_uuid)

    expected = set(REPRESENTATIVE_WORKLOAD_LABELS)
    actual = set(configured)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise SystemExit(
            f"Error: {section} names must be exactly "
            f"{', '.join(REPRESENTATIVE_WORKLOAD_LABELS)} ({'; '.join(details)})."
        )

    try:
        records = [
            json.loads(line)
            for line in workloads_path.read_text().splitlines()
            if line.strip()
        ]
        available = {str(record["workload"]["uuid"]) for record in records}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"Error: could not read workload UUIDs from {workloads_path}: {exc}"
        ) from exc

    missing_uuids = [
        f"{name}={configured[name]}"
        for name in REPRESENTATIVE_WORKLOAD_LABELS
        if configured[name] not in available
    ]
    if missing_uuids:
        raise SystemExit(
            f"Error: configured representative workload(s) not found in "
            f"{workloads_path}: {', '.join(missing_uuids)}."
        )

    return {
        name: configured[name]
        for name in REPRESENTATIVE_WORKLOAD_LABELS
    }


def _server_command(repo_root: Path, workspace: Path) -> list[str]:
    """Stable stdio-server command shared by Claude Code and Codex."""
    return [
        sys.executable,
        str((repo_root / "scripts" / "mcp_server.py").resolve()),
        "--workspace",
        str(workspace.resolve()),
        "--config",
        str((repo_root / "config.toml").resolve()),
    ]


def _write_mcp_config(workspace: Path, repo_root: Path) -> None:
    """Write the project-scoped Claude Code MCP configuration."""
    command = _server_command(repo_root, workspace)
    config = {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": command[0],
                "args": command[1:],
            }
        }
    }
    (workspace / ".mcp.json").write_text(json.dumps(config, indent=2) + "\n")


def _write_claude_settings(workspace: Path, tool_names: list[str]) -> None:
    """Write <workspace>/.claude/settings.local.json so a Claude Code session
    pre-approves the kernel tools (no per-call prompts), enables the MCP server,
    and forbids touching task/ directly — the agent goes through the tools."""
    settings = {
        "permissions": {
            "allow": [f"mcp__{MCP_SERVER_NAME}__{name}" for name in tool_names],
            # Keep the product tool-mediated so direct filesystem operations do
            # not bypass benchmark-cache and journal invariants.
            "deny": ["Read(**)", "Edit(**)", "Write(**)", "Bash(*)"],
        },
        "enabledMcpjsonServers": [MCP_SERVER_NAME],
    }
    settings_dir = workspace / ".claude"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.local.json").write_text(json.dumps(settings, indent=2) + "\n")


CLIENT_INSTRUCTIONS = """You are a GPU kernel performance engineer.

Use the kernel tools to optimize the kernel. The goal is to minimize latency.

Start by reading the optimization journal. It is the only state carried across
iterations: it records the task and build contract, the head and current-best
experiments, measured results, prior branches, profiling observations, open
hypotheses, facts, and hazards.

Revisit conclusions inherited from the journal when new evidence conflicts
with them, and replace or remove stale annotations instead of allowing them to
harden into false constraints.

Do not infer a bottleneck or dismiss a hypothesis without evidence. Try promising rewrites
instead of ruling them out by argument. Do not abandon a high-upside structural
change after its first regression or correctness failure; give the new
structure a fair implementation and tuning budget. Continue exploring genuinely
different structures while you can identify a plausible untried one.

This is an optimization experiment, not a production kernel. Use
`checkout_experiment` to branch from recorded states, preserve useful measured
attempts, compare alternatives, and return to a known-good implementation
without protecting the current working kernel from ambitious changes.
"""


def _write_client_instructions(workspace: Path) -> None:
    """Give Claude Code and Codex the same product-level operating contract."""
    for name in ("CLAUDE.md", "AGENTS.md"):
        (workspace / name).write_text(CLIENT_INSTRUCTIONS)


def _stage_files(src_dir: Path, files: list[tuple[str, str]]) -> None:
    """Replace src/ with the given (name, content) source files. src/ is a flat
    directory of bare filenames — the agent addresses files by name and the tools
    key on it — so a solution declaring a nested source path is rejected here
    rather than crashing or being silently flattened downstream."""
    for name, _ in files:
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise SystemExit(
                f"Error: baseline source path {name!r} is nested; the working "
                f"kernel is a flat set of files — give each source a bare filename."
            )
    if src_dir.exists():
        shutil.rmtree(src_dir)
    src_dir.mkdir(parents=True)
    for name, content in files:
        (src_dir / name).write_text(content)


# Tools that only work with some benchmark adapters. A tool absent here works with
# every adapter. profile_kernel drives ncu against an in-process runnable, so it
# needs an adapter that implements build_profilable(). A profiler for another
# language (proton, torch.profiler) belongs in its own tool — write profile_triton,
# list the adapters it supports here, and enable it in [tools].
TOOL_ADAPTERS = {"profile_kernel": ("flashinfer", "sol")}


def _check_tool_adapters(tool_names: list[str], adapter_name: str) -> None:
    """Reject a tool the selected adapter cannot serve. Aborts rather than dropping
    it: the tool surface is what a run measures, so degrading it quietly would make
    two runs of the same config incomparable."""
    for tool in tool_names:
        supported = TOOL_ADAPTERS.get(tool)
        if supported and adapter_name not in supported:
            raise SystemExit(
                f"Error: [tools] enabled includes {tool!r}, which does not support the "
                f"{adapter_name!r} adapter (supported: {', '.join(supported)}). "
                f"Remove it from [tools] enabled."
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sort-workloads",
        action="store_true",
        help="Order the task's workload fixture smallest-to-largest before seeding "
        "the tree. Named representatives are selected by UUID and are unaffected. "
        "Reports whether the fixture changed.",
    )
    parser.add_argument(
        "--skip-baseline-benchmark",
        action="store_true",
        help="Stage the baseline without benchmarking or logging it. Useful for "
        "an incomplete scaffold; the first experiment logged by the agent is v0.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Get the benchmark adapter from config (the kernel's language comes from the baseline).
    repo_root = Path(__file__).resolve().parent.parent
    config = tomllib.loads((repo_root / "config.toml").read_text())
    task_cfg = config["task"]
    adapter_name = task_cfg["benchmark"]
    dependencies = {"flashinfer": "flashinfer_bench", "sol": "sol_execbench"}
    if adapter_name not in dependencies:
        raise SystemExit(
            f"Error: unknown [task] benchmark {adapter_name!r}; expected one of: "
            f"{', '.join(dependencies)}."
        )
    dependency = dependencies[adapter_name]
    if importlib.util.find_spec(dependency) is None:
        raise SystemExit(
            f"Error: benchmark adapter {adapter_name!r} requires the Python "
            f"module {dependency!r}, but it is not importable."
        )

    # Resolve the workspace and require its task fixtures (adapter-native format).
    workspace = Path(task_cfg["workspace_path"])
    if not workspace.is_absolute():
        workspace = repo_root / workspace
    task_dir = workspace / "task"
    if not task_dir.is_dir():
        raise SystemExit(
            f"Error: {task_dir} not found — supply the task fixtures there "
            f"in {adapter_name}'s native format before running setup."
        )
    missing_fixtures = [
        name for name in ("definition.json", "workloads.jsonl")
        if not (task_dir / name).is_file()
    ]
    if missing_fixtures:
        raise SystemExit(
            f"Error: missing task fixture(s) in {task_dir}: "
            f"{', '.join(missing_fixtures)}."
        )
    representative_workloads = _representative_workloads_from_config(
        task_cfg, task_dir / "workloads.jsonl"
    )

    # Resolve the required starting kernel: a path to a baseline solution .json
    # (relative to the workspace, or absolute) or "reference" (optimize the
    # task's reference).
    src_dir = workspace / "src"
    baseline = task_cfg.get("baseline")
    if not baseline:
        raise SystemExit(
            "Error: [task] baseline is required — a path to a baseline solution .json, or 'reference'."
        )
    baseline_path = None
    if baseline != "reference":
        baseline_path = _resolve_path(workspace, baseline)
        if not baseline_path.is_file():
            raise SystemExit(
                f"Error: [task] baseline = {baseline!r} is not a file (expected a solution .json or 'reference')."
            )

    # Resolve the optional comparison target; [task.target] carries its journal
    # metadata and a `path` to the target solution .json.
    target_cfg = task_cfg.get("target")
    target_path = target_label = target_description = None
    if target_cfg:
        target_label = target_cfg["label"]
        target_description = target_cfg.get("description", target_label)
        target_path = _resolve_path(workspace, target_cfg["path"])
        if not target_path.is_file():
            raise SystemExit(
                f"Error: [task.target] path = {target_cfg['path']!r} is not a file."
            )

    # Check the exposed tools against the adapter before touching anything.
    enabled = config.get("tools", {}).get("enabled")
    try:
        validate_enabled(registry.schemas, enabled)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    tool_names = sorted(s["name"] for s in select_schemas(registry.schemas, enabled))
    _check_tool_adapters(tool_names, adapter_name)
    if "profile_kernel" in tool_names and shutil.which("ncu") is None:
        raise SystemExit(
            "Error: profile_kernel is enabled but ncu (Nsight Compute) is not on PATH. "
            "Install ncu or remove profile_kernel from [tools] enabled."
        )

    # Wipe prior-run state, leaving task/ untouched.
    for stale in (src_dir, workspace / "experiments", workspace / ".state",
                  workspace / _tree.JOURNAL_PATH):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.is_file():
            stale.unlink()

    # Write MCP config + Claude Code settings, pre-approving exactly the exposed tools.
    _write_mcp_config(workspace, repo_root)
    _write_claude_settings(workspace, tool_names)
    _write_client_instructions(workspace)
    print(
        f"[Setup] wrote {_disp(workspace / '.mcp.json', repo_root)} and "
        f"{_disp(workspace / '.claude' / 'settings.local.json', repo_root)}, "
        "plus Claude Code/Codex instructions."
    )

    cwd = Path.cwd()
    os.chdir(workspace)
    try:
        # Record the selected adapter as the run-level state for the runtime tools.
        write_benchmark_state(
            {
                "adapter": adapter_name,
                "representative_workloads": representative_workloads,
            }
        )
        adapter = get_adapter()

        # Optionally canonicalize full-suite order. Named representatives resolve
        # by UUID, so sorting cannot change the smoke/profile workload set.
        if args.sort_workloads:
            changed = adapter.sort_workloads_fixture()
            print(
                "[Setup] sorted workload fixture smallest-to-largest (order changed)."
                if changed
                else "[Setup] workload fixture already sorted (left untouched)."
            )

        # Stage the starting kernel into src/ and freeze its build spec for the run.
        if baseline == "reference":
            files = adapter.reference_baseline_files()
            if not files:
                raise SystemExit(
                    "[Setup] [task] baseline = 'reference' but the task has no runnable reference."
                )
            print(f"[Setup] seeded the initial kernel from the task reference ({len(files)} file(s)).")
        else:
            try:
                files = adapter.baseline_files(baseline_path)
            except ValueError as e:
                raise SystemExit(f"Error: could not load baseline: {e}")
            print(f"[Setup] staged initial kernel from {_disp(baseline_path, repo_root)}")
        _stage_files(src_dir, files)

        # Seed the tree from the adapter (the kernel language now comes from the
        # frozen baseline spec), so the journal reads the task from the tree.
        spec = adapter.task_spec()
        tree = _tree.bootstrap_tree(
            task=spec.name,
            kernel_description=spec.description,
            hardware=adapter.hardware(),
            language=adapter.language(),
        )
        tree["task_spec"] = spec.model_dump()
        tree["representative_workload_axes"] = adapter.representative_axes()
        tree["build_contract"] = adapter.build_contract()
        _tree.save_tree(tree)
        print(
            f"[Setup] created {_disp(workspace / '.state' / 'tree.json', repo_root)} "
            f"and {_disp(workspace / _tree.JOURNAL_PATH, repo_root)}."
        )

        # Measure the comparison target and record it on the tree.
        if target_path is not None:
            print(f"[Setup] measuring target '{target_label}' from {_disp(target_path, repo_root)}...")
            try:
                leaves, solution_id = adapter.benchmark_target(target_path)
            except ValueError as e:
                raise SystemExit(f"Error: could not load target '{target_label}': {e}")
            except BenchmarkUnavailable as e:
                raise SystemExit(f"Error: could not benchmark target '{target_label}': {e}")
            evaluation = aggregate(leaves)
            if evaluation.status != "ALL_PASSED":
                raise SystemExit(
                    f"[Setup] target '{target_label}' did not pass every workload "
                    f"(status={evaluation.status}); aborting."
                )
            tree["target"] = {
                "label": target_label,
                "solution": solution_id,
                "description": target_description,
                "evaluation": evaluation.model_dump(),
            }
            _tree.save_tree(tree)
            print(f"[Setup] recorded target '{target_label}'.")

        # An incomplete scaffold is useful source material but not useful evidence.
        # Leave the tree empty in that mode; the agent's first logged result is v0.
        if args.skip_baseline_benchmark:
            print(
                "[Setup] skipped the baseline benchmark; no initial experiment "
                "was logged."
            )
        else:
            test_result = benchmark_kernel(scope="full")
            if test_result.startswith("Error"):
                raise SystemExit(f"[Setup] benchmark_kernel failed: {test_result}...")
            log_result = log_experiment(
                slug="baseline",
                description="Initial baseline kernel.",
            )
            if log_result.startswith("Error"):
                raise SystemExit(f"[Setup] log_experiment failed: {log_result}")
            print(f"[Setup] {log_result}")
        print()
        server_command = _server_command(repo_root, workspace)
        print("[Setup] Claude Code: cd into the workspace and run `claude`.")
        print("[Setup] Codex registration (once):")
        print(f"  codex mcp add {MCP_SERVER_NAME} -- {shlex.join(server_command)}")
        print(f"[Setup] Then run: codex -C {shlex.quote(str(workspace.resolve()))}")
    finally:
        os.chdir(cwd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
