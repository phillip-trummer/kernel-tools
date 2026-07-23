#!/usr/bin/env python3
"""Non-destructive preflight for MCP, config, and task fixtures."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 11):
        failures.append(f"Python 3.11+ required; found {sys.version.split()[0]}")

    for module in ("mcp", "pydantic", "torch", "trio"):
        if importlib.util.find_spec(module) is None:
            failures.append(f"missing Python module: {module}")

    config_path = REPO / "config.toml"
    try:
        with config_path.open("rb") as f:
            config = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        failures.append(f"cannot load config.toml: {exc}")
        config = {}

    task_cfg = config.get("task") or {}
    adapter = task_cfg.get("benchmark")
    if adapter not in ("flashinfer", "sol"):
        failures.append("[task] benchmark must be 'flashinfer' or 'sol'")
    dependency = {"flashinfer": "flashinfer_bench", "sol": "sol_execbench"}.get(adapter)
    if dependency and importlib.util.find_spec(dependency) is None:
        failures.append(f"adapter {adapter!r} requires importable module {dependency!r}")

    raw_workspace = task_cfg.get("workspace_path")
    if not raw_workspace:
        failures.append("[task] workspace_path is required")
        workspace = REPO / ".agent_workspace"
    else:
        workspace = Path(raw_workspace)
        if not workspace.is_absolute():
            workspace = REPO / workspace

    task_dir = workspace / "task"
    for name in ("definition.json", "workloads.jsonl"):
        if not (task_dir / name).is_file():
            failures.append(f"missing task fixture: {task_dir / name}")

    baseline = task_cfg.get("baseline")
    if not baseline:
        failures.append("[task] baseline is required")
    elif baseline != "reference":
        path = Path(baseline)
        if not path.is_absolute():
            path = workspace / path
        if not path.is_file():
            failures.append(f"baseline Solution not found: {path}")

    enabled = (config.get("tools") or {}).get("enabled")
    try:
        from tools import registry
        from tools.registry import validate_enabled

        validate_enabled(registry.schemas, enabled)
    except Exception as exc:
        failures.append(f"invalid tool configuration: {exc}")

    exposed = enabled or ["profile_kernel"]
    if "profile_kernel" in exposed and shutil.which("ncu") is None:
        warnings.append("profile_kernel is enabled but ncu is not on PATH")
    if shutil.which("claude") is None:
        warnings.append("Claude Code CLI not found (optional if using Codex)")
    if shutil.which("codex") is None:
        warnings.append("Codex CLI not found (optional if using Claude Code)")

    report = {
        "status": "FAILED" if failures else "OK",
        "adapter": adapter,
        "workspace": str(workspace.resolve()),
        "failures": failures,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
