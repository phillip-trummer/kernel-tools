"""Workspace filesystem ABI.

Owns the agent workspace directory layout (paths, file naming), the generic
filesystem helpers the rest of the codebase composes against, and the `.state/`
run artifacts — the opaque run-level state (`benchmark.json`) and the typed run
benchmark cache (`BenchmarkCache`, over the harness's own `Evaluation` type).
No flashinfer or tree imports — the adapters (`adapters/`) and `_tree.py` build
on this layer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from tools._evaluation import Evaluation


# --- Paths ---
# All workspace paths are relative, i.e. resolved against the current working
# directory. Tools therefore assume they are invoked from the agent workspace
# root (the dir holding task/, src/, archive/, ...); running from elsewhere
# yields FileNotFoundError on task fixtures rather than a clearer error.
TASK_DIR = Path("task")
ARCHIVE_DIR = Path("archive")
SRC_DIR = Path("src")
EXPERIMENTS_DIR = Path("experiments")
BENCHMARK_CACHE_PATH = Path(".state/benchmark_cache.json")
# One run-level config/provenance file: the selected adapter + the frozen build
# spec (and, later, environment/timing/oracle/normalizer/anchors). Internal —
# Keep internal run state separate from agent-facing memory.
BENCHMARK_STATE_PATH = Path(".state/benchmark.json")


# --- Experiment snapshots ---
def resolve_experiment_dir(experiment_id: str) -> Path | str:
    """Resolve experiments/<experiment_id>/ with path-traversal protection.
    Returns the resolved Path on success, or an error message string."""
    experiments_dir = EXPERIMENTS_DIR.resolve()
    exp_dir = (experiments_dir / experiment_id).resolve()
    if experiments_dir not in exp_dir.parents:
        return f"{experiment_id!r} is not a valid experiment id."
    if not exp_dir.is_dir():
        available = (
            sorted(p.name for p in experiments_dir.iterdir() if p.is_dir())
            if experiments_dir.is_dir()
            else []
        )
        return f"experiment {experiment_id!r} not found. Available: {available}"
    return exp_dir


def restore_experiment(experiment_id: str) -> int | str:
    """Mirror a logged experiment into the working kernel."""
    exp_dir = resolve_experiment_dir(experiment_id)
    if not isinstance(exp_dir, Path):
        return exp_dir
    exp_files = [p for p in exp_dir.iterdir() if p.is_file()]
    if not exp_files:
        return f"experiment {experiment_id!r} has no source files; refusing to wipe working kernel."

    # Prepare working directory
    src_dir = SRC_DIR.resolve()
    src_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale files
    exp_names = {p.name for p in exp_files}
    for stale in src_dir.iterdir():
        if stale.is_file() and stale.name not in exp_names:
            stale.unlink()

    # Restore snapshot
    for source in exp_files:
        shutil.copyfile(source, src_dir / source.name)
    return len(exp_files)


# --- Working source (src/) ---
def read_src_files() -> list[tuple[str, str]]:
    """Return [(name, content), ...] for every file in src/, sorted by name.
    src/ is a flat directory of bare filenames (setup rejects nested source
    paths), so basenames are unambiguous keys."""
    if not SRC_DIR.is_dir():
        return []
    return [
        (p.name, p.read_text())
        for p in sorted(SRC_DIR.iterdir())
        if p.is_file()
    ]


def src_files_hash(file_pairs: list[tuple[str, str]]) -> str:
    """SHA256 over (name, content) pairs."""
    h = hashlib.sha256()
    for name, content in file_pairs:
        h.update(name.encode())
        h.update(b"\0")
        h.update(content.encode())
        h.update(b"\0")
    return h.hexdigest()


def solution_name_from_src_files(file_pairs: list[tuple[str, str]]) -> str:
    """The canonical `candidate_<hash16>` name identifying a src/ snapshot.
    Used as both the flashinfer Solution.name and the experiment dedup key."""
    return f"candidate_{src_files_hash(file_pairs)[:16]}"


# --- Run-level benchmark state (.state/benchmark.json) ---
def read_benchmark_state() -> dict | None:
    """The resolved run-level state for this workspace (`adapter`, `build_spec`,
    …), or None if unset. Written by setup so tools resolve the adapter and its
    frozen build spec without the repo config."""
    if not BENCHMARK_STATE_PATH.is_file():
        return None
    return json.loads(BENCHMARK_STATE_PATH.read_text())


def write_benchmark_state(state: dict) -> None:
    BENCHMARK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_STATE_PATH.write_text(json.dumps(state, indent=2))


def read_build_spec() -> dict | None:
    """The build spec frozen from the baseline at setup (adapter-native, opaque
    JSON here), or None if unset. The working kernel rebuilds against this on
    every test, so whatever the baseline declared governs the whole run."""
    state = read_benchmark_state() or {}
    return state.get("build_spec")


def write_build_spec(spec: dict) -> None:
    """Freeze the build spec onto the run-level state, preserving the rest of it
    (e.g. the `adapter` key setup already wrote)."""
    state = read_benchmark_state() or {}
    state["build_spec"] = spec
    write_benchmark_state(state)


# --- Run benchmark cache (.state/benchmark_cache.json) ---
class BenchmarkCache(BaseModel):
    """Benchmark results for every source snapshot benchmarked in the current
    run, keyed by its candidate_<hash16> name. benchmark_kernel records into it;
    log_experiment recomputes the current src hash and looks it up, so a revert
    to any previously-benchmarked snapshot can be logged without re-running (a
    single most-recent slot would lose the result the moment the agent explored
    past a peak and reverted). `last` names the most recently recorded snapshot."""
    entries: dict[str, Evaluation] = Field(default_factory=dict)
    last: str | None = None

    @classmethod
    def load(cls, path: Path) -> "BenchmarkCache":
        if path.is_file():
            return cls.model_validate_json(path.read_text())
        return cls()

    def record(self, solution_name: str, evaluation: Evaluation) -> None:
        self.entries[solution_name] = evaluation
        self.last = solution_name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
