"""Durable state and rendering for the optimization memory."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tools._workloads import REPRESENTATIVE_WORKLOAD_LABELS
from tools._workspace import read_src_files, solution_name_from_src_files


SCHEMA_VERSION = 2
MEMORY_PATH = Path(".state/memory.json")
MEMORY_VIEW_PATH = Path("optimization_memory.md")

EXPERIMENT_RE = re.compile(r"^e(\d+)_")
BRANCH_RE = re.compile(r"^b(\d+)_")

_EXPERIMENT_FIELD = {
    "note": "notes",
    "profiling_observation": "profiling_observations",
}
_TOP_LEVEL_KEY = {
    "open_hypothesis": "open_hypotheses",
    "global_fact": "global_facts",
    "hazard": "hazards",
}

PER_EXPERIMENT_SCOPES = tuple(_EXPERIMENT_FIELD)
TOP_LEVEL_SCOPES = tuple(_TOP_LEVEL_KEY)


def bootstrap_memory(
    *,
    task: str,
    kernel_description: str,
    hardware: str,
    language: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "kernel_description": kernel_description,
        "hardware": hardware,
        "language": language,
        "task_spec": {},
        "build_contract": None,
        "head": None,
        "head_state": None,
        "current_best": None,
        "best_by_representative_workload": {},
        "representative_workload_axes": {},
        "target": None,
        "active_branch": None,
        "frontier": [],
        "branches": {},
        "experiments": {},
        "open_hypotheses": [],
        "global_facts": [],
        "hazards": [],
    }


def load_memory() -> dict:
    # Load current state
    if not MEMORY_PATH.is_file():
        raise FileNotFoundError(
            f"{MEMORY_PATH} not found; run scripts/setup_workspace.py before "
            "invoking memory tools."
        )
    memory = json.loads(MEMORY_PATH.read_text())
    if memory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported optimization memory schema "
            f"{memory.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    return memory


def save_memory(memory: dict) -> None:
    # Refresh working state
    refresh_head_state(memory)

    # Persist canonical state
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(json.dumps(memory, indent=2) + "\n")

    # Render readable memory
    MEMORY_VIEW_PATH.write_text(render_memory(memory))


def get_head(memory: dict) -> Optional[str]:
    return memory["head"]


def get_experiment(memory: dict, experiment_id: str) -> dict:
    return memory["experiments"][experiment_id]


def has_experiment(memory: dict, experiment_id: str) -> bool:
    return experiment_id in memory["experiments"]


def list_experiment_ids(memory: dict) -> list[str]:
    return sorted(memory["experiments"], key=_experiment_order)


def has_branch(memory: dict, branch_id: str) -> bool:
    return branch_id in memory["branches"]


def list_branch_ids(memory: dict) -> list[str]:
    return sorted(memory["branches"], key=_branch_order)


def find_experiment_by_solution(memory: dict, solution_name: str) -> Optional[str]:
    for experiment_id, experiment in memory["experiments"].items():
        if experiment["solution"] == solution_name:
            return experiment_id
    return None


def find_branch_for_experiment(memory: dict, experiment_id: str) -> Optional[str]:
    for branch_id, branch in memory["branches"].items():
        if experiment_id in branch["experiments"]:
            return branch_id
    return None


def parent_branch_id(memory: dict, branch_id: str) -> Optional[str]:
    base = memory["branches"][branch_id]["base_experiment"]
    return find_branch_for_experiment(memory, base) if base else None


def active_branch(memory: dict) -> Optional[dict]:
    branch_id = memory["active_branch"]
    return memory["branches"].get(branch_id) if branch_id else None


def refresh_head_state(memory: dict) -> None:
    head = memory["head"]
    experiment = memory["experiments"].get(head) if head else None
    solution = experiment.get("solution") if experiment else None
    files = read_src_files() if solution else []
    if not solution or not files:
        memory["head_state"] = None
        return
    memory["head_state"] = (
        "clean" if solution_name_from_src_files(files) == solution else "dirty"
    )


def next_experiment_number(memory: dict, experiments_dir: Path) -> int:
    numbers: list[int] = []
    if experiments_dir.is_dir():
        for path in experiments_dir.iterdir():
            match = EXPERIMENT_RE.match(path.name) if path.is_dir() else None
            if match:
                numbers.append(int(match.group(1)))
    for experiment_id in memory["experiments"]:
        match = EXPERIMENT_RE.match(experiment_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_branch_number(memory: dict) -> int:
    numbers = []
    for branch_id in memory["branches"]:
        match = BRANCH_RE.match(branch_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def add_branch(
    memory: dict,
    *,
    branch_id: str,
    base_experiment: Optional[str],
    strategy: str,
) -> None:
    memory["branches"][branch_id] = {
        "base_experiment": base_experiment,
        "strategy": strategy,
        "state": "planned",
        "experiments": [],
        "representative": None,
        "conclusion": None,
    }
    memory["active_branch"] = branch_id
    if branch_id not in memory["frontier"]:
        memory["frontier"].append(branch_id)


def add_experiment(
    memory: dict,
    *,
    experiment_id: str,
    branch_id: str,
    parent: Optional[str],
    solution: str,
    description: str,
    tags: list[str],
    evaluation: dict,
    initial_note: Optional[str] = None,
) -> None:
    memory["experiments"][experiment_id] = {
        "parent": parent,
        "solution": solution,
        "description": description,
        "tags": list(tags),
        "evaluation": evaluation,
        "notes": [initial_note] if initial_note else [],
        "profiling_observations": [],
    }
    branch = memory["branches"][branch_id]
    branch["experiments"].append(experiment_id)
    branch["state"] = "active"
    update_branch_representative(memory, branch_id, experiment_id)


def add_baseline_branch(
    memory: dict,
    *,
    branch_id: str,
    experiment_id: str,
    solution: str,
    description: str,
    tags: list[str],
    evaluation: dict,
    initial_note: Optional[str] = None,
) -> None:
    add_branch(
        memory,
        branch_id=branch_id,
        base_experiment=None,
        strategy=description,
    )
    add_experiment(
        memory,
        experiment_id=experiment_id,
        branch_id=branch_id,
        parent=None,
        solution=solution,
        description=description,
        tags=tags,
        evaluation=evaluation,
        initial_note=initial_note,
    )
    memory["branches"][branch_id]["state"] = "complete"
    memory["active_branch"] = None


def complete_branch(
    memory: dict,
    branch_id: str,
    conclusion: str,
    keep_on_frontier: bool,
) -> None:
    branch = memory["branches"][branch_id]
    branch["state"] = "complete"
    branch["conclusion"] = conclusion
    if keep_on_frontier:
        if branch_id not in memory["frontier"]:
            memory["frontier"].append(branch_id)
    elif branch_id in memory["frontier"]:
        memory["frontier"].remove(branch_id)


def set_head(memory: dict, experiment_id: str) -> None:
    memory["head"] = experiment_id


def update_branch_representative(
    memory: dict,
    branch_id: str,
    experiment_id: str,
) -> bool:
    branch = memory["branches"][branch_id]
    current_id = branch["representative"]
    if current_id is None:
        branch["representative"] = experiment_id
        return True

    new_evaluation = memory["experiments"][experiment_id]["evaluation"]
    current_evaluation = memory["experiments"][current_id]["evaluation"]
    new_scorable = _scorable(new_evaluation)
    current_scorable = _scorable(current_evaluation)
    if (new_scorable and not current_scorable) or (
        new_scorable and _geomean_better(new_evaluation, current_evaluation)
    ) or (not new_scorable and not current_scorable):
        branch["representative"] = experiment_id
        return True
    return False


def update_bests(memory: dict, experiment_id: str, evaluation: dict) -> list[str]:
    updated: list[str] = []
    experiments = memory["experiments"]

    current_id = memory["current_best"]
    current_evaluation = experiments[current_id]["evaluation"] if current_id else None
    if _geomean_better(evaluation, current_evaluation):
        memory["current_best"] = experiment_id
        updated.append("current_best")

    bests = memory["best_by_representative_workload"]
    for label, result in evaluation["representative_workloads"].items():
        if result["outcome"] != "PASSED":
            continue
        current_id = bests.get(label)
        current_result = (
            experiments[current_id]["evaluation"]["representative_workloads"].get(label)
            if current_id
            else None
        )
        if _result_better(result, current_result):
            bests[label] = experiment_id
            updated.append(label)
    return updated


def _scorable(evaluation: dict) -> bool:
    return (
        evaluation.get("geomean_speedup_factor") is not None
        or evaluation.get("geomean_latency_ms") is not None
    )


def _geomean_better(new: dict, current: Optional[dict]) -> bool:
    return _metric_better(
        new.get("geomean_speedup_factor"),
        new.get("geomean_latency_ms"),
        (current or {}).get("geomean_speedup_factor"),
        (current or {}).get("geomean_latency_ms"),
    )


def _result_better(new: dict, current: Optional[dict]) -> bool:
    return _metric_better(
        new.get("speedup_factor"),
        new.get("latency_ms"),
        (current or {}).get("speedup_factor"),
        (current or {}).get("latency_ms"),
    )


def _metric_better(
    new_speedup: Optional[float],
    new_latency: Optional[float],
    current_speedup: Optional[float],
    current_latency: Optional[float],
) -> bool:
    if new_speedup is not None:
        return current_speedup is None or new_speedup > current_speedup
    if new_latency is None:
        return False
    return current_latency is None or new_latency < current_latency


def _annotation_list(
    memory: dict,
    scope: str,
    experiment_id: Optional[str],
) -> list[str]:
    if scope in _EXPERIMENT_FIELD:
        return memory["experiments"][experiment_id][_EXPERIMENT_FIELD[scope]]
    return memory[_TOP_LEVEL_KEY[scope]]


def add_annotation(
    memory: dict,
    scope: str,
    text: str,
    experiment_id: Optional[str] = None,
) -> None:
    _annotation_list(memory, scope, experiment_id).append(text)


def edit_annotation(
    memory: dict,
    scope: str,
    old_text: str,
    new_text: Optional[str],
    experiment_id: Optional[str] = None,
) -> Optional[str]:
    items = _annotation_list(memory, scope, experiment_id)
    matches = [index for index, item in enumerate(items) if old_text in item]
    if not matches:
        return f"Error: no {scope!r} entry contains {old_text!r}."
    if len(matches) > 1:
        return (
            f"Error: old_text {old_text!r} matches {len(matches)} {scope!r} entries; "
            "use a longer substring unique to the one you mean."
        )
    index = matches[0]
    if new_text is None:
        items.pop(index)
    else:
        items[index] = new_text
    return None


def render_memory(memory: dict) -> str:
    lines = ["# Optimization Memory", ""]
    lines.extend(_render_header(memory))
    lines.extend(_render_task_spec(memory))
    lines.extend(_render_branch_collection(memory, list_branch_ids(memory), full=True))
    return "\n".join(lines).rstrip() + "\n"


def render_frontier_memory(memory: dict) -> str:
    lines = ["# Optimization Memory", ""]
    lines.extend(_render_header(memory))
    lines.extend(_render_task_spec(memory))
    branch_ids = [
        branch_id
        for branch_id in memory["frontier"]
        if branch_id in memory["branches"]
    ]
    lines.extend(_render_branch_collection(memory, branch_ids, full=False))
    lines.append(
        f"_({len(branch_ids)} frontier structure"
        f"{'s' if len(branch_ids) != 1 else ''} shown from "
        f"{len(memory['branches'])} total; pass branch_id to read_memory "
        "to inspect its complete experiment history.)_"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_branch_memory(memory: dict, branch_id: str) -> str:
    lines = ["# Optimization Memory", ""]
    lines.extend(_render_header(memory))
    lines.extend(_render_task_spec(memory))
    lines.extend(_render_branch_collection(memory, [branch_id], full=True))
    return "\n".join(lines).rstrip() + "\n"


def _render_branch_collection(
    memory: dict,
    branch_ids: list[str],
    *,
    full: bool,
) -> list[str]:
    lines = ["## Structural branches", ""]
    if not branch_ids:
        lines.extend(["_(none)_", ""])
        return lines
    for branch_id in branch_ids:
        lines.extend(_render_branch(memory, branch_id, full=full))
        lines.append("")
    return lines


def _render_header(memory: dict) -> list[str]:
    lines = [f"- **Task:** {memory['task']}"]
    if memory["kernel_description"]:
        lines.append(f"- **Kernel:** {memory['kernel_description']}")
    lines.extend(
        [
            f"- **Hardware:** {memory['hardware']}",
            f"- **Language:** {memory['language']}",
        ]
    )
    axes_by_label = memory.get("representative_workload_axes") or {}
    labels = [
        label
        for label in REPRESENTATIVE_WORKLOAD_LABELS
        if axes_by_label.get(label)
    ]
    if labels:
        lines.append("- **Representative workloads:**")
        for label in labels:
            axes = ", ".join(
                f"{key}={value}" for key, value in axes_by_label[label].items()
            )
            lines.append(f"  - {label}: {axes}")
    lines.append("")

    target_label, target_evaluation = _target(memory)
    if target_label is not None:
        lines.append(
            f"- **Target:** `{target_label}` — "
            f"{_evaluation_summary(target_evaluation)}"
        )
        lines.extend(_representative_workload_lines(target_evaluation))

    current_best = memory["current_best"]
    if current_best:
        evaluation = memory["experiments"][current_best]["evaluation"]
        lines.append(
            f"- **Current best:** `{current_best}` — "
            f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
        )
    else:
        lines.append("- **Current best:** _(unset)_")

    bests = memory["best_by_representative_workload"]
    if bests:
        rendered = ", ".join(
            f"{label}=`{bests[label]}`"
            for label in REPRESENTATIVE_WORKLOAD_LABELS
            if label in bests
        )
        lines.append(f"- **Best by representative workload:** {rendered}")

    head = memory["head"]
    if head:
        suffix = (
            " _(dirty — working kernel has diverged from head)_"
            if memory["head_state"] == "dirty"
            else ""
        )
        lines.append(f"- **Head:** `{head}`{suffix}")
    else:
        lines.append("- **Head:** _(unset)_")
    lines.append("")

    lines.extend(_render_active_branch(memory))
    for title, key in (
        ("Open hypotheses", "open_hypotheses"),
        ("Global facts", "global_facts"),
        ("Hazards", "hazards"),
    ):
        lines.extend([f"## {title}", ""])
        items = memory[key]
        lines.extend((f"- {item}" for item in items) if items else ["_(none)_"])
        lines.append("")
    return lines


def _render_active_branch(memory: dict) -> list[str]:
    lines = ["## Active structural branch", ""]
    branch_id = memory["active_branch"]
    if branch_id is None:
        lines.extend(
            [
                "_(none — call create_handoff to assign the next structural strategy)_",
                "",
            ]
        )
        return lines
    branch = memory["branches"][branch_id]
    lines.extend(
        [
            f"- **Branch:** `{branch_id}` ({branch['state']})",
            (
                f"- **Base experiment:** `{branch['base_experiment']}`"
                if branch["base_experiment"]
                else "- **Base experiment:** _(working kernel)_"
            ),
            f"- **Strategy:** {branch['strategy']}",
            "",
        ]
    )
    return lines


def _render_branch(memory: dict, branch_id: str, *, full: bool) -> list[str]:
    branch = memory["branches"][branch_id]
    parent_branch = parent_branch_id(memory, branch_id)
    lines = [f"### `{branch_id}` ({branch['state']})"]
    lines.append(
        f"- **Parent branch:** `{parent_branch}`"
        if parent_branch
        else "- **Parent branch:** _(root)_"
    )
    lines.append(
        f"- **Base experiment:** `{branch['base_experiment']}`"
        if branch["base_experiment"]
        else "- **Base experiment:** _(none)_"
    )
    lines.append(f"- **Strategy:** {branch['strategy']}")
    representative = branch["representative"]
    lines.append(
        f"- **Representative:** `{representative}`"
        if representative
        else "- **Representative:** _(no experiment logged yet)_"
    )
    if branch["conclusion"]:
        lines.append(f"- **Conclusion:** {branch['conclusion']}")

    experiment_ids = branch["experiments"] if full else ([representative] if representative else [])
    if experiment_ids:
        target_label, target_evaluation = _target(memory)
        lines.append("- **Experiments:**")
        for experiment_id in experiment_ids:
            lines.extend(
                _indent(
                    _render_experiment(
                        experiment_id,
                        memory["experiments"][experiment_id],
                        target_label,
                        target_evaluation,
                        heading_level=0,
                    ),
                    "  ",
                )
            )
    return lines


def _indent(lines: list[str], prefix: str) -> list[str]:
    return [prefix + line if line else line for line in lines]


def _render_task_spec(memory: dict) -> list[str]:
    spec = memory.get("task_spec") or {}
    if not spec:
        return []
    lines = ["## Task specification", ""]
    contract = memory.get("build_contract")
    if contract:
        lines.append(f"- **Build contract:** {contract}")
    if spec.get("op_type"):
        lines.append(f"- **Operation:** {spec['op_type']}")
    tolerance = spec.get("tolerance")
    if tolerance:
        description = _tolerance_desc(tolerance)
        if description:
            lines.append(f"- **Correctness tolerance:** {description}")
    axes = spec.get("axes") or {}
    if axes:
        lines.append("- **Axes:**")
        for name, axis in axes.items():
            lines.append(f"  - `{name}`: {_axis_desc(axis)}")
    for title, key in (("Inputs", "inputs"), ("Outputs", "outputs")):
        fields = spec.get(key) or {}
        if fields:
            lines.append(f"- **{title}:**")
            for name, field in fields.items():
                lines.append(f"  - `{name}`: {_tensor_desc(field)}")
    constraints = spec.get("constraints") or []
    if constraints:
        lines.append("- **Constraints:**")
        lines.extend(f"  - {constraint}" for constraint in constraints)
    reference = spec.get("reference") or ""
    if reference:
        lines.extend(
            [
                "- **Reference implementation:**",
                "",
                "```python",
                *reference.splitlines(),
                "```",
            ]
        )
    lines.append("")
    return lines


def _axis_desc(axis: dict) -> str:
    kind = axis.get("kind", "var")
    if kind == "const":
        head = f"const = {axis.get('value')}"
    elif kind == "expr":
        expression = axis.get("expression")
        head = f"expr = {expression}" if expression else "expr"
    else:
        head = kind
    description = axis.get("description")
    return f"{head} — {description}" if description else head


def _tolerance_desc(tolerance: dict) -> str:
    parts = []
    if tolerance.get("max_rtol") is not None:
        parts.append(f"rtol {tolerance['max_rtol']}")
    if tolerance.get("max_atol") is not None:
        parts.append(f"atol {tolerance['max_atol']}")
    ratio = tolerance.get("required_matched_ratio")
    if ratio is not None:
        parts.append(f"≥{ratio:.0%} of elements within tolerance")
    return ", ".join(parts)


def _tensor_desc(field: dict) -> str:
    shape = field.get("shape")
    shape_text = f"[{', '.join(map(str, shape))}]" if shape else "scalar"
    base = f"{shape_text} {field.get('dtype', '?')}"
    description = field.get("description")
    return f"{base} — {description}" if description else base


def render_commit(memory: dict, experiment_id: str) -> str:
    branch_id = find_branch_for_experiment(memory, experiment_id)
    lines = [f"Logged {experiment_id} on branch {branch_id}.", ""]
    lines.extend(_render_header(memory))
    target_label, target_evaluation = _target(memory)
    lines.extend(
        _render_experiment(
            experiment_id,
            memory["experiments"][experiment_id],
            target_label,
            target_evaluation,
        )
    )
    lines.extend(
        [
            "",
            f"_({len(memory['experiments'])} experiments across "
            f"{len(memory['branches'])} structural branches — call read_memory "
            "for the frontier or pass branch_id for its full history.)_",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_checkout(
    memory: dict,
    experiment_id: str,
    previous_head: Optional[str],
    file_count: int,
) -> str:
    lines = [
        f"Checked out {experiment_id} (head was {previous_head!r}); restored "
        f"{file_count} file(s) into the working kernel.",
        "",
    ]
    lines.extend(_render_header(memory))
    target_label, target_evaluation = _target(memory)
    lines.extend(
        _render_experiment(
            experiment_id,
            memory["experiments"][experiment_id],
            target_label,
            target_evaluation,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_handoff(
    memory: dict,
    branch_id: str,
    file_count: int,
    is_bootstrap: bool,
) -> str:
    branch = memory["branches"][branch_id]
    base = branch["base_experiment"]
    restored = (
        f" Restored `{base}` into the working kernel ({file_count} file(s))."
        if base
        else ""
    )
    next_action = (
        "This is the initial assignment. Implement and fully tune this branch now; "
        "finish the session by creating its handoff."
        if is_bootstrap
        else "End this optimization session. The next session should read_memory and "
        "implement only this branch before creating another handoff."
    )
    return (
        f"Created structural branch `{branch_id}`.{restored}\n\n"
        f"- **Strategy:** {branch['strategy']}\n"
        f"- **State:** planned\n\n"
        f"{next_action}"
    )


def render_annotation(
    memory: dict,
    scope: str,
    experiment_id: Optional[str] = None,
) -> str:
    if scope in _EXPERIMENT_FIELD:
        field = _EXPERIMENT_FIELD[scope]
        items = memory["experiments"][experiment_id][field]
        lines = [f"`{experiment_id}` {field.replace('_', ' ')}:"]
    else:
        key = _TOP_LEVEL_KEY[scope]
        items = memory[key]
        lines = [f"{key.replace('_', ' ').capitalize()}:"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _experiment_order(experiment_id: str) -> tuple[int, int, str]:
    match = EXPERIMENT_RE.match(experiment_id)
    if match:
        return (0, int(match.group(1)), experiment_id)
    return (-1, -1, experiment_id)


def _branch_order(branch_id: str) -> tuple[int, str]:
    match = BRANCH_RE.match(branch_id)
    return (int(match.group(1)), branch_id) if match else (-1, branch_id)


def _render_experiment(
    experiment_id: str,
    experiment: dict,
    target_label: Optional[str],
    target_evaluation: Optional[dict],
    *,
    heading_level: int = 4,
) -> list[str]:
    evaluation = experiment["evaluation"]
    prefix = f"{'#' * heading_level} " if heading_level else ""
    lines = [f"{prefix}`{experiment_id}` ({evaluation['status']})"]
    parent = experiment["parent"]
    lines.append(f"- **Parent:** `{parent}`" if parent else "- **Parent:** _(root)_")
    if experiment["tags"]:
        lines.append(f"- **Tags:** {', '.join(experiment['tags'])}")
    if experiment["description"]:
        lines.append(f"- **Description:** {experiment['description']}")
    lines.append(
        f"- **Evaluation:** "
        f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
    )
    lines.extend(
        _representative_workload_lines(
            evaluation,
            target_label,
            target_evaluation,
        )
    )
    lines.append("- **Notes:**")
    lines.extend(
        (f"  - {note}" for note in experiment["notes"])
        if experiment["notes"]
        else ["  - _(none)_"]
    )
    lines.append("- **Profiling observations:**")
    lines.extend(
        (f"  - {observation}" for observation in experiment["profiling_observations"])
        if experiment["profiling_observations"]
        else ["  - _(none)_"]
    )
    return lines


def _target(memory: dict) -> tuple[Optional[str], Optional[dict]]:
    target = memory.get("target")
    if not target:
        return None, None
    return target["label"], target["evaluation"]


def _evaluation_summary(
    evaluation: dict,
    target_label: Optional[str] = None,
    target_evaluation: Optional[dict] = None,
) -> str:
    parts = [evaluation["status"]]
    count = evaluation.get("workload_count")
    over = f" over {count} workloads" if count else ""
    geomean = evaluation.get("geomean_speedup_factor")
    if geomean is not None:
        parts.append(f"geomean {geomean:.2f}× vs reference{over}")
    else:
        latency = evaluation.get("geomean_latency_ms")
        if latency is not None:
            parts.append(f"geomean {_fmt_ms(latency)}{over}")
    ratio = _target_geomean_ratio(evaluation, target_evaluation)
    if target_label is not None and ratio is not None:
        parts.append(f"{ratio:.2f}× vs {target_label}")
    return "; ".join(parts)


def _fmt_ms(milliseconds: float) -> str:
    return f"{milliseconds:.4g} ms"


def _representative_workload_lines(
    evaluation: dict,
    target_label: Optional[str] = None,
    target_evaluation: Optional[dict] = None,
) -> list[str]:
    lines = []
    representatives = evaluation["representative_workloads"]
    target_representatives = (
        target_evaluation.get("representative_workloads", {})
        if target_evaluation
        else {}
    )
    for label in REPRESENTATIVE_WORKLOAD_LABELS:
        if label not in representatives:
            continue
        result = representatives[label]
        outcome = result["outcome"]
        line = f"  - representative {label}: {outcome}"
        if outcome == "PASSED":
            metrics = []
            speedup = result.get("speedup_factor")
            if speedup is not None:
                metrics.append(f"{speedup:.2f}× vs reference")
            elif result.get("latency_ms") is not None:
                metrics.append(_fmt_ms(result["latency_ms"]))
            ratio = _target_representative_workload_ratio(
                result,
                target_representatives.get(label),
            )
            if target_label is not None and ratio is not None:
                metrics.append(f"{ratio:.2f}× vs {target_label}")
            if metrics:
                line += f"; {'; '.join(metrics)}"
        else:
            tolerance = result.get("tolerance")
            rendered = _format_tolerance(tolerance) if tolerance else ""
            if rendered:
                line += f"; tolerance {rendered}"
        lines.append(line)
    return lines


def _format_tolerance(tolerance: dict) -> str:
    parts = []
    for key, label in (
        ("max_atol", "atol"),
        ("max_rtol", "rtol"),
        ("required_matched_ratio", "matched"),
    ):
        value = tolerance.get(key)
        if value is not None:
            parts.append(f"{label}={value:g}")
    return ", ".join(parts)


def _target_geomean_ratio(
    evaluation: dict,
    target_evaluation: Optional[dict],
) -> Optional[float]:
    if target_evaluation is None:
        return None
    return _vs_target(
        evaluation.get("geomean_speedup_factor"),
        evaluation.get("geomean_latency_ms"),
        target_evaluation.get("geomean_speedup_factor"),
        target_evaluation.get("geomean_latency_ms"),
    )


def _target_representative_workload_ratio(
    result: dict,
    target_result: Optional[dict],
) -> Optional[float]:
    if target_result is None:
        return None
    return _vs_target(
        result.get("speedup_factor"),
        result.get("latency_ms"),
        target_result.get("speedup_factor"),
        target_result.get("latency_ms"),
    )


def _vs_target(
    speedup: Optional[float],
    latency: Optional[float],
    target_speedup: Optional[float],
    target_latency: Optional[float],
) -> Optional[float]:
    if speedup is not None and target_speedup is not None:
        return _ratio(speedup, target_speedup)
    return _ratio(target_latency, latency)


def _ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator
