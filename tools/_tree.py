"""Durable state for the optimization tree: tree.json + optimization_journal.md.

This module owns the tree's data shape end-to-end. Tools route every tree
read and mutation through the helpers here so the on-disk schema (node
fields, top-level keys, head/best pointers) can evolve in one file. The
journal is a compact, agent-readable rendering of tree.json and is always
regenerated from it — never hand-edited and never the source of truth.
save_tree writes the json and re-renders the journal in a single call so
the two stay in sync.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tools._workloads import REPRESENTATIVE_WORKLOAD_LABELS
from tools._workspace import read_src_files, solution_name_from_src_files


TREE_PATH = Path(".state/tree.json")
JOURNAL_PATH = Path("optimization_journal.md")

VN_RE = re.compile(r"^v(\d+)_")

# Scope names exposed by annotate_journal -> tree storage location.
_NODE_FIELD = {
    "note": "notes",
    "profiling_observation": "profiling_observations",
}
_TOP_LEVEL_KEY = {
    "open_hypothesis": "open_hypotheses",
    "global_fact": "global_facts",
    "hazard": "hazards",
}


# --- Persistence ---
def bootstrap_tree(
    *,
    task: str,
    kernel_description: str,
    hardware: str,
    language: str,
) -> dict:
    """Return a fresh tree with every schema key initialized to its empty value.
    The single place the top-level schema is declared. Tools mutate values
    within this shape and add/remove rows in `nodes` /
    `best_by_representative_workload`, but never add or remove top-level keys."""
    return {
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
        "open_hypotheses": [],
        "global_facts": [],
        "hazards": [],
        "nodes": {},
    }


def load_tree() -> dict:
    """Load tree.json. Raises FileNotFoundError if the workspace has not been
    bootstrapped — callers assume the bootstrap_tree schema is on disk and
    accessing missing keys would fail with KeyError further down. Fail at the
    boundary instead, with a message that points at the actual cause."""
    if not TREE_PATH.is_file():
        raise FileNotFoundError(
            f"{TREE_PATH} not found; run scripts/setup_workspace.py before invoking tree tools."
        )
    return json.loads(TREE_PATH.read_text())


def save_tree(tree: dict) -> None:
    """Persist tree.json and re-render the journal. Refreshes head_state from
    the current working kernel hash before writing, so tree.json on disk and
    the rendered journal always agree on whether head is clean/dirty."""
    refresh_head_state(tree)
    TREE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TREE_PATH.write_text(json.dumps(tree, indent=2) + "\n")
    JOURNAL_PATH.write_text(render_journal(tree))


# --- Reads ---
def get_head(tree: dict) -> Optional[str]:
    return tree["head"]


def has_node(tree: dict, node_id: str) -> bool:
    return node_id in tree["nodes"]


def list_node_ids(tree: dict) -> list[str]:
    return sorted(tree["nodes"].keys())


def find_node_by_solution(tree: dict, solution_name: str) -> Optional[str]:
    for node_id, node in tree["nodes"].items():
        if node["solution"] == solution_name:
            return node_id
    return None


def refresh_head_state(tree: dict) -> None:
    """Set tree['head_state'] to 'clean', 'dirty', or None depending on whether
    the working kernel matches the head node's content hash. None when there
    is nothing to compare against (head unset, head node missing, or src/
    empty). Called by save_tree so tree.json always carries a current
    head_state — kernel tools never touch this field."""
    head = tree["head"]
    head_node = tree["nodes"].get(head) if head else None
    head_solution = head_node.get("solution") if head_node else None
    src_files = read_src_files() if head_solution else []
    if not head_solution or not src_files:
        tree["head_state"] = None
        return
    tree["head_state"] = (
        "clean" if solution_name_from_src_files(src_files) == head_solution else "dirty"
    )


def next_version(tree: dict, experiments_dir: Path) -> int:
    """Smallest vN+1 not yet used by any tree node or experiments/ snapshot dir."""
    versions: list[int] = []
    if experiments_dir.is_dir():
        for p in experiments_dir.iterdir():
            if not p.is_dir():
                continue
            m = VN_RE.match(p.name)
            if m:
                versions.append(int(m.group(1)))
    for nid in tree["nodes"]:
        m = VN_RE.match(nid)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 0


# --- Mutations ---
def add_node(
    tree: dict,
    *,
    node_id: str,
    parent: Optional[str],
    solution: str,
    description: str,
    tags: list[str],
    evaluation: dict,
    initial_note: Optional[str] = None,
) -> None:
    tree["nodes"][node_id] = {
        "parent": parent,
        "solution": solution,
        "description": description,
        "tags": list(tags),
        "evaluation": evaluation,
        "notes": [initial_note] if initial_note else [],
        "profiling_observations": [],
    }


def set_head(tree: dict, node_id: str) -> None:
    tree["head"] = node_id


def update_bests(tree: dict, node_id: str, evaluation: dict) -> list[str]:
    """Promote node to current_best / representative bests where it wins.
    Returns a list of which pointers were updated. Ranks by speedup when present
    (it cancels same-run machine noise), else by latency (lower = better) — one
    rule, derived per the run's available metric, no stored primary flag."""
    updated: list[str] = []
    nodes = tree["nodes"]

    cur_id = tree["current_best"]
    cur_ev = nodes[cur_id]["evaluation"] if cur_id else None
    if _geomean_better(evaluation, cur_ev):
        tree["current_best"] = node_id
        updated.append("current_best")

    by_rep = tree["best_by_representative_workload"]
    for label, result in evaluation["representative_workloads"].items():
        if result["outcome"] != "PASSED":
            continue
        cur_rid = by_rep.get(label)
        cur_result = (
            nodes[cur_rid]["evaluation"]["representative_workloads"].get(label)
            if cur_rid
            else None
        )
        if _result_better(result, cur_result):
            by_rep[label] = node_id
            updated.append(label)

    return updated


def _geomean_better(new_ev: dict, cur_ev: Optional[dict]) -> bool:
    """Is new_ev's geomean better than cur_ev's? Speedup (higher) if present,
    else latency (lower). A candidate with no scorable geomean never wins."""
    return _metric_better(
        new_ev.get("geomean_speedup_factor"), new_ev.get("geomean_latency_ms"),
        (cur_ev or {}).get("geomean_speedup_factor"), (cur_ev or {}).get("geomean_latency_ms"),
    )


def _result_better(new_result: dict, cur_result: Optional[dict]) -> bool:
    """Same speedup-then-latency rule for one representative workload."""
    return _metric_better(
        new_result.get("speedup_factor"), new_result.get("latency_ms"),
        (cur_result or {}).get("speedup_factor"), (cur_result or {}).get("latency_ms"),
    )


def _metric_better(
    new_speedup: Optional[float], new_latency: Optional[float],
    cur_speedup: Optional[float], cur_latency: Optional[float],
) -> bool:
    """True if the new (speedup, latency) beats the current. Speedup wins when
    the new run reports one (higher better); otherwise latency (lower better).
    Within a run every eval comes from the same adapter, so the metric is
    consistent across candidates."""
    if new_speedup is not None:
        return cur_speedup is None or new_speedup > cur_speedup
    if new_latency is None:
        return False
    return cur_latency is None or new_latency < cur_latency


def _annotation_list(tree: dict, scope: str, experiment_id: Optional[str]) -> list[str]:
    """The mutable annotation list for a scope: a node's notes / profiling_observations
    (per-experiment) or a top-level open_hypotheses / global_facts / hazards list. The
    single place scope names map to storage, so add/edit/render address the same slot."""
    if scope in _NODE_FIELD:
        return tree["nodes"][experiment_id][_NODE_FIELD[scope]]
    return tree[_TOP_LEVEL_KEY[scope]]


def add_annotation(
    tree: dict, scope: str, text: str, experiment_id: Optional[str] = None
) -> None:
    """Append a new entry to the scope's annotation list."""
    _annotation_list(tree, scope, experiment_id).append(text)


def edit_annotation(
    tree: dict,
    scope: str,
    old_text: str,
    new_text: Optional[str],
    experiment_id: Optional[str] = None,
) -> Optional[str]:
    """Replace (new_text given) or remove (new_text None) the single entry whose text
    contains old_text. Returns an error message when zero or several entries match — so
    the caller surfaces it without mutating — otherwise edits in place and returns None."""
    items = _annotation_list(tree, scope, experiment_id)
    matches = [i for i, item in enumerate(items) if old_text in item]
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


PER_EXPERIMENT_SCOPES = tuple(_NODE_FIELD)
TOP_LEVEL_SCOPES = tuple(_TOP_LEVEL_KEY)


# --- Rendering ---
def render_journal(tree: dict) -> str:
    lines: list[str] = ["# Optimization Journal", ""]
    lines.extend(_render_header(tree))
    lines.extend(_render_task_spec(tree))

    lines.append("## Experiments")
    lines.append("")
    nodes = tree["nodes"]
    if not nodes:
        lines.append("_(none)_")
    else:
        target_label, target_ev = _target(tree)
        for nid in _ordered_nodes(nodes):
            lines.extend(_render_node(nid, nodes[nid], target_label, target_ev))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_header(tree: dict) -> list[str]:
    """Metadata, target/best/head, and the curated memory sections — everything
    above the per-experiment list. Reused by render_commit so a commit echoes the
    live top-level state and surfaces empty memory sections at the moment the
    agent has fresh interpretation to record."""
    lines: list[str] = []

    lines.append(f"- **Task:** {tree['task']}")
    if tree["kernel_description"]:
        lines.append(f"- **Kernel:** {tree['kernel_description']}")
    lines.append(f"- **Hardware:** {tree['hardware']}")
    lines.append(f"- **Language:** {tree['language']}")
    axes_by_label = tree.get("representative_workload_axes") or {}
    present = [
        label
        for label in REPRESENTATIVE_WORKLOAD_LABELS
        if axes_by_label.get(label)
    ]
    if present:
        lines.append("- **Representative workloads:**")
        for label in present:
            axes_str = ", ".join(f"{k}={v}" for k, v in axes_by_label[label].items())
            lines.append(f"  - {label}: {axes_str}")
    lines.append("")

    head = tree["head"]
    cur_best = tree["current_best"]
    target_label, target_ev = _target(tree)
    if target_label is not None:
        lines.append(
            f"- **Target:** `{target_label}` — {_evaluation_summary(target_ev)}"
        )
        lines.extend(_representative_workload_lines(target_ev))
    if cur_best:
        ev = tree["nodes"][cur_best]["evaluation"]
        lines.append(
            f"- **Current best:** `{cur_best}` — "
            f"{_evaluation_summary(ev, target_label, target_ev)}"
        )
    else:
        lines.append("- **Current best:** _(unset)_")
    by_rep = tree["best_by_representative_workload"]
    if by_rep:
        bests_str = ", ".join(
            f"{label}=`{by_rep[label]}`"
            for label in REPRESENTATIVE_WORKLOAD_LABELS
            if label in by_rep
        )
        lines.append(f"- **Best by representative workload:** {bests_str}")
    if head:
        suffix = " _(dirty — working kernel has diverged from head)_" if tree["head_state"] == "dirty" else ""
        lines.append(f"- **Head:** `{head}`{suffix}")
    else:
        lines.append("- **Head:** _(unset)_")
    lines.append("")

    for title, key in (
        ("Open hypotheses", "open_hypotheses"),
        ("Global facts", "global_facts"),
        ("Hazards", "hazards"),
    ):
        lines.append(f"## {title}")
        items = tree[key]
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("_(none)_")
        lines.append("")

    return lines


def _render_task_spec(tree: dict) -> list[str]:
    """The full task contract — operation, axes, tensor signatures, constraints,
    and the reference implementation — so the agent can read the complete task
    from the journal alone. Lives only in the full journal, never in the header
    echoed by every commit/checkout, where the reference code would be noise."""
    spec = tree.get("task_spec") or {}
    if not spec:
        return []
    lines = ["## Task specification", ""]

    # The fixed build contract the working kernel must honor (entry symbol,
    # calling convention, available dependencies) — read once, never tuned.
    contract = tree.get("build_contract")
    if contract:
        lines.append(f"- **Build contract:** {contract}")

    if spec.get("op_type"):
        lines.append(f"- **Operation:** {spec['op_type']}")

    # The numerical bar the working kernel must clear to count as correct.
    tolerance = spec.get("tolerance")
    if tolerance:
        desc = _tolerance_desc(tolerance)
        if desc:
            lines.append(f"- **Correctness tolerance:** {desc}")

    # Axes define the workload space: which dims are fixed vs. vary across workloads.
    axes = spec.get("axes") or {}
    if axes:
        lines.append("- **Axes:**")
        for name, axis in axes.items():
            lines.append(f"  - `{name}`: {_axis_desc(axis)}")

    for title, key in (("Inputs", "inputs"), ("Outputs", "outputs")):
        fields = spec.get(key) or {}
        if not fields:
            continue
        lines.append(f"- **{title}:**")
        for name, field in fields.items():
            lines.append(f"  - `{name}`: {_tensor_desc(field)}")

    constraints = spec.get("constraints") or []
    if constraints:
        lines.append("- **Constraints:**")
        for c in constraints:
            lines.append(f"  - {c}")

    # Reference implementation: the ground-truth semantics correctness is checked against.
    reference = spec.get("reference") or ""
    if reference:
        lines.append("- **Reference implementation:**")
        lines.append("")
        lines.append("```python")
        lines.extend(reference.splitlines())
        lines.append("```")

    lines.append("")
    return lines


def _axis_desc(axis: dict) -> str:
    """One axis: a const value, a derived expression, or 'var', plus its optional
    description."""
    kind = axis.get("kind", "var")
    if kind == "const":
        head = f"const = {axis.get('value')}"
    elif kind == "expr":
        expr = axis.get("expression")
        head = f"expr = {expr}" if expr else "expr"
    else:
        head = kind
    desc = axis.get("description")
    return f"{head} — {desc}" if desc else head


def _tolerance_desc(tol: dict) -> str:
    """The correctness bar as one line: the tolerances and the required match
    ratio the adapter surfaced."""
    parts = []
    if tol.get("max_rtol") is not None:
        parts.append(f"rtol {tol['max_rtol']}")
    if tol.get("max_atol") is not None:
        parts.append(f"atol {tol['max_atol']}")
    ratio = tol.get("required_matched_ratio")
    if ratio is not None:
        parts.append(f"≥{ratio:.0%} of elements within tolerance")
    return ", ".join(parts)


def _tensor_desc(field: dict) -> str:
    """One input/output tensor: shape, dtype, optional description. A null shape
    (a scalar argument) renders as 'scalar'."""
    shape = field.get("shape")
    shape_str = f"[{', '.join(map(str, shape))}]" if shape else "scalar"
    base = f"{shape_str} {field.get('dtype', '?')}"
    desc = field.get("description")
    return f"{base} — {desc}" if desc else base


def _render_head_move(tree: dict, node_id: str, lead: str, node_tag: str) -> str:
    """Shared return shape for the tools that move head (log_experiment,
    checkout_experiment): a lead line, the live journal header (top-level state +
    curated memory sections), the node head now points at, and a pointer to the
    full tree. Deliberately not the whole experiment list — that's read_journal."""
    lines = [lead, ""]
    lines.extend(_render_header(tree))

    target_label, target_ev = _target(tree)
    node_lines = _render_node(node_id, tree["nodes"][node_id], target_label, target_ev)
    node_lines[0] += f" — {node_tag}"
    lines.extend(node_lines)
    lines.append("")

    n = len(tree["nodes"])
    lines.append(
        f"_({n} experiment{'s' if n != 1 else ''} total — call read_journal "
        "for the full tree.)_"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_commit(tree: dict, node_id: str) -> str:
    """log_experiment's return value. Echoing the node's recorded evaluation also
    signals that speedup numbers need not be restated in notes."""
    return _render_head_move(tree, node_id, f"Logged {node_id}.", "NEW")


def render_checkout(tree: dict, node_id: str, prev_head: Optional[str], n_files: int) -> str:
    """checkout_experiment's return value. Head and the working kernel just
    changed, so re-ground the agent on the global picture and the experiment it
    now sits on — the same shape a commit returns."""
    lead = (
        f"Checked out {node_id} (head was {prev_head!r}); mirrored {n_files} "
        "file(s) into the working kernel."
    )
    return _render_head_move(tree, node_id, lead, "HEAD")


def render_annotation(tree: dict, scope: str, experiment_id: Optional[str] = None) -> str:
    """annotate_journal's return value: the current contents of the slot just
    appended to, so accumulation (and staleness) is visible at the moment the
    agent adds to it. Lighter than the full header — annotate is an incremental
    act, often called several times in a row."""
    if scope in _NODE_FIELD:
        field = _NODE_FIELD[scope]
        items = tree["nodes"][experiment_id][field]
        lines = [f"`{experiment_id}` {field.replace('_', ' ')}:"]
    else:
        key = _TOP_LEVEL_KEY[scope]
        items = tree[key]
        lines = [f"{key.replace('_', ' ').capitalize()}:"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _ordered_nodes(nodes: dict) -> list[str]:
    """Newest first by vN prefix; non-conforming ids fall back to insertion order."""
    def order(nid: str) -> int:
        m = VN_RE.match(nid)
        return int(m.group(1)) if m else -1
    return sorted(nodes.keys(), key=order, reverse=True)


def _render_node(
    node_id: str, node: dict, target_label: Optional[str], target_ev: Optional[dict]
) -> list[str]:
    ev = node["evaluation"]
    lines = [f"### `{node_id}` ({ev['status']})"]

    parent = node["parent"]
    lines.append(f"- **Parent:** `{parent}`" if parent else "- **Parent:** _(root)_")
    if node["tags"]:
        lines.append(f"- **Tags:** {', '.join(node['tags'])}")
    if node["description"]:
        lines.append(f"- **Description:** {node['description']}")

    lines.append(f"- **Evaluation:** {_evaluation_summary(ev, target_label, target_ev)}")
    lines.extend(_representative_workload_lines(ev, target_label, target_ev))

    lines.append("- **Notes:**")
    if node["notes"]:
        for n in node["notes"]:
            lines.append(f"  - {n}")
    else:
        lines.append("  - _(none)_")
    lines.append("- **Profiling observations:**")
    if node["profiling_observations"]:
        for o in node["profiling_observations"]:
            lines.append(f"  - {o}")
    else:
        lines.append("  - _(none)_")
    return lines



def _target(tree: dict) -> tuple[Optional[str], Optional[dict]]:
    """(label, evaluation) for the target, or (None, None)."""
    tgt = tree.get("target")
    if not tgt:
        return None, None
    return tgt["label"], tgt["evaluation"]


def _evaluation_summary(
    ev: dict,
    target_label: Optional[str] = None,
    target_ev: Optional[dict] = None,
) -> str:
    parts = [ev["status"]]
    count = ev.get("workload_count")
    over = f" over {count} workloads" if count else ""
    geomean = ev.get("geomean_speedup_factor")
    if geomean is not None:
        parts.append(f"geomean {geomean:.2f}× vs reference{over}")
    else:
        latency = ev.get("geomean_latency_ms")
        if latency is not None:
            parts.append(f"geomean {_fmt_ms(latency)}{over}")
    target_ratio = _target_geomean_ratio(ev, target_ev)
    if target_label is not None and target_ratio is not None:
        parts.append(f"{target_ratio:.2f}× vs {target_label}")
    return "; ".join(parts)


def _fmt_ms(ms: float) -> str:
    """Compact latency: milliseconds, dropping trailing precision."""
    return f"{ms:.4g} ms"


def _representative_workload_lines(
    ev: dict,
    target_label: Optional[str] = None,
    target_ev: Optional[dict] = None,
) -> list[str]:
    lines: list[str] = []
    reps = ev["representative_workloads"]
    target_reps = target_ev.get("representative_workloads", {}) if target_ev else {}
    for label in REPRESENTATIVE_WORKLOAD_LABELS:
        if label not in reps:
            continue
        r = reps[label]
        outcome = r["outcome"]
        line = f"  - representative {label}: {outcome}"
        if outcome == "PASSED":
            metrics = []
            spd = r.get("speedup_factor")
            if spd is not None:
                metrics.append(f"{spd:.2f}× vs reference")
            elif r.get("latency_ms") is not None:
                metrics.append(_fmt_ms(r["latency_ms"]))
            target_ratio = _target_representative_workload_ratio(
                r,
                target_reps.get(label),
            )
            if target_label is not None and target_ratio is not None:
                metrics.append(f"{target_ratio:.2f}× vs {target_label}")
            if metrics:
                line += f"; {'; '.join(metrics)}"
        else:
            # The bar only matters next to a miss; on a pass the header's
            # run-wide tolerance already says what was cleared.
            tolerance = r.get("tolerance")
            if tolerance:
                rendered = _format_tolerance(tolerance)
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


def _target_geomean_ratio(ev: dict, target_ev: Optional[dict]) -> Optional[float]:
    if target_ev is None:
        return None
    return _vs_target(
        ev.get("geomean_speedup_factor"), ev.get("geomean_latency_ms"),
        target_ev.get("geomean_speedup_factor"), target_ev.get("geomean_latency_ms"),
    )


def _target_representative_workload_ratio(
    result: dict,
    target_result: Optional[dict],
) -> Optional[float]:
    if target_result is None:
        return None
    return _vs_target(
        result.get("speedup_factor"), result.get("latency_ms"),
        target_result.get("speedup_factor"), target_result.get("latency_ms"),
    )


def _vs_target(
    speedup: Optional[float], latency: Optional[float],
    target_speedup: Optional[float], target_latency: Optional[float],
) -> Optional[float]:
    """Candidate-vs-target ratio where >1 means the candidate is faster.

    When both were normalized against a same-run reference, compare speedups —
    each run measures its own reference beside the candidate, so the ratio is
    less sensitive to cold/throttled/co-located GPU runs. With no reference
    (latency-only), fall back to the cross-run latency ratio
    target_latency / candidate_latency (noisier — a cross-run comparison)."""
    if speedup is not None and target_speedup is not None:
        return _ratio(speedup, target_speedup)
    return _ratio(target_latency, latency)


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator
