"""Representative workload conventions shared by benchmark-facing tools.

This module is intentionally flashinfer-free. It only knows our user-facing
representative workload labels and how to pick representative positions from
an ordered workload list. Tools pass in native trace/workload objects and keep
their types intact.
"""
from __future__ import annotations

from typing import Callable, Optional


REPRESENTATIVE_WORKLOAD_LABELS = ("small", "medium", "large", "xlarge")


def representative_workload_indexes(n: int) -> dict[str, int]:
    """Map representative labels to indices into a length-n workload list.

    The list is assumed ordered smallest-to-largest. It is split into one
    contiguous, near-equal stratum per label and the last (largest) item of each
    stratum is taken as that label's representative. Taking the last rather than
    the first keeps "small" off the degenerate floor of the list (e.g. a
    batch=1, near-empty-KV decode whose speedup is launch-overhead-bound and
    carries no transferable signal). For n < 4 the smaller labels clamp to the
    last available index.
    """
    if n == 0:
        return {}
    last = n - 1
    b = len(REPRESENTATIVE_WORKLOAD_LABELS)
    if n <= b:
        return {
            label: min(i, last)
            for i, label in enumerate(REPRESENTATIVE_WORKLOAD_LABELS)
        }
    return {
        label: (i + 1) * n // b - 1
        for i, label in enumerate(REPRESENTATIVE_WORKLOAD_LABELS)
    }


def representative_items(
    items: list, result_labels: Optional[list[str]] = None
) -> list[tuple[str, object]]:
    """Pair representative labels with items from a result list.

    Adapter-marked results take precedence over the positional fallback, keeping
    UUID-selected representatives attached to a full benchmark result.
    """
    if result_labels is not None:
        return list(zip(result_labels, items))
    marked = {
        label: item
        for item in items
        if (label := getattr(item, "representative_name", None)) is not None
    }
    if marked:
        return [
            (label, marked[label])
            for label in REPRESENTATIVE_WORKLOAD_LABELS
            if label in marked
        ]
    seen: set[int] = set()
    out: list[tuple[str, object]] = []
    for label, idx in representative_workload_indexes(len(items)).items():
        if idx in seen:
            continue
        seen.add(idx)
        out.append((label, items[idx]))
    return out


def select_representative_workloads(
    items: list,
    representatives: dict[str, str] | None = None,
    item_id: Callable[[object], str] | None = None,
) -> tuple[list, list[str]]:
    """Pick named representative items, by configured ID when supplied.

    The positional fallback supports non-setup workspaces and short test
    fixtures.
    """
    if representatives is not None:
        if item_id is None:
            raise ValueError("item_id is required for configured representatives")
        by_id = {str(item_id(item)): item for item in items}
        missing = [
            f"{label}={workload_id}"
            for label, workload_id in representatives.items()
            if workload_id not in by_id
        ]
        if missing:
            raise ValueError(
                "configured representative workload(s) not found: "
                + ", ".join(missing)
            )
        return (
            [by_id[workload_id] for workload_id in representatives.values()],
            list(representatives),
        )
    pairs = representative_items(items)
    return [item for _, item in pairs], [label for label, _ in pairs]


def representative_item_for_label(
    items: list,
    label: str,
    representatives: dict[str, str] | None = None,
    item_id: Callable[[object], str] | None = None,
):
    """Return the item at a representative label's position (types intact)."""
    if label not in REPRESENTATIVE_WORKLOAD_LABELS:
        valid = ", ".join(REPRESENTATIVE_WORKLOAD_LABELS)
        raise ValueError(
            f"invalid representative_workload {label!r}; expected one of: {valid}"
        )
    if not items:
        raise ValueError("no workloads available")
    if representatives is not None:
        selected, labels = select_representative_workloads(
            items, representatives, item_id
        )
        return dict(zip(labels, selected))[label]
    return _representative_item_by_label(items)[label]


def _representative_item_by_label(items: list) -> dict[str, object]:
    return {
        label: items[idx]
        for label, idx in representative_workload_indexes(len(items)).items()
    }
