"""Representative workload conventions shared by benchmark-facing tools.

This module is intentionally flashinfer-free. It only knows our user-facing
representative workload labels and how to pick representative positions from
an ordered workload list. Tools pass in native trace/workload objects and keep
their types intact.
"""
from __future__ import annotations

from typing import Optional


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
    """Pair representative labels with items from a result list."""
    if result_labels is not None:
        return list(zip(result_labels, items))
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
) -> tuple[list, list[str]]:
    """Pick the representative items for the fixed labels (types intact).

    Returns (selected_items, labels) lined up one-for-one. For short lists,
    labels that collapse onto the same position are deduped, so the result may
    contain fewer than 4 entries.
    """
    pairs = representative_items(items)
    return [item for _, item in pairs], [label for label, _ in pairs]


def representative_item_for_label(items: list, label: str):
    """Return the item at a representative label's position (types intact)."""
    if label not in REPRESENTATIVE_WORKLOAD_LABELS:
        valid = ", ".join(REPRESENTATIVE_WORKLOAD_LABELS)
        raise ValueError(
            f"invalid representative_workload {label!r}; expected one of: {valid}"
        )
    if not items:
        raise ValueError("no workloads available")
    return _representative_item_by_label(items)[label]


def _representative_item_by_label(items: list) -> dict[str, object]:
    return {
        label: items[idx]
        for label, idx in representative_workload_indexes(len(items)).items()
    }
