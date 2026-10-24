"""Export delete selections in MCA Selector's CSV format.

Format (MCA Selector ``Selection.readFromFile``): one line per chunk,
``rx;rz;cx;cz`` with absolute chunk coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .worldio import REGION_CHUNKS

if TYPE_CHECKING:
    from .config import Config
    from .store import Store

Pos = tuple[int, int]


@dataclass(frozen=True)
class ExportPlan:
    delete: list[Pos]
    kept_by_label: int
    kept_by_buffer: int
    kept_unlabeled: int


def dilate(positions: Iterable[Pos], radius: int) -> set[Pos]:
    result: set[Pos] = set()
    for x, z in positions:
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                result.add((x + dx, z + dz))
    return result


def plan_export(
    chunks: set[Pos],
    labels: dict[Pos, str],
    buffer: int,
    delete_unlabeled: bool = False,
) -> ExportPlan:
    keep = {p for p, label in labels.items() if label == "keep"}
    protected = dilate(keep, buffer) if buffer > 0 else keep
    delete, by_buffer, unlabeled = [], 0, 0
    for p in sorted(chunks):
        label = labels.get(p)
        if label == "keep":
            continue
        if label is None and not delete_unlabeled:
            unlabeled += 1
            continue
        if p in protected:
            by_buffer += 1
            continue
        delete.append(p)
    return ExportPlan(delete, len(keep & chunks), by_buffer, unlabeled)


def write_selection_csv(path: Path, positions: Iterable[Pos]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cx, cz in positions:
        lines.append(f"{cx // REGION_CHUNKS};{cz // REGION_CHUNKS};{cx};{cz}\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return len(lines)


def export_delete_selection(cfg: Config, store: Store) -> Path:
    """Write the delete selection for the configured dimension; returns the CSV path."""
    from datetime import datetime

    dim = cfg.dimension
    labels = {p: v[0] for p, v in store.labels(dim).items()}
    plan = plan_export(store.chunks(dim), labels, cfg.export.buffer, cfg.export.delete_unlabeled)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_dim = dim.replace(":", "_").replace("/", "_")
    path = cfg.exports_dir / f"delete-{safe_dim}-{stamp}.csv"
    write_selection_csv(path, plan.delete)
    return path
