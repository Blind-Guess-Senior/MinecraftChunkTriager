"""Write chunk selections in MCA Selector's CSV format.

Format (MCA Selector ``Selection.readFromFile``): one line per chunk,
``rx;rz;cx;cz`` with absolute chunk coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .worldio import REGION_CHUNKS

Pos = tuple[int, int]


def write_selection_csv(path: Path, positions: Iterable[Pos]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cx, cz in positions:
        lines.append(f"{cx // REGION_CHUNKS};{cz // REGION_CHUNKS};{cx};{cz}\n")
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return len(lines)

