"""SQLite storage for the chunk index and labels.

A label is a final decision, either ``keep`` or ``delete``. Chunks without
a label are simply not decided yet. Every change is appended to
``label_log`` grouped by batch, which backs undo.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

LABELS = ("keep", "delete")
SOURCES = ("single", "area", "model")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    dim TEXT NOT NULL, x INTEGER NOT NULL, z INTEGER NOT NULL,
    PRIMARY KEY (dim, x, z)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS labels (
    dim TEXT NOT NULL, x INTEGER NOT NULL, z INTEGER NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('keep', 'delete')),
    source TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (dim, x, z)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS label_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch INTEGER NOT NULL,
    dim TEXT NOT NULL, x INTEGER NOT NULL, z INTEGER NOT NULL,
    old_label TEXT, old_source TEXT,
    new_label TEXT, new_source TEXT,
    at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS label_log_batch ON label_log (batch);
"""


class StoreError(Exception):
    pass


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class Store:
    def __init__(self, path: Path):
        self.conn = connect(path)

    def close(self) -> None:
        self.conn.close()

    # chunk index -----------------------------------------------------------

    def replace_chunks(self, dim: str, positions: Iterable[tuple[int, int]]) -> int:
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE dim = ?", (dim,))
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO chunks (dim, x, z) VALUES (?, ?, ?)",
                ((dim, x, z) for x, z in positions),
            )
        return cur.rowcount

    def chunks(self, dim: str) -> set[tuple[int, int]]:
        rows = self.conn.execute("SELECT x, z FROM chunks WHERE dim = ?", (dim,))
        return {(x, z) for x, z in rows}

    def chunk_exists(self, dim: str, x: int, z: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM chunks WHERE dim = ? AND x = ? AND z = ?", (dim, x, z)
        ).fetchone()
        return row is not None

    # labels ----------------------------------------------------------------

    def labels(
        self, dim: str, bbox: tuple[int, int, int, int] | None = None
    ) -> dict[tuple[int, int], tuple[str, str]]:
        """``{(x, z): (label, source)}``, optionally limited to an inclusive bbox."""
        sql = "SELECT x, z, label, source FROM labels WHERE dim = ?"
        params: list[object] = [dim]
        if bbox is not None:
            sql += " AND x BETWEEN ? AND ? AND z BETWEEN ? AND ?"
            params += [bbox[0], bbox[2], bbox[1], bbox[3]]
        return {(x, z): (label, source) for x, z, label, source in self.conn.execute(sql, params)}

    def label_counts(self, dim: str) -> dict[str, int]:
        counts = dict.fromkeys(LABELS, 0)
        rows = self.conn.execute(
            "SELECT label, COUNT(*) FROM labels WHERE dim = ? GROUP BY label", (dim,)
        )
        counts.update(dict(rows))
        return counts

    def set_labels(
        self,
        dim: str,
        positions: Iterable[tuple[int, int]],
        label: str | None,
        source: str,
        known: set[tuple[int, int]] | None = None,
    ) -> int:
        """Set (or clear, with ``label=None``) labels as one undoable batch.

        Positions that are not in the chunk index are ignored. Returns the
        number of chunks whose label actually changed.
        """
        if label is not None and label not in LABELS:
            raise StoreError(f"invalid label {label!r}")
        if source not in SOURCES:
            raise StoreError(f"invalid source {source!r}")
        if known is None:
            known = self.chunks(dim)
        targets = sorted({p for p in positions if p in known})
        if not targets:
            return 0
        current = self.labels(dim)
        now = time.time()
        with self.conn:
            batch = self._next_batch()
            changed = 0
            for x, z in targets:
                old = current.get((x, z))
                new = (label, source) if label is not None else None
                if old == new or (old is None and new is None):
                    continue
                self.conn.execute(
                    "INSERT INTO label_log (batch, dim, x, z, old_label, old_source,"
                    " new_label, new_source, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (batch, dim, x, z, *(old or (None, None)), *(new or (None, None)), now),
                )
                self._write(dim, x, z, new, now)
                changed += 1
        return changed

    def undo(self, dim: str) -> int:
        """Revert the most recent batch for ``dim``; returns chunks reverted."""
        row = self.conn.execute(
            "SELECT MAX(batch) FROM label_log WHERE dim = ?", (dim,)
        ).fetchone()
        if row[0] is None:
            return 0
        batch = row[0]
        entries = self.conn.execute(
            "SELECT x, z, old_label, old_source FROM label_log WHERE batch = ? AND dim = ?",
            (batch, dim),
        ).fetchall()
        now = time.time()
        with self.conn:
            for x, z, old_label, old_source in entries:
                old = (old_label, old_source) if old_label is not None else None
                self._write(dim, x, z, old, now)
            self.conn.execute("DELETE FROM label_log WHERE batch = ? AND dim = ?", (batch, dim))
        return len(entries)

    def _next_batch(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM label_log").fetchone()
        return row[0]

    def _write(self, dim: str, x: int, z: int, value: tuple[str, str] | None, now: float) -> None:
        if value is None:
            self.conn.execute("DELETE FROM labels WHERE dim = ? AND x = ? AND z = ?", (dim, x, z))
        else:
            self.conn.execute(
                "INSERT INTO labels (dim, x, z, label, source, updated_at) VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (dim, x, z) DO UPDATE SET label = excluded.label,"
                " source = excluded.source, updated_at = excluded.updated_at",
                (dim, x, z, value[0], value[1], now),
            )
