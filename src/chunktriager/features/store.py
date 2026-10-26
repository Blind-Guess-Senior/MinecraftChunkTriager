"""On-disk feature tables.

One Parquet file per region under ``<workspace>/features/<schema fingerprint>/``.
Rows are sparse (most of the ~5000 columns are zero for any chunk): each
row stores the indices and values of its non-zero columns. ``manifest.json``
next to the files records the column names, so a table can be read without
the code that wrote it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import SCHEMA_VERSION, Schema

ARROW_SCHEMA = pa.schema([
    ("x", pa.int32()),
    ("z", pa.int32()),
    ("idx", pa.list_(pa.int32())),
    ("val", pa.list_(pa.float32())),
])


def table_dir(workspace: Path, schema: Schema) -> Path:
    return workspace / "features" / schema.fingerprint


def write_manifest(directory: Path, schema: Schema, game_version: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": schema.fingerprint,
        "vocab_hash": schema.vocab_hash,
        "game_version": game_version,
        "columns": list(schema.columns),
    }
    (directory / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")


def write_region(path: Path, positions: list[tuple[int, int]], rows: list[np.ndarray]) -> None:
    idx, val = [], []
    for row in rows:
        nz = np.flatnonzero(row)
        idx.append(nz.astype(np.int32))
        val.append(row[nz].astype(np.float32))
    table = pa.table({
        "x": pa.array([p[0] for p in positions], pa.int32()),
        "z": pa.array([p[1] for p in positions], pa.int32()),
        "idx": pa.array(idx, pa.list_(pa.int32())),
        "val": pa.array(val, pa.list_(pa.float32())),
    }, schema=ARROW_SCHEMA)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def read_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def read_table(directory: Path):
    """Load all regions as ``(positions, matrix)``.

    ``positions`` is an (n, 2) int array of chunk x, z; ``matrix`` is a
    scipy-free CSR triple ``(indptr, indices, data)`` plus the column count,
    which callers turn into whatever sparse type their model wants.
    """
    n_cols = len(read_manifest(directory)["columns"])
    xs, zs, indptr, indices, data = [], [], [0], [], []
    for path in sorted(directory.glob("r.*.parquet")):
        t = pq.read_table(path)
        xs.append(t.column("x").to_numpy())
        zs.append(t.column("z").to_numpy())
        for i, v in zip(t.column("idx").to_pylist(), t.column("val").to_pylist()):
            indices.extend(i)
            data.extend(v)
            indptr.append(len(indices))
    positions = np.stack([np.concatenate(xs), np.concatenate(zs)], axis=1) if xs else np.zeros((0, 2), np.int32)
    csr = (np.asarray(indptr, np.int64), np.asarray(indices, np.int32), np.asarray(data, np.float32))
    return positions, csr, n_cols
