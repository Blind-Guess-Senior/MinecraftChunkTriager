"""Training matrices built from feature tables and labels.

Row features are the chunk's own columns. Neighbourhood features add, for
every column, the sum over the 8 surrounding chunks, so a building that
crosses a chunk edge is visible from both sides. Chunks outside the world
(not generated) contribute zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..features.store import read_manifest, read_table

NEIGHBOURS = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dz) != (0, 0)]


@dataclass
class FeatureSet:
    positions: np.ndarray            # (n, 2) chunk x, z
    own: sp.csr_matrix               # (n, c) own features
    columns: list[str]
    fingerprint: str
    game_version: str
    index: dict[tuple[int, int], int]

    @classmethod
    def load(cls, directory: Path) -> FeatureSet:
        manifest = read_manifest(directory)
        positions, (indptr, indices, data), n_cols = read_table(directory)
        own = sp.csr_matrix((data, indices, indptr), shape=(len(positions), n_cols))
        index = {(int(x), int(z)): i for i, (x, z) in enumerate(positions)}
        return cls(positions, own, manifest["columns"], manifest["fingerprint"],
                   manifest["game_version"], index)

    def column(self, name: str) -> np.ndarray:
        return self.own[:, self.columns.index(name)].toarray().ravel()

    def neighbour_sum(self) -> sp.csr_matrix:
        """(n, c) sum of the 8 neighbours' own features."""
        n = len(self.positions)
        rows, cols = [], []
        for dx, dz in NEIGHBOURS:
            for i, (x, z) in enumerate(self.positions):
                j = self.index.get((int(x) + dx, int(z) + dz))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        adjacency = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
        return (adjacency @ self.own).tocsr()

    def matrix(self, neighbours: bool = True) -> tuple[sp.csr_matrix, list[str]]:
        if not neighbours:
            return self.own, list(self.columns)
        X = sp.hstack([self.own, self.neighbour_sum()], format="csr")
        return X, list(self.columns) + [f"nb.{c}" for c in self.columns]


@dataclass
class World:
    """One workspace's full chunks, ready for training or prediction."""

    name: str
    positions: np.ndarray        # (n, 2)
    X: sp.csr_matrix             # own + neighbour + derived features
    y: np.ndarray | None         # 1 player-made, 0 natural, -1 not usable; None if unlabeled world
    rail_only: np.ndarray        # chunks whose only evidence is rails
    columns: list[str]
    fingerprint: str


def load_world(features_dir: Path, labels: dict[tuple[int, int], str] | None,
               unlabeled: str = "natural", name: str = "") -> World:
    from .derived import compute, derived_columns

    fs = FeatureSet.load(features_dir)
    rows = np.flatnonzero(fs.column("chunk.full") == 1)
    own = fs.own[rows].tocsr()
    nb = fs.neighbour_sum()[rows].tocsr()
    derived = compute(own, nb, fs.columns)
    X = sp.hstack([own, nb, derived], format="csr")
    columns = list(fs.columns) + [f"nb.{c}" for c in fs.columns] + derived_columns()
    pos = fs.positions[rows]
    y = None
    if labels is not None:
        default = 0 if unlabeled == "natural" else -1
        code = {"keep": 1, "delete": 0}
        y = np.array([code.get(labels.get((int(x), int(z))), default) for x, z in pos], dtype=np.int8)
    rail_only = derived[:, derived_columns().index("rail.only")].toarray().ravel() > 0
    return World(name or features_dir.parent.parent.name, pos, X, y, rail_only, columns, fs.fingerprint)
