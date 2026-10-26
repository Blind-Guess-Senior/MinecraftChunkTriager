"""World-relative features, computed from a whole world's feature table.

Rarity: each evidence column (banded block, block entity, entity) gets a
surprise value ``-log10(presence frequency)`` over the full chunks of the
*same world*. Biome context comes from the biome columns, which the model
combines with rarity itself. (Frequencies taken per dominant biome were
tried and lowered CV AUC from 0.96 to 0.84: in a world built mostly in one
biome, the build itself makes player blocks look common there.)

Rails: player-laid rail lines sit at the surface; mineshaft rails do not.
Rail counts in the above/surface bands separate the two, and also mark
chunks whose only evidence is rails, which protection spreads less far.

Neither group needs labels, so they are computed the same way for training
and prediction, and for any world of the version.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

DERIVED_VERSION = 1
BANDS = ("above", "surface", "shallow", "deep")
CUTS = (3.0, 4.0)                 # surprise thresholds: rarer than 1e-3, 1e-4
RAILS = ("rail", "powered_rail", "detector_rail", "activator_rail")
RAIL_LIKE = RAILS + ("redstone_torch", "redstone_wall_torch")


def _masks(columns: list[str]) -> dict[str, np.ndarray]:
    m = {b: np.array([c.startswith(f"block.{b}.") for c in columns]) for b in BANDS}
    m["be_entity"] = np.array([c.startswith(("be.", "entity.")) for c in columns])
    return m


def _surprise(P: sp.csr_matrix, evidence: np.ndarray) -> np.ndarray:
    freq = np.asarray(P.mean(axis=0)).ravel()
    s = -np.log10(np.maximum(freq, 1.0 / max(1, P.shape[0])))
    return np.where(evidence, s, 0.0).astype(np.float32)


def derived_columns() -> list[str]:
    names = []
    for scope in ("own", "nb"):
        for group in (*BANDS, "be_entity"):
            for cut in CUTS:
                names += [f"rare.{scope}.{group}.kinds_{cut:g}", f"rare.{scope}.{group}.count_{cut:g}"]
            names.append(f"rare.{scope}.{group}.surprise")
        names += [f"rail.{scope}.surface", f"rail.{scope}.torch"]
    names.append("rail.only")
    return names


def compute(own: sp.csr_matrix, neighbours: sp.csr_matrix, columns: list[str]) -> sp.csr_matrix:
    """(n, len(derived_columns())) features for the rows of one world."""
    masks = _masks(columns)
    evidence = np.logical_or.reduce(list(masks.values()))
    P_own = (own > 0).astype(np.float32).tocsr()

    sur = _surprise(P_own, evidence)
    out = np.zeros((own.shape[0], len(derived_columns())), dtype=np.float32)
    col = 0
    for M in (own, neighbours):
        P = (M > 0).astype(np.float32)
        for m in masks.values():
            for cut in CUTS:
                sel = (m & (sur >= cut)).astype(np.float32)
                out[:, col] = P @ sel
                out[:, col + 1] = M @ sel
                col += 2
            out[:, col] = P @ (sur * m)
            col += 1
        col += 2                          # rail columns are filled below
    _rails(own, neighbours, columns, out)
    return sp.csr_matrix(out)


def _rails(own, neighbours, columns, out) -> None:
    def band_sum(M, names, bands):
        sel = np.array([any(c == f"block.{b}.minecraft:{n}" for b in bands for n in names) for c in columns])
        return np.asarray(M[:, np.flatnonzero(sel)].sum(axis=1)).ravel()

    names = derived_columns()
    for scope, M in (("own", own), ("nb", neighbours)):
        out[:, names.index(f"rail.{scope}.surface")] = band_sum(M, RAILS, ("above", "surface"))
        out[:, names.index(f"rail.{scope}.torch")] = band_sum(M, ("redstone_torch", "redstone_wall_torch"), ("above", "surface"))
    # Chunk whose rare surface evidence is only rails and redstone torches.
    rare_kinds = out[:, names.index("rare.own.surface.kinds_3")] + out[:, names.index("rare.own.above.kinds_3")]
    sel = np.array([any(c == f"block.{b}.minecraft:{n}" for b in ("above", "surface") for n in RAIL_LIKE) for c in columns])
    rail_kinds = np.asarray((own[:, np.flatnonzero(sel)] > 0).sum(axis=1)).ravel()
    out[:, names.index("rail.only")] = (out[:, names.index("rail.own.surface")] > 0) & (rare_kinds <= rail_kinds)
