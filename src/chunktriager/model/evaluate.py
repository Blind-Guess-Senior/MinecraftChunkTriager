"""Metrics that match how predictions are used.

Deletion is conservative: a chunk is kept if any chunk within ``radius`` is
predicted player-made (``protect``). What matters is therefore
  * recall on player-made chunks (a miss can delete part of a build), and
  * the natural false-positive rate, since every false positive protects a
    (2r+1)^2 square of chunks around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


TILE = 256    # chunks per side of the dense grids ``protect`` works on


def protect(positions: np.ndarray, prob: np.ndarray, radius: int) -> np.ndarray:
    """Keep score per chunk: max probability within a (2r+1)^2 square.

    Worked out tile by tile, each tile padded by ``radius`` and filled from
    its neighbour tiles, so memory follows the chunks that exist rather
    than the world's bounding box (worlds can have a few chunks loaded
    near the world border, tens of thousands of chunks from the rest).
    """
    if len(positions) == 0:
        return prob.copy()
    if radius >= TILE:
        raise ValueError(f"radius must be below {TILE}")
    positions = np.asarray(positions, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    tiles = positions // TILE
    by_tile: dict[tuple[int, int], np.ndarray] = {}
    order = np.lexsort((tiles[:, 1], tiles[:, 0]))
    keys, starts = np.unique(tiles[order], axis=0, return_index=True)
    for key, rows in zip(map(tuple, keys.tolist()), np.split(order, starts[1:])):
        by_tile[key] = rows
    out = np.zeros(len(positions), dtype=np.float32)
    size = TILE + 2 * radius
    for (tx, tz), own in by_tile.items():
        origin = np.array([tx * TILE - radius, tz * TILE - radius])
        grid = np.zeros((size, size), dtype=np.float32)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                rows = by_tile.get((tx + dx, tz + dz))
                if rows is None:
                    continue
                ix = positions[rows] - origin
                inside = ((ix >= 0) & (ix < size)).all(axis=1)
                grid[ix[inside, 0], ix[inside, 1]] = prob[rows[inside]]
        spread = maximum_filter(grid, size=2 * radius + 1, mode="constant", cval=0.0)
        ix = positions[own] - origin
        out[own] = spread[ix[:, 0], ix[:, 1]]
    return out


def protect_two_radii(positions: np.ndarray, prob: np.ndarray, short_only: np.ndarray,
                      radius: int, short_radius: int) -> np.ndarray:
    """Like ``protect``, but chunks flagged ``short_only`` spread only ``short_radius``.

    Used for linear builds such as rail lines: a chunk whose only player-made
    evidence is rails protects itself and its close neighbours, not a wide square.
    """
    wide = np.where(short_only, 0.0, prob).astype(np.float32)
    narrow = np.where(short_only, prob, 0.0).astype(np.float32)
    return np.maximum(protect(positions, wide, radius), protect(positions, narrow, short_radius))


@dataclass(frozen=True)
class Threshold:
    threshold: float
    recall: float              # player-made chunks with prob >= threshold
    natural_fpr: float         # natural chunks with prob >= threshold
    false_positives: int


def at_recall(y: np.ndarray, prob: np.ndarray, targets=(0.9, 0.95, 0.99, 1.0)) -> list[Threshold]:
    """For each recall target, the highest threshold reaching it and its natural FPR."""
    pos = np.sort(prob[y == 1])[::-1]
    neg = prob[y == 0]
    result = []
    for r in targets:
        need = max(1, int(np.ceil(r * len(pos))))
        t = float(pos[need - 1])
        fp = int((neg >= t).sum())
        result.append(Threshold(t, float((prob[y == 1] >= t).mean()), fp / max(1, len(neg)), fp))
    return result


# Reports used by `chunktriager evaluate` -------------------------------------------

THRESHOLDS = (3e-2, 1e-2, 3e-3)


def _usable(w):
    use = w.y >= 0
    return w.X[use], w.y[use].astype(np.int32)


def _print_protection(positions, prob, rail_only, y, prot) -> None:
    for t in sorted({*THRESHOLDS, prot.threshold}, reverse=True):
        hit = prob >= t
        kept = protect_two_radii(positions, hit.astype(np.float32), rail_only & hit,
                                 prot.radius, prot.short_radius) > 0
        made = f"player-made protected {kept[y == 1].mean():.3f}, " if (y == 1).any() else ""
        mark = "  <- threshold" if t == prot.threshold else ""
        print(f"    t={t:.0e}: {made}deletable {(~kept).mean():.3f}, "
              f"natural flagged {int((hit & (y == 0)).sum())} ({(hit & (y == 0)).sum() / max(1, (y == 0).sum()):.4f}){mark}")


def report_grouped_cv(worlds, training, prot) -> None:
    """Site-grouped CV per train world; the other train worlds join every fold's training set."""
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score

    from .train import assign_folds, fit, site_groups

    print(f"cross-validation on train worlds ({training.folds} folds, grouped by site)")
    for w in worlds:
        use = np.flatnonzero(w.y >= 0)
        y = w.y[use].astype(np.int32)
        if not (y == 1).any():
            print(f"  {w.name}: no player-made chunks, used as training data only")
            continue
        folds = assign_folds(site_groups(w.positions[use], y), y, training.folds)
        extra = [_usable(o) for o in worlds if o is not w]
        oof = np.zeros(len(use))
        for f in range(training.folds):
            te = folds == f
            if not te.any():
                continue
            model = fit(sp.vstack([w.X[use][~te], *(X for X, _ in extra)], format="csr"),
                        np.concatenate([y[~te], *(y_ for _, y_ in extra)]),
                        training.rounds, training.params)
            oof[te] = model.predict(w.X[use][te])
        print(f"  {w.name}: ROC AUC {roc_auc_score(y, oof):.4f}")
        _print_protection(w.positions[use], oof, w.rail_only[use], y, prot)


def report_held_out(train, held, training, prot) -> None:
    """Fit on all train worlds, score each held-out world."""
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score

    from .train import fit

    parts = [_usable(w) for w in train]
    model = fit(sp.vstack([X for X, _ in parts], format="csr"), np.concatenate([y for _, y in parts]),
                training.rounds, training.params)
    print("held-out worlds (model fitted on all train worlds)")
    for w in held:
        use = np.flatnonzero(w.y >= 0)
        y = w.y[use].astype(np.int32)
        p = model.predict(w.X[use])
        auc = f", ROC AUC {roc_auc_score(y, p):.4f}" if 0 < y.sum() < len(y) else ", all natural"
        print(f"  {w.name}: {len(y)} chunks, {int(y.sum())} player-made{auc}")
        _print_protection(w.positions[use], p, w.rail_only[use], y, prot)
