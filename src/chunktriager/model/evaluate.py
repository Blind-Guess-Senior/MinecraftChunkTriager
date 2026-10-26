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


def protect(positions: np.ndarray, prob: np.ndarray, radius: int) -> np.ndarray:
    """Keep score per chunk: max probability within a (2r+1)^2 square."""
    if len(positions) == 0:
        return prob.copy()
    lo = positions.min(axis=0)
    shape = positions.max(axis=0) - lo + 1
    grid = np.zeros(shape, dtype=np.float32)
    ix = positions - lo
    grid[ix[:, 0], ix[:, 1]] = prob
    spread = maximum_filter(grid, size=2 * radius + 1, mode="constant", cval=0.0)
    return spread[ix[:, 0], ix[:, 1]]


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
