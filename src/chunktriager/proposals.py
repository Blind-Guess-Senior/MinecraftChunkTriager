"""Labeling proposals from the world generator.

In a superflat or void world every naturally generated chunk consists of
the generator's layers. A chunk that holds any other block, or any entity,
was most likely changed by players. Such chunks are proposed as
player-made; the labeling page shows them and the user writes them as
labels after checking.

Proposals in chunks that reference a generated structure (e.g. ocean ruins
in a flat ocean preset) are kept apart, since there the extra blocks may
be the structure itself.

Only blocks are compared, not their heights: a build made purely of layer
blocks (say gravel in a gravel-layer world) is not proposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .versions.base import Generator

EXTRA = "e"          # content beyond the generator layers
IN_STRUCTURE = "s"   # the same, in a chunk referencing a generated structure
KINDS = (EXTRA, IN_STRUCTURE)


@dataclass
class Proposals:
    baseline: str                          # generator description
    kind: dict[tuple[int, int], str]       # chunk -> EXTRA or IN_STRUCTURE

    def counts(self) -> dict[str, int]:
        result = dict.fromkeys(KINDS, 0)
        for k in self.kind.values():
            result[k] += 1
        return result


def baseline_blocks(gen: Generator) -> set[str]:
    return {block for block, _ in gen.layers}


def from_features(features_dir: Path, gen: Generator | None) -> Proposals | None:
    """Proposals for a flat world, or None when the generator gives no baseline."""
    from .model.dataset import FeatureSet

    if gen is None or gen.kind != "flat":
        return None
    fs = FeatureSet.load(features_dir)
    base = baseline_blocks(gen)
    cols = fs.columns
    evidence = np.array([
        (c.startswith("block.") and c.split(".", 2)[2] not in base) or c.startswith("entity.minecraft:")
        for c in cols
    ])
    structure = np.array([c.startswith("structure.") for c in cols])
    own = fs.own.tocsc()
    extra = np.asarray(own[:, evidence].sum(axis=1)).ravel() > 0
    in_structure = np.asarray(own[:, structure].sum(axis=1)).ravel() > 0
    full = fs.column("chunk.full") == 1
    kind = {}
    for i in np.flatnonzero(extra & full):
        x, z = fs.positions[i]
        kind[(int(x), int(z))] = IN_STRUCTURE if in_structure[i] else EXTRA
    return Proposals(gen.describe(), kind)
