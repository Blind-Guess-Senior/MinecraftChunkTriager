"""Feature schema: the fixed list of per-chunk feature columns.

Columns depend only on the vocabulary and the constants below, never on the
world being processed, so every world of a version yields the same columns.
Bump ``SCHEMA_VERSION`` whenever a column's meaning changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..versions.vocab import Vocabulary

SCHEMA_VERSION = 1

# Height bands relative to the ground surface of each block column
# (``ground`` = y of the highest surface block). Bounds are inclusive offsets.
SURFACE_HALF_WIDTH = 7
SHALLOW_DEPTH = 32
BANDS = ("above", "surface", "shallow", "deep")


def band_of(dy):
    """Band index for offsets ``dy = y - ground``; works on numpy arrays."""
    import numpy as np

    return np.select(
        [dy > SURFACE_HALF_WIDTH, dy >= -SURFACE_HALF_WIDTH, dy >= -SHALLOW_DEPTH],
        [0, 1, 2],
        default=3,
    )


@dataclass(frozen=True)
class Schema:
    vocab_hash: str
    columns: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        text = f"{SCHEMA_VERSION}|{self.vocab_hash}|" + ",".join(self.columns)
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# Scalar columns that do not come from a vocabulary.
SCALARS = (
    "chunk.full",                 # generation finished
    "chunk.unknown_blocks",       # ids missing from the vocabulary; should be 0
    "ground.mean", "ground.std", "ground.min", "ground.max",
    "be.loot_table",              # block entities with an unopened loot table
    "be.with_items",              # containers holding items
    "entity.named",
    "entity.leashed",
)


@dataclass(frozen=True)
class Layout:
    """Column offsets of each feature group, for building rows quickly."""

    schema: Schema
    solid_blocks: tuple[str, ...]      # vocabulary blocks except air
    block: int                         # start of block.<band>.<name>
    biome: int
    block_entity: int
    entity: int
    structure: int
    scalar: dict[str, int]


def build_layout(vocab: Vocabulary) -> Layout:
    # "Added, not removed": air is not counted, so digging out space does
    # not produce block features.
    solid = tuple(b for b in vocab.blocks if b not in vocab.air_blocks)
    cols: list[str] = []
    block = len(cols)
    cols += [f"block.{band}.{name}" for band in BANDS for name in solid]
    biome = len(cols)
    cols += [f"biome.{name}" for name in vocab.biomes]
    block_entity = len(cols)
    cols += [f"be.{name}" for name in vocab.block_entities]
    entity = len(cols)
    cols += [f"entity.{name}" for name in vocab.entities]
    structure = len(cols)
    cols += [f"structure.{name}" for name in vocab.structures]
    scalar = {}
    for name in SCALARS:
        scalar[name] = len(cols)
        cols.append(name)
    return Layout(Schema(vocab.hash, tuple(cols)), solid, block, biome,
                  block_entity, entity, structure, scalar)
