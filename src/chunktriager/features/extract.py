"""Per-chunk feature rows.

Block counts are split into height bands relative to the ground surface of
each column. Ground is the highest block from the version's surface set
(solid ground, not water, leaves or plants); columns without any surface
block use the lowest section's floor.
"""

from __future__ import annotations

import numpy as np

from ..versions.base import ChunkData, EntityData
from ..versions.vocab import Vocabulary, index
from .schema import BANDS, Layout, band_of

DTYPE = np.float32


class Extractor:
    def __init__(self, vocab: Vocabulary, layout: Layout):
        self.layout = layout
        self.n_solid = len(layout.solid_blocks)
        self.block_idx = index(layout.solid_blocks)
        self.air = vocab.air_blocks
        self.surface = vocab.surface_blocks
        self.biome_idx = index(vocab.biomes)
        self.be_idx = index(vocab.block_entities)
        self.entity_idx = index(vocab.entities)
        self.structure_idx = index(vocab.structures)

    def row(self, chunk: ChunkData, entities: list[EntityData]) -> np.ndarray:
        L = self.layout
        out = np.zeros(len(L.schema.columns), dtype=DTYPE)
        s = L.scalar
        out[s["chunk.full"]] = float(chunk.full)
        if chunk.sections:
            ground = self._ground(chunk)
            out[s["ground.mean"]] = ground.mean()
            out[s["ground.std"]] = ground.std()
            out[s["ground.min"]] = ground.min()
            out[s["ground.max"]] = ground.max()
            out[s["chunk.unknown_blocks"]] = self._blocks(chunk, ground, out)
            self._biomes(chunk, out)
        for be in chunk.block_entities:
            i = self.be_idx.get(be.id)
            if i is not None:
                out[L.block_entity + i] += 1
            out[s["be.loot_table"]] += be.has_loot_table
            out[s["be.with_items"]] += be.has_items
        for e in entities:
            i = self.entity_idx.get(e.id)
            if i is not None:
                out[L.entity + i] += 1
            out[s["entity.named"]] += e.named
            out[s["entity.leashed"]] += e.leashed
        for name in chunk.structures:
            i = self.structure_idx.get(name)
            if i is not None:
                out[L.structure + i] = 1
        return out

    def _ground(self, chunk: ChunkData) -> np.ndarray:
        """(16, 16) y of the highest surface block per column."""
        ground = np.full((16, 16), chunk.sections[0].y * 16, dtype=np.int32)
        found = np.zeros((16, 16), dtype=bool)
        for sec in reversed(chunk.sections):           # top-down
            is_surface = np.array([name in self.surface for name in sec.palette])
            if not is_surface.any():
                continue
            mask = is_surface[sec.blocks]               # (y, z, x)
            hit = mask.any(axis=0) & ~found
            if hit.any():
                top = 15 - np.argmax(mask[::-1], axis=0)  # highest y in section
                ground[hit] = sec.y * 16 + top[hit]
                found |= hit
            if found.all():
                break
        return ground

    def _blocks(self, chunk: ChunkData, ground: np.ndarray, out: np.ndarray) -> int:
        """Add banded block counts to ``out``; returns the count of unknown ids."""
        base = self.layout.block
        unknown = 0
        ys = np.arange(16, dtype=np.int32)[:, None, None]
        for sec in chunk.sections:
            # Map palette entries to vocabulary columns; -1 for air, -2 unknown.
            cols = np.array([
                -1 if name in self.air else self.block_idx.get(name, -2)
                for name in sec.palette
            ], dtype=np.int64)
            if (cols == -1).all():
                continue
            col = cols[sec.blocks]                                   # (16, 16, 16)
            band = band_of(sec.y * 16 + ys - ground[None, :, :])     # broadcast
            keep = col >= 0
            unknown += int((col == -2).sum())
            flat = band[keep].astype(np.int64) * self.n_solid + col[keep]
            counts = np.bincount(flat, minlength=len(BANDS) * self.n_solid)
            out[base:base + counts.size] += counts
        return unknown

    def _biomes(self, chunk: ChunkData, out: np.ndarray) -> None:
        """Fraction of 4x4x4 biome cells per biome, over the whole chunk height."""
        total = 0
        for sec in chunk.sections:
            for i, n in zip(*np.unique(sec.biomes, return_counts=True)):
                j = self.biome_idx.get(sec.biome_palette[i])
                if j is not None:
                    out[self.layout.biome + j] += n
                total += n
        if total:
            out[self.layout.biome:self.layout.biome + len(self.biome_idx)] /= total
