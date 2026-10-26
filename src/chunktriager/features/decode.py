"""Decode paletted section data into numpy arrays.

Since 1.16 the packed long arrays do not split an entry across two longs:
each long holds ``64 // bits`` entries starting at the low bit, and the
leftover high bits are padding.
"""

from __future__ import annotations

import numpy as np

SECTION_BLOCKS = 4096
SECTION_BIOMES = 64


def unpack(data, bits: int, count: int) -> np.ndarray:
    """Unpack ``count`` palette indices of ``bits`` bits from a long array."""
    longs = np.asarray(data, dtype=">i8").astype(np.int64).view(np.uint64)
    per_long = 64 // bits
    shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
    mask = np.uint64((1 << bits) - 1)
    values = (longs[:, None] >> shifts[None, :]) & mask
    return values.reshape(-1)[:count].astype(np.int32)


def block_indices(block_states) -> tuple[list[str], np.ndarray]:
    """Palette names and a (16, 16, 16) array of palette indices in y, z, x order."""
    palette = [str(entry["Name"]) for entry in block_states["palette"]]
    data = block_states.get("data")
    if data is None or len(palette) == 1:
        return palette, np.zeros((16, 16, 16), dtype=np.int32)
    bits = max(4, (len(palette) - 1).bit_length())
    return palette, unpack(data, bits, SECTION_BLOCKS).reshape(16, 16, 16)


def biome_indices(biomes) -> tuple[list[str], np.ndarray]:
    """Palette names and a (4, 4, 4) array of palette indices in y, z, x order."""
    palette = [str(name) for name in biomes["palette"]]
    data = biomes.get("data")
    if data is None or len(palette) == 1:
        return palette, np.zeros((4, 4, 4), dtype=np.int32)
    bits = (len(palette) - 1).bit_length()
    return palette, unpack(data, bits, SECTION_BIOMES).reshape(4, 4, 4)
