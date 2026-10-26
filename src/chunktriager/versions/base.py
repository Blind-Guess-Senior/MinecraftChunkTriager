from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import nbtlib
import numpy as np


class UnsupportedChunkError(Exception):
    """Chunk data does not match the configured version profile."""


@dataclass(frozen=True)
class BlockEntity:
    id: str
    x: int
    y: int
    z: int


@dataclass(frozen=True)
class ChunkInfo:
    x: int
    z: int
    data_version: int
    block_entities: tuple[BlockEntity, ...] = field(default=())


# Version-neutral chunk content consumed by feature extraction. Profiles
# translate their NBT layout into these types.

@dataclass(frozen=True)
class Section:
    y: int                       # section index; blocks span y*16 .. y*16+15
    palette: list[str]
    blocks: np.ndarray           # (16, 16, 16) palette indices, y/z/x order
    biome_palette: list[str]
    biomes: np.ndarray           # (4, 4, 4) palette indices, y/z/x order


@dataclass(frozen=True)
class BlockEntityData:
    id: str
    y: int
    has_loot_table: bool         # unopened structure loot
    has_items: bool


@dataclass(frozen=True)
class EntityData:
    id: str
    x: float
    y: float
    z: float
    named: bool
    leashed: bool


@dataclass(frozen=True)
class ChunkData:
    x: int
    z: int
    full: bool                   # generation finished
    sections: list[Section]
    block_entities: list[BlockEntityData]
    structures: frozenset[str]   # structures with a start or reference here


@dataclass(frozen=True)
class Generator:
    """How a dimension's terrain is generated, as far as labeling cares.

    ``kind`` is ``noise`` (normal terrain), ``flat`` (superflat, including
    void presets) or ``other``. For flat worlds ``layers`` lists
    ``(block, height)`` from the bottom up.
    """

    kind: str
    type: str
    layers: tuple[tuple[str, int], ...] = ()
    biome: str | None = None

    def describe(self) -> str:
        if self.kind != "flat":
            return self.kind if self.kind == "noise" else self.type
        solid = [(b, h) for b, h in self.layers if b != "minecraft:air"]
        if not solid:
            return "void"
        return "flat: " + ", ".join(f"{b.removeprefix('minecraft:')} x{h}" for b, h in solid)


class VersionProfile(ABC):
    #: Value used for ``game_version`` in the configuration.
    name: str
    #: Accepted DataVersion range, ``min <= v < max``.
    data_version_min: int
    data_version_max: int

    def accepts(self, data_version: int) -> bool:
        return self.data_version_min <= data_version < self.data_version_max

    @abstractmethod
    def dimension_dir(self, world: Path, dimension: str) -> Path:
        """Folder holding ``region``/``entities``/``poi`` for a dimension."""

    @abstractmethod
    def read_generator(self, world: Path, dimension: str) -> Generator | None:
        """The dimension's generator, or None if the world does not record one."""

    #: Name of the id vocabulary file in ``versions/data``.
    vocabulary: str

    @abstractmethod
    def parse_chunk(self, root: nbtlib.Compound) -> ChunkInfo:
        """Extract the fields the tool uses from a chunk's root compound."""

    @abstractmethod
    def read_chunk(self, root: nbtlib.Compound) -> ChunkData:
        """Decode a region-file chunk into version-neutral content."""

    @abstractmethod
    def read_entities(self, root: nbtlib.Compound) -> list[EntityData]:
        """Decode an entities-file chunk."""
