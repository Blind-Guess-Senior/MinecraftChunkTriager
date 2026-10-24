from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import nbtlib


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


class VersionProfile(ABC):
    #: Value used for ``game_version`` in the workspace config.
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
    def parse_chunk(self, root: nbtlib.Compound) -> ChunkInfo:
        """Extract the fields the tool uses from a chunk's root compound."""
