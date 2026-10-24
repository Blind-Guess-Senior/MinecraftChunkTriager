"""Read-only access to Anvil region files (``r.<x>.<z>.mca``).

The region header's timestamp table is intentionally ignored: it records
history, which the project does not use.
"""

from __future__ import annotations

import gzip
import io
import re
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import nbtlib

SECTOR = 4096
REGION_CHUNKS = 32
_REGION_NAME = re.compile(r"^r\.(-?\d+)\.(-?\d+)\.mca$")

COMPRESSION_GZIP = 1
COMPRESSION_ZLIB = 2
COMPRESSION_NONE = 3
COMPRESSION_LZ4 = 4
_EXTERNAL_FLAG = 128


class RegionFormatError(Exception):
    pass


@dataclass(frozen=True, order=True)
class RegionPos:
    x: int
    z: int

    @property
    def filename(self) -> str:
        return f"r.{self.x}.{self.z}.mca"


def parse_region_name(name: str) -> RegionPos | None:
    m = _REGION_NAME.match(name)
    return RegionPos(int(m.group(1)), int(m.group(2))) if m else None


def list_regions(region_dir: Path) -> list[RegionPos]:
    """Region files that contain at least one chunk."""
    if not region_dir.is_dir():
        return []
    result = []
    for path in region_dir.iterdir():
        pos = parse_region_name(path.name)
        if pos is not None and path.stat().st_size > 2 * SECTOR:
            result.append(pos)
    return sorted(result)


def _decompress(kind: int, data: bytes, external: Path | None) -> bytes:
    if kind & _EXTERNAL_FLAG:
        if external is None or not external.is_file():
            raise RegionFormatError(f"external chunk file {external} not found")
        data = external.read_bytes()
        kind &= ~_EXTERNAL_FLAG
    if kind == COMPRESSION_ZLIB:
        return zlib.decompress(data)
    if kind == COMPRESSION_GZIP:
        return gzip.decompress(data)
    if kind == COMPRESSION_NONE:
        return data
    if kind == COMPRESSION_LZ4:
        raise RegionFormatError("LZ4 chunk compression is not supported yet")
    raise RegionFormatError(f"unknown chunk compression type {kind}")


class RegionFile:
    def __init__(self, path: Path):
        self.path = path
        pos = parse_region_name(path.name)
        if pos is None:
            raise RegionFormatError(f"not a region file name: {path.name}")
        self.pos = pos
        self._data = path.read_bytes()
        if len(self._data) < 2 * SECTOR:
            raise RegionFormatError(f"{path} is shorter than the region header")

    def _location(self, index: int) -> tuple[int, int]:
        entry = int.from_bytes(self._data[index * 4 : index * 4 + 4], "big")
        return entry >> 8, entry & 0xFF

    def chunk_positions(self) -> list[tuple[int, int]]:
        """Absolute chunk coordinates of all chunks present in this region."""
        result = []
        for index in range(REGION_CHUNKS * REGION_CHUNKS):
            offset, count = self._location(index)
            if offset and count:
                lx, lz = index % REGION_CHUNKS, index // REGION_CHUNKS
                result.append((self.pos.x * REGION_CHUNKS + lx, self.pos.z * REGION_CHUNKS + lz))
        return result

    def read_chunk(self, cx: int, cz: int) -> nbtlib.Compound | None:
        index = (cx % REGION_CHUNKS) + (cz % REGION_CHUNKS) * REGION_CHUNKS
        offset, count = self._location(index)
        if not offset or not count:
            return None
        start = offset * SECTOR
        length = int.from_bytes(self._data[start : start + 4], "big")
        if length < 1 or start + 4 + length > len(self._data):
            raise RegionFormatError(f"chunk ({cx}, {cz}) in {self.path} is truncated")
        kind = self._data[start + 4]
        payload = self._data[start + 5 : start + 4 + length]
        external = self.path.with_name(f"c.{cx}.{cz}.mcc")
        raw = _decompress(kind, payload, external)
        return nbtlib.File.parse(io.BytesIO(raw))

    def iter_chunks(self) -> Iterator[tuple[int, int, nbtlib.Compound]]:
        for cx, cz in self.chunk_positions():
            root = self.read_chunk(cx, cz)
            if root is not None:
                yield cx, cz, root
