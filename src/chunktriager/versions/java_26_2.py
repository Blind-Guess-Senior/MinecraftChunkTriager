"""Java Edition 26.2 profile.

DataVersion bounds follow MCA Selector's version table: 26.2 snapshot 1 is
4883 and 26.3 snapshot 1 is 4998. Since 26.1 snapshot 6 dimensions live
under ``dimensions/<namespace>/<path>``.
"""

from __future__ import annotations

from pathlib import Path

import nbtlib

from .base import BlockEntity, ChunkInfo, UnsupportedChunkError, VersionProfile


class Java26_2(VersionProfile):
    name = "java-26.2"
    data_version_min = 4883
    data_version_max = 4998

    def dimension_dir(self, world: Path, dimension: str) -> Path:
        namespace, sep, path = dimension.partition(":")
        if not sep or not namespace or not path:
            raise ValueError(f"dimension must look like 'namespace:path', got {dimension!r}")
        return world / "dimensions" / namespace / path

    def parse_chunk(self, root: nbtlib.Compound) -> ChunkInfo:
        try:
            data_version = int(root["DataVersion"])
            x, z = int(root["xPos"]), int(root["zPos"])
        except KeyError as e:
            raise UnsupportedChunkError(f"chunk is missing {e.args[0]!r}") from None
        if not self.accepts(data_version):
            raise UnsupportedChunkError(
                f"chunk ({x}, {z}) has DataVersion {data_version}, outside "
                f"[{self.data_version_min}, {self.data_version_max}) for {self.name}; "
                "run forceUpgrade on the world copy"
            )
        entities = tuple(
            BlockEntity(str(be["id"]), int(be["x"]), int(be["y"]), int(be["z"]))
            for be in root.get("block_entities", [])
        )
        return ChunkInfo(x, z, data_version, entities)


PROFILE = Java26_2()
