"""Java Edition 26.2 profile.

DataVersion bounds follow MCA Selector's version table: 26.2 snapshot 1 is
4883 and 26.3 snapshot 1 is 4998. Since 26.1 snapshot 6 dimensions live
under ``dimensions/<namespace>/<path>``.
"""

from __future__ import annotations

from pathlib import Path

import nbtlib

from ..features.decode import biome_indices, block_indices
from .base import (
    BlockEntity,
    BlockEntityData,
    ChunkData,
    Generator,
    ChunkInfo,
    EntityData,
    Section,
    UnsupportedChunkError,
    VersionProfile,
)


class Java26_2(VersionProfile):
    name = "java-26.2"
    data_version_min = 4883
    data_version_max = 4998
    vocabulary = "java-26.2"

    def dimension_dir(self, world: Path, dimension: str) -> Path:
        namespace, sep, path = dimension.partition(":")
        if not sep or not namespace or not path:
            raise ValueError(f"dimension must look like 'namespace:path', got {dimension!r}")
        return world / "dimensions" / namespace / path

    def read_generator(self, world: Path, dimension: str) -> Generator | None:
        # Since 26.1 world generation settings have their own file.
        path = world / "data" / "minecraft" / "world_gen_settings.dat"
        if not path.is_file():
            return None
        root = nbtlib.load(path)
        root = root.get("data", root)
        gen = root.get("dimensions", {}).get(dimension, {}).get("generator")
        if gen is None:
            return None
        kind = str(gen.get("type", ""))
        if kind == "minecraft:noise":
            return Generator("noise", kind)
        if kind == "minecraft:flat":
            settings = gen.get("settings", {})
            layers = tuple((str(l["block"]), int(l["height"])) for l in settings.get("layers", []))
            biome = settings.get("biome")
            return Generator("flat", kind, layers, str(biome) if biome is not None else None)
        return Generator("other", kind)

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


    def read_chunk(self, root: nbtlib.Compound) -> ChunkData:
        info = self.parse_chunk(root)        # validates DataVersion
        sections = []
        for sec in root.get("sections", []):
            if "block_states" not in sec or "biomes" not in sec:
                continue
            palette, blocks = block_indices(sec["block_states"])
            biome_palette, biomes = biome_indices(sec["biomes"])
            sections.append(Section(int(sec["Y"]), palette, blocks, biome_palette, biomes))
        block_entities = [
            BlockEntityData(
                str(be["id"]), int(be["y"]),
                has_loot_table="LootTable" in be,
                has_items=len(be.get("Items", [])) > 0,
            )
            for be in root.get("block_entities", [])
        ]
        structures = root.get("structures", {})
        names = set(structures.get("starts", {}).keys()) | set(structures.get("References", {}).keys())
        return ChunkData(
            info.x, info.z,
            full=str(root.get("Status", "")) == "minecraft:full",
            sections=sorted(sections, key=lambda s: s.y),
            block_entities=block_entities,
            structures=frozenset(names),
        )

    def read_entities(self, root: nbtlib.Compound) -> list[EntityData]:
        result = []
        for e in root.get("Entities", []):
            pos = e.get("Pos", [0.0, 0.0, 0.0])
            result.append(EntityData(
                str(e["id"]), float(pos[0]), float(pos[1]), float(pos[2]),
                named="CustomName" in e,
                leashed="leash" in e or "Leash" in e,
            ))
        return result


PROFILE = Java26_2()
