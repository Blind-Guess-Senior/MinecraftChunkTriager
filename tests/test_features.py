import io
from pathlib import Path

import nbtlib
import numpy as np
import pytest
from nbtlib import Compound, Double, List, String

from chunktriager.features.decode import unpack
from chunktriager.features.extract import Extractor
from chunktriager.features.pipeline import extract_dimension
from chunktriager.features.schema import build_layout
from chunktriager.features.store import read_manifest, read_table
from chunktriager.versions import get_profile
from chunktriager.versions.vocab import load_vocabulary

from .conftest import full_chunk_nbt, pack, section_nbt, write_region

PROFILE = get_profile("java-26.2")
VOCAB = load_vocabulary(PROFILE.vocabulary)
LAYOUT = build_layout(VOCAB)
COL = {name: i for i, name in enumerate(LAYOUT.schema.columns)}


def parse(raw: bytes):
    return nbtlib.File.parse(io.BytesIO(raw))


@pytest.mark.parametrize("bits", [1, 4, 5, 7, 13])
def test_unpack_roundtrip(bits):
    rng = np.random.default_rng(bits)
    values = rng.integers(0, 1 << bits, 4096)
    assert (unpack(pack(values, bits), bits, 4096) == values).all()


def flat_world_chunk(cx=0, cz=0, extra=None, **kw):
    """Stone up to y=63 with grass on top (ground y=64); ``extra`` maps (y,z,x)->block."""
    palette = ["minecraft:air", "minecraft:stone", "minecraft:grass_block"]
    sections = []
    for sy in range(-4, 8):
        idx = np.zeros((16, 16, 16), dtype=int)
        for ly in range(16):
            y = sy * 16 + ly
            idx[ly] = 1 if y < 64 else 2 if y == 64 else 0
        names = list(palette)
        for (y, z, x), name in (extra or {}).items():
            if y // 16 == sy:
                if name not in names:
                    names.append(name)
                idx[y % 16, z, x] = names.index(name)
        sections.append(section_nbt(sy, names, idx.reshape(-1)))
    return full_chunk_nbt(cx, cz, sections, **kw)


def row(raw: bytes, entities=()):
    return Extractor(VOCAB, LAYOUT).row(PROFILE.read_chunk(parse(raw)), list(entities))


def test_ground_and_bands():
    # Torches do not count as ground, so they mark positions without moving it.
    # ground+7 (surface band edge), ground+8 (above), ground-8 (shallow), ground-33 (deep).
    t = "minecraft:torch"
    r = row(flat_world_chunk(extra={(71, 0, 0): t, (72, 0, 1): t, (56, 0, 2): t, (31, 0, 3): t}))
    assert r[COL["ground.mean"]] == 64 and r[COL["ground.std"]] == 0
    assert r[COL["block.surface.minecraft:torch"]] == 1
    assert r[COL["block.above.minecraft:torch"]] == 1
    assert r[COL["block.shallow.minecraft:torch"]] == 1
    assert r[COL["block.deep.minecraft:torch"]] == 1
    # Grass layer is entirely in the surface band; air is never counted.
    assert r[COL["block.surface.minecraft:grass_block"]] == 256
    assert not any(c.endswith("minecraft:air") for c in LAYOUT.schema.columns)
    assert r[COL["chunk.unknown_blocks"]] == 0 and r[COL["chunk.full"]] == 1


def test_solid_player_blocks_raise_ground():
    # Documents current behaviour: a glass block on top becomes that column's ground.
    r = row(flat_world_chunk(extra={(72, 0, 0): "minecraft:glass"}))
    assert r[COL["ground.max"]] == 72


def test_water_and_leaves_are_not_ground():
    r = row(flat_world_chunk(extra={(65 + i, 5, 5): "minecraft:oak_leaves" for i in range(10)}
                             | {(65, 6, 6): "minecraft:water"}))
    assert r[COL["ground.max"]] == 64


def test_block_entities_structures_entities():
    raw = flat_world_chunk(
        block_entities=[("minecraft:chest", 64, {"LootTable": String("x")}),
                        ("minecraft:chest", 65, {"Items": List[Compound]([Compound({"id": String("minecraft:dirt")})])})],
        structures=["minecraft:village_plains"],
    )
    ent = PROFILE.read_entities(Compound({"Entities": List[Compound]([
        Compound({"id": String("minecraft:armor_stand"), "Pos": List[Double]([Double(1), Double(65), Double(1)])}),
        Compound({"id": String("minecraft:sheep"), "Pos": List[Double]([Double(2), Double(65), Double(2)]),
                  "CustomName": String("Bob"), "leash": Compound({})}),
    ])}))
    r = row(raw, ent)
    assert r[COL["be.minecraft:chest"]] == 2
    assert (r[COL["be.loot_table"]], r[COL["be.with_items"]]) == (1, 1)
    assert r[COL["structure.minecraft:village_plains"]] == 1
    assert r[COL["entity.minecraft:armor_stand"]] == 1
    assert (r[COL["entity.named"]], r[COL["entity.leashed"]]) == (1, 1)
    assert r[COL["biome.minecraft:plains"]] == pytest.approx(1.0)


def test_unknown_block_is_counted_not_dropped_silently():
    assert row(flat_world_chunk(extra={(70, 0, 0): "minecraft:not_a_block"}))[COL["chunk.unknown_blocks"]] == 1


def test_pipeline_writes_and_skips(tmp_path: Path):
    dim = tmp_path / "world/dimensions/minecraft/overworld"
    write_region(dim / "region/r.0.0.mca", {(0, 0): flat_world_chunk(0, 0), (1, 0): flat_world_chunk(1, 0)})
    write_region(dim / "region/r.-1.0.mca", {(-1, 0): flat_world_chunk(-1, 0)})
    out, results = extract_dimension(tmp_path / "ws", dim, "java-26.2", workers=2)
    assert sum(r.chunks for r in results) == 3 and not any(r.error for r in results)
    positions, (indptr, indices, data), n_cols = read_table(out)
    assert n_cols == len(LAYOUT.schema.columns) == len(read_manifest(out)["columns"])
    assert sorted(map(tuple, positions)) == [(-1, 0), (0, 0), (1, 0)]
    # Second run: nothing changed, everything skipped.
    _, again = extract_dimension(tmp_path / "ws", dim, "java-26.2", workers=2)
    assert all(r.skipped for r in again)
