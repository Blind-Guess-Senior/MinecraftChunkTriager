from pathlib import Path

import pytest

from chunktriager.versions import UnsupportedChunkError, get_profile
from chunktriager.worldio import RegionFile, RegionPos, list_regions, parse_region_name

from .conftest import chunk_nbt, write_region

REGION_DIR = Path("dimensions/minecraft/overworld/region")


def test_parse_region_name():
    assert parse_region_name("r.-3.12.mca") == RegionPos(-3, 12)
    assert parse_region_name("r.1.2.mcc") is None
    assert parse_region_name("c.1.2.mcc") is None


def test_list_regions_skips_empty_files(world: Path):
    (world / REGION_DIR / "r.5.5.mca").write_bytes(b"\x00" * 8192)
    assert list_regions(world / REGION_DIR) == [RegionPos(-1, 0), RegionPos(0, 0)]


def test_chunk_positions_are_absolute(world: Path):
    region = RegionFile(world / REGION_DIR / "r.-1.0.mca")
    assert sorted(region.chunk_positions()) == [(-32, 0), (-1, 0)]


def test_read_chunk_and_parse(world: Path):
    region = RegionFile(world / REGION_DIR / "r.0.0.mca")
    root = region.read_chunk(0, 0)
    info = get_profile("java-26.2").parse_chunk(root)
    assert (info.x, info.z) == (0, 0)
    assert [be.id for be in info.block_entities].count("minecraft:chest") == 2
    assert region.read_chunk(5, 5) is None


def test_profile_rejects_other_data_versions(tmp_path: Path):
    path = tmp_path / "r.0.0.mca"
    write_region(path, {(0, 0): chunk_nbt(0, 0, data_version=3700)})
    root = RegionFile(path).read_chunk(0, 0)
    with pytest.raises(UnsupportedChunkError, match="forceUpgrade"):
        get_profile("java-26.2").parse_chunk(root)


def test_dimension_dir_layout(tmp_path: Path):
    profile = get_profile("java-26.2")
    assert profile.dimension_dir(tmp_path, "minecraft:overworld") == tmp_path / REGION_DIR.parent
    with pytest.raises(ValueError):
        profile.dimension_dir(tmp_path, "overworld")
