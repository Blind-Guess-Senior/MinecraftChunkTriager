from pathlib import Path

import nbtlib
import pytest
from nbtlib import Byte, Compound, Int, List, String

from chunktriager.cli import main
from chunktriager.config import load_config

from .conftest import write_config, write_region
from .test_cli_server import make_client
from .test_features import flat_world_chunk


def write_flat_generator(world: Path, layers) -> None:
    settings = Compound({"biome": String("minecraft:plains"), "features": Byte(0), "layers": List[Compound](
        [Compound({"block": String(b), "height": Int(h)}) for b, h in layers])})
    gen = Compound({"type": String("minecraft:flat"), "settings": settings})
    root = nbtlib.File({"data": Compound({"dimensions": Compound(
        {"minecraft:overworld": Compound({"generator": gen})})})})
    (world / "data/minecraft").mkdir(parents=True, exist_ok=True)
    root.save(world / "data/minecraft/world_gen_settings.dat", gzipped=True)


@pytest.fixture
def flat(tmp_path: Path) -> Path:
    """Stone/grass flat world: (0,0) untouched, (1,0) a build, (2,0) a build inside a structure."""
    world = tmp_path / "flat"
    write_flat_generator(world, [("minecraft:stone", 64), ("minecraft:grass_block", 1)])
    build = {(65, 3, 3): "minecraft:oak_planks"}
    write_region(world / "dimensions/minecraft/overworld/region/r.0.0.mca", {
        (0, 0): flat_world_chunk(0, 0),
        (1, 0): flat_world_chunk(1, 0, extra=build),
        (2, 0): flat_world_chunk(2, 0, extra=build, structures=["minecraft:ocean_ruin_cold"]),
    })
    workdir = write_config(tmp_path / "setup", {"flat": (world, "none")})
    assert main(["-w", str(workdir), "prepare", "--workers", "1"]) == 0
    return workdir


def test_proposals_mark_content_beyond_the_layers(flat: Path):
    client = make_client(flat, "flat")
    meta = client.get("/api/meta").json()["proposals"]
    assert meta["baseline"] == "flat: stone x64, grass_block x1"
    assert meta["counts"] == {"e": 1, "s": 1} and meta["pending"] == {"e": 1, "s": 1}
    row = client.get("/api/proposals", params=dict(rx0=0, rz0=0, rx1=0, rz1=0)).json()["0,0"]
    assert row[:3] == ".es"


def test_accept_writes_pending_proposals_as_one_undoable_batch(flat: Path):
    client = make_client(flat, "flat")
    client.post("/api/labels", json={"label": "delete", "chunks": [[1, 0]]})   # user already decided
    r = client.post("/api/proposals/accept", json={"kinds": ["e", "s"]}).json()
    assert r["changed"] == 1 and r["proposals"]["pending"] == {"e": 0, "s": 0}
    assert client.get("/api/chunk/2/0").json()["source"] == "proposal"
    assert client.get("/api/chunk/1/0").json()["label"] == "delete"
    client.post("/api/undo")
    assert client.get("/api/chunk/2/0").json()["label"] is None


def test_accept_skips_structures_unless_asked(flat: Path):
    client = make_client(flat, "flat")
    assert client.post("/api/proposals/accept", json={}).json()["changed"] == 1
    assert client.get("/api/chunk/2/0").json()["label"] is None


def test_noise_worlds_have_no_proposals(config: Path):
    assert main(["-w", str(config), "prepare", "--workers", "1"]) == 0
    client = make_client(config)
    assert client.get("/api/meta").json()["proposals"] is None
    assert client.post("/api/proposals/accept", json={}).status_code == 404
