from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from chunktriager.cli import main
from chunktriager.config import load_config
from chunktriager.render import build_pyramid, cut_block_image, group_blocks, pyramid_levels, tile_path
from chunktriager.server import create_app
from chunktriager.versions import get_profile
from chunktriager.worldio import RegionPos

from .conftest import chunk_nbt, write_region


@pytest.fixture
def client(workspace: Path) -> TestClient:
    assert main(["-w", str(workspace), "scan"]) == 0
    cfg = load_config(workspace)
    profile = get_profile(cfg.game_version)
    return TestClient(create_app(cfg, profile, profile.dimension_dir(cfg.world, cfg.dimension)))


def test_scan_refuses_unupgraded_chunks(workspace: Path, world: Path, capsys):
    region = world / "dimensions/minecraft/overworld/region/r.1.0.mca"
    write_region(region, {(32, 0): chunk_nbt(32, 0, data_version=3700)})
    assert main(["-w", str(workspace), "scan"]) == 1
    assert "index NOT updated" in capsys.readouterr().err
    assert main(["-w", str(workspace), "scan", "--allow-problems"]) == 0


def test_meta_and_state(client: TestClient):
    meta = client.get("/api/meta").json()
    assert meta["chunk_count"] == 5
    assert meta["regions"] == [[-1, 0], [0, 0]]
    state = client.get("/api/state", params=dict(rx0=-1, rz0=0, rx1=0, rz1=0)).json()
    assert set(state) == {"-1,0", "0,0"}
    s = state["0,0"]
    assert (s[0], s[1], s[32], s[2]) == ("u", "u", "u", ".")
    assert state["-1,0"][31] == "u" and state["-1,0"][0] == "u"


def test_label_undo_and_clear_all(client: TestClient):
    r = client.post("/api/labels", json={"label": "delete", "source": "area", "rect": [-40, -5, 5, 5]})
    assert r.json()["changed"] == 5
    client.post("/api/labels", json={"label": "keep", "chunks": [[0, 1]]})
    s = client.get("/api/state", params=dict(rx0=0, rz0=0, rx1=0, rz1=0)).json()["0,0"]
    assert (s[0], s[32]) == ("d", "k")

    client.post("/api/undo")
    assert client.get("/api/chunk/0/1").json()["label"] == "delete"

    r = client.post("/api/labels/clear").json()
    assert r["changed"] == 5 and r["counts"] == {"keep": 0, "delete": 0}
    client.post("/api/undo")
    assert client.get("/api/meta").json()["counts"] == {"keep": 0, "delete": 5}


def test_chunk_detail(client: TestClient):
    d = client.get("/api/chunk/0/0").json()
    assert d["block_entities"][0] == ["minecraft:chest", 2]
    assert d["tp"] == "/tp @s 8 ~ 8"
    assert client.get("/api/chunk/5/5").status_code == 404


def test_rect_too_large_rejected(client: TestClient):
    r = client.post("/api/labels", json={"label": "keep", "rect": [0, 0, 5000, 5000]})
    assert r.status_code == 400


def test_tiles_cut_and_pyramid(tmp_path: Path):
    regions = [RegionPos(-1, 0), RegionPos(0, 0), RegionPos(0, 1)]
    img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    img.paste((255, 0, 0, 255), (0, 0, 512, 512))      # r.-1.0
    img.paste((0, 0, 255, 255), (512, 512, 1024, 1024))  # r.0.1
    img.save(tmp_path / "block.png")
    cut_block_image(tmp_path / "block.png", regions, tmp_path / "tiles", "surface")
    with Image.open(tile_path(tmp_path / "tiles", "surface", 0, -1, 0)) as t:
        assert t.getpixel((10, 10)) == (255, 0, 0, 255)
    with Image.open(tile_path(tmp_path / "tiles", "surface", 0, 0, 1)) as t:
        assert t.getpixel((10, 10)) == (0, 0, 255, 255)

    levels = pyramid_levels(regions)
    assert levels == 0
    build_pyramid(tmp_path / "tiles", "surface", 1)
    with Image.open(tile_path(tmp_path / "tiles", "surface", -1, -1, 0)) as t:
        assert t.getpixel((300, 10)) == (255, 0, 0, 255)


def test_group_blocks_handles_negative_coordinates():
    blocks = group_blocks([RegionPos(-1, 0), RegionPos(0, 0), RegionPos(7, 7), RegionPos(8, 0)], 8)
    assert blocks == [[RegionPos(-1, 0)], [RegionPos(0, 0), RegionPos(7, 7)], [RegionPos(8, 0)]]
