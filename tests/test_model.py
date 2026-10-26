import numpy as np

from chunktriager.model.evaluate import at_recall, protect
from chunktriager.model.train import assign_folds, site_groups


def test_protect_spreads_max_within_radius():
    pos = np.array([(x, 0) for x in range(-5, 6)])
    prob = np.zeros(len(pos), np.float32)
    prob[5] = 0.9                                   # chunk (0, 0)
    out = protect(pos, prob, 2)
    assert list(out > 0) == [abs(x) <= 2 for x in range(-5, 6)]
    assert out.max() == np.float32(0.9)


def test_site_groups_keep_nearby_chunks_together():
    pos = np.array([(0, 0), (10, 0), (100, 0), (5, 3), (200, 200)])
    y = np.array([1, 1, 1, 0, 0])
    g = site_groups(pos, y)
    assert g[0] == g[1] == g[3]                     # within 16 of each other / of the site
    assert g[2] != g[0] and g[4] not in (g[0], g[2])


def test_folds_spread_positive_sites():
    groups = np.array([0, 0, 1, 2, 3, 4, 5])
    y = np.array([1, 1, 1, 1, 0, 0, 0])
    f = assign_folds(groups, y, 3)
    assert len({f[0], f[2], f[3]}) == 3             # each positive site in its own fold
    assert f[0] == f[1]


def test_at_recall():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.5, 0.1, 0.1, 0.1])
    full = at_recall(y, p, (1.0,))[0]
    assert full.threshold == 0.4 and full.false_positives == 1 and full.natural_fpr == 0.25


# End-to-end: features -> train -> predict -> server ------------------------------

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chunktriager.cli import main
from chunktriager.config import load_config
from chunktriager.model.derived import derived_columns
from chunktriager.model.predict import load_predictions
from chunktriager.server import create_app
from chunktriager.store import Store
from chunktriager.versions import get_profile

from .test_features import flat_world_chunk
from .conftest import write_region


def _make_world(root: Path, houses, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    region = root / "dimensions/minecraft/overworld/region/r.0.0.mca"
    chunks = {}
    for x in range(32):
        for z in range(32):
            extra = {(65, int(a), int(b)): "minecraft:poppy" for a, b in rng.integers(0, 16, (3, 2))}
            if (x, z) in houses:
                extra |= {(65, a, b): "minecraft:glass" for a in range(4) for b in range(4)}
                extra[(66, 0, 0)] = "minecraft:oak_planks"
            chunks[(x, z)] = flat_world_chunk(x, z, extra=extra)
    write_region(region, chunks)
    return root


def _workspace(tmp: Path, name: str, world: Path) -> Path:
    ws = tmp / name
    assert main(["-w", str(ws), "init", "--world", str(world)]) == 0
    for cmd in (["scan"], ["features", "--workers", "1"]):
        assert main(["-w", str(ws), *cmd]) == 0
    return ws


@pytest.fixture
def trained(tmp_path: Path, monkeypatch):
    """Train on one labelled world, predict on a second world with other builds."""
    monkeypatch.setenv("CHUNKTRIAGER_HOME", str(tmp_path / "home"))
    train_houses = {(x, z) for x in range(0, 3) for z in range(0, 3)} | {(x, z) for x in range(28, 31) for z in range(28, 31)}
    ws_a = _workspace(tmp_path, "a", _make_world(tmp_path / "world_a", train_houses, 1))
    store = Store(load_config(ws_a).db_path)
    store.set_labels("minecraft:overworld", train_houses, "keep", "area")
    store.close()
    assert main(["-w", str(ws_a), "train"]) == 0

    test_houses = {(x, z) for x in range(14, 17) for z in range(5, 7)}
    ws_b = _workspace(tmp_path, "b", _make_world(tmp_path / "world_b", test_houses, 2))
    store = Store(load_config(ws_b).db_path)
    store.set_labels("minecraft:overworld", test_houses, "keep", "area")
    store.close()
    assert main(["-w", str(ws_b), "predict", "--model", str(load_config(ws_a).models_dir / "current")]) == 0
    return ws_b, test_houses


def test_predict_finds_builds_and_protects_around_them(trained):
    ws, houses = trained
    p = load_predictions(ws)
    prob = {tuple(map(int, xz)): v for xz, v in zip(p.positions, p.prob)}
    assert min(prob[h] for h in houses) > 0.5
    prot = {tuple(map(int, xz)) for xz, v in zip(p.positions, p.protected) if v}
    assert {(24, 5), (6, 14)} <= prot and (26, 26) not in prot     # radius 8 around (14..16, 5..6)


def test_server_serves_predictions_and_review(trained):
    ws, houses = trained
    cfg = load_config(ws)
    profile = get_profile(cfg.game_version)
    client = TestClient(create_app(cfg, profile, profile.dimension_dir(cfg.world, cfg.dimension)))
    meta = client.get("/api/meta").json()["predictions"]
    assert meta["chunks"] == 1024 and meta["flagged"] >= len(houses)
    region = client.get("/api/predictions", params=dict(rx0=0, rz0=0, rx1=0, rz1=0)).json()["0,0"]
    i = 5 * 32 + 14                                                  # chunk (14, 5), a build
    assert int(region["prob"][2 * i:2 * i + 2], 16) > 127 and region["protected"][i] == "p"
    assert client.get("/api/review", params={"kind": "missed"}).json() == []


def test_model_refuses_mismatched_features(trained):
    from chunktriager.model.dataset import World
    from chunktriager.model.train import ModelBundle

    ws, _ = trained
    bundle = ModelBundle.load(ws.parent / "a" / "models" / "current")
    fake = World("x", np.zeros((0, 2)), None, None, np.zeros(0, bool), ["a"], "different")
    with pytest.raises(ValueError, match="feature schema"):
        bundle.check(fake)


def test_derived_columns_are_stable():
    cols = derived_columns()
    assert len(cols) == len(set(cols)) == 55 and cols[-1] == "rail.only"
