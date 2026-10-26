from pathlib import Path

import pytest

from chunktriager.cli import main
from chunktriager.config import ConfigError, load_config

from .conftest import write_config


def test_everything_lives_in_the_working_directory(tmp_path: Path, world: Path):
    workdir = tmp_path / "setup"
    workdir.mkdir()
    (workdir / "chunktriager.toml").write_text('[[worlds]]\npath = "../world"\n', encoding="utf-8")
    cfg = load_config(workdir)
    w = cfg.worlds[0]
    assert w.name == "world" and w.path == world.resolve() and w.role == "train"
    assert w.labels_path == (workdir / "labels/world.sqlite3").resolve()
    assert w.cache_dir == (workdir / "cache/world").resolve()
    assert cfg.model_dir == (workdir / "models/current").resolve()
    assert cfg.mcaselector is None and [l.name for l in cfg.layers] == ["surface", "below-60"]


def test_roles_and_unlabeled_are_validated(tmp_path: Path, world: Path):
    workdir = write_config(tmp_path / "c", {"w": (world, "trian")})
    with pytest.raises(ConfigError, match="role"):
        load_config(workdir)
    (workdir / "chunktriager.toml").write_text(f'[[worlds]]\npath = "{world.as_posix()}"\nunlabeled = "x"\n',
                                               encoding="utf-8")
    with pytest.raises(ConfigError, match="unlabeled"):
        load_config(workdir)


def test_world_names_must_be_unique_and_safe(tmp_path: Path, world: Path):
    path = tmp_path / "chunktriager.toml"
    path.write_text(f'[[worlds]]\npath = "{world.as_posix()}"\n[[worlds]]\nname = "WORLD"\npath = "{world.as_posix()}"\n',
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(tmp_path)
    path.write_text(f'[[worlds]]\nname = "a/b"\npath = "{world.as_posix()}"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="file name"):
        load_config(tmp_path)


def test_select_by_name_or_role(tmp_path: Path, world: Path):
    cfg = load_config(write_config(tmp_path / "c", {
        "a": (world, "train"), "b": (world, "val"), "c": (world, "test")}))
    assert [w.name for w in cfg.select(roles=("train",))] == ["a"]
    assert [w.name for w in cfg.select(["c", "a"])] == ["c", "a"]
    with pytest.raises(ConfigError, match="configured: a, b, c"):
        cfg.select(["missing"])


def test_init_discovers_saves_and_writes_valid_toml(tmp_path: Path, world: Path, monkeypatch):
    (world / "level.dat").write_bytes(b"")
    saves = world.parent
    named = saves / "存档 [1]"
    (named / "dimensions/minecraft/overworld/region").mkdir(parents=True)
    (named / "level.dat").write_bytes(b"")
    workdir = tmp_path / "triage"
    assert main(["-w", str(workdir), "init", "--worlds", str(saves)]) == 0
    cfg = load_config(workdir)
    assert sorted(w.name for w in cfg.worlds) == ["world", "存档 [1]"]
    assert {w.path for w in cfg.worlds} == {world.resolve(), named.resolve()}
    assert main(["-w", str(workdir), "init"]) == 1  # never overwrites
    monkeypatch.chdir(workdir)
    assert main(["worlds"]) == 0                    # current directory is the default


def test_missing_config_gives_hint(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["worlds"]) == 1
    assert "not a working directory" in capsys.readouterr().err


def test_flat_generator_is_read(tmp_path: Path, world: Path):
    import nbtlib
    from nbtlib import Byte, Compound, Int, List, String

    from chunktriager.versions import get_profile

    settings = Compound({"biome": String("minecraft:the_void"), "features": Byte(1), "layers": List[Compound](
        [Compound({"block": String("minecraft:air"), "height": Int(1)})])})
    gen = Compound({"type": String("minecraft:flat"), "settings": settings})
    root = nbtlib.File({"data": Compound({"dimensions": Compound(
        {"minecraft:overworld": Compound({"generator": gen})})})})
    (world / "data/minecraft").mkdir(parents=True)
    root.save(world / "data/minecraft/world_gen_settings.dat", gzipped=True)
    g = get_profile("java-26.2").read_generator(world, "minecraft:overworld")
    assert g.kind == "flat" and g.layers == (("minecraft:air", 1),) and g.describe() == "void"
