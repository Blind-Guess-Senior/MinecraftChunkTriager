from pathlib import Path

import pytest

from chunktriager.cli import main
from chunktriager.config import ConfigError
from chunktriager.workspace import ENV_HOME, ENV_WORKSPACE, resolve, saved_default


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "home"))
    monkeypatch.delenv(ENV_WORKSPACE, raising=False)
    monkeypatch.chdir(tmp_path)


def test_init_sets_default_and_commands_need_no_w(tmp_path: Path, world: Path):
    ws = tmp_path / "Triager"
    assert main(["-w", str(ws), "init", "--world", str(world)]) == 0
    assert saved_default() == ws.resolve()
    assert main(["scan"]) == 0


def test_world_path_with_backslashes_is_valid_toml(tmp_path: Path, world: Path):
    ws = tmp_path / "ws"
    main(["-w", str(ws), "init", "--world", str(world)])
    assert "\\\\" not in (ws / "triage.toml").read_text(encoding="utf-8")
    assert main(["scan"]) == 0


def test_use_switches_default(tmp_path: Path, workspace: Path, capsys):
    assert main(["use", str(tmp_path / "missing")]) == 1
    assert main(["use", str(workspace)]) == 0
    assert resolve(None) == workspace.resolve()


def test_resolution_order(tmp_path: Path, workspace: Path, monkeypatch):
    main(["use", str(workspace)])
    monkeypatch.setenv(ENV_WORKSPACE, str(tmp_path / "env"))
    assert resolve(None) == tmp_path / "env"
    assert resolve(tmp_path / "flag") == tmp_path / "flag"


def test_no_workspace_gives_hint():
    with pytest.raises(ConfigError, match="chunktriager use"):
        resolve(None)
    with pytest.raises(ConfigError, match="init"):
        resolve(None, for_init=True)
