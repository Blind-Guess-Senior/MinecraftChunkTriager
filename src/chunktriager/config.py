"""Workspace configuration loaded from ``triage.toml``.

A workspace is a directory outside the repository that holds the config,
the label database, rendered tiles and exported selections for one world.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_NAME = "triage.toml"
_LAYER_NAME = re.compile(r"^[a-z0-9_-]+$")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class RenderLayer:
    """One MCA Selector render preset, e.g. surface or a Y-limited view."""

    name: str
    title: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class McaSelectorConfig:
    # Full command prefix, e.g. ("java", "-jar", "mcaselector.jar").
    command: tuple[str, ...]
    # Regions per side rendered by one MCA Selector invocation.
    block_regions: int = 8
    timeout: int = 3600


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


RAILS = ("minecraft:rail", "minecraft:powered_rail", "minecraft:detector_rail", "minecraft:activator_rail")


@dataclass(frozen=True)
class TrainingConfig:
    # "natural": unlabeled chunks count as natural; label only player-made
    # chunks. "ignore": train only on explicitly labeled chunks.
    unlabeled: str = "natural"
    folds: int = 5


@dataclass(frozen=True)
class ProtectionConfig:
    # A chunk is protected if a chunk within `radius` (Chebyshev, in chunks)
    # scores at least `threshold`. Evidence that comes only from
    # `short_radius_blocks` (e.g. a lone rail line) protects `short_radius`.
    radius: int = 8
    short_radius: int = 1
    short_radius_blocks: tuple[str, ...] = RAILS
    threshold: float = 0.01


@dataclass(frozen=True)
class Config:
    workspace: Path
    world: Path
    dimension: str
    game_version: str
    mcaselector: McaSelectorConfig
    layers: tuple[RenderLayer, ...]
    server: ServerConfig
    training: TrainingConfig
    protection: ProtectionConfig

    @property
    def db_path(self) -> Path:
        return self.workspace / "labels.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.workspace / "models"

    @property
    def tiles_dir(self) -> Path:
        return self.workspace / "tiles"

    @property
    def exports_dir(self) -> Path:
        return self.workspace / "exports"

    def layer(self, name: str) -> RenderLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _str_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(value)


def load_config(workspace: Path) -> Config:
    workspace = workspace.resolve()
    path = workspace / CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"{path} not found; run `chunktriager init` first")
    with path.open("rb") as f:
        raw = tomllib.load(f)

    try:
        world = _resolve(workspace, raw["world"])
        game_version = str(raw["game_version"])
        mca = raw["mcaselector"]
        command = _str_tuple(mca["command"], "mcaselector.command")
    except KeyError as e:
        raise ConfigError(f"missing required key {e.args[0]!r} in {path}") from None

    layers = []
    for entry in raw.get("layers", []):
        name = entry.get("name", "")
        if not _LAYER_NAME.match(name):
            raise ConfigError(f"invalid layer name {name!r}")
        args = _str_tuple(entry.get("args", []), f"layers.{name}.args")
        layers.append(RenderLayer(name, entry.get("title", name), args))
    if not layers:
        raise ConfigError("at least one [[layers]] entry is required")
    if len({layer.name for layer in layers}) != len(layers):
        raise ConfigError("layer names must be unique")

    server = raw.get("server", {})
    training = raw.get("training", {})
    if training.get("unlabeled", "natural") not in ("ignore", "natural"):
        raise ConfigError('training.unlabeled must be "ignore" or "natural"')
    prot = raw.get("protection", {})
    return Config(
        workspace=workspace,
        world=world,
        dimension=raw.get("dimension", "minecraft:overworld"),
        game_version=game_version,
        mcaselector=McaSelectorConfig(
            command=command,
            block_regions=int(mca.get("block_regions", 8)),
            timeout=int(mca.get("timeout", 3600)),
        ),
        layers=tuple(layers),
        server=ServerConfig(
            host=server.get("host", "127.0.0.1"),
            port=int(server.get("port", 8765)),
        ),
        training=TrainingConfig(
            unlabeled=training.get("unlabeled", "natural"),
            folds=int(training.get("folds", 5)),
        ),
        protection=ProtectionConfig(
            radius=int(prot.get("radius", 8)),
            short_radius=int(prot.get("short_radius", 1)),
            short_radius_blocks=_str_tuple(prot.get("short_radius_blocks", list(RAILS)),
                                           "protection.short_radius_blocks"),
            threshold=float(prot.get("threshold", 0.01)),
        ),
    )
