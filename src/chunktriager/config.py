"""Working directory describing one training setup.

A working directory holds ``chunktriager.toml`` (the worlds to work with,
the role each one plays, tool settings) and everything the commands
produce. Users create one anywhere with ``chunktriager init``. Relative
paths in the file are resolved against the working directory.

Layout::

    chunktriager.toml
    labels/<world>.sqlite3     human labels, the only data that cannot be rebuilt
    cache/<world>/             features (also the chunk index), tiles, predictions
    models/current/            model written by `train` (unless `model` is set)
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "chunktriager.toml"
_LAYER_NAME = re.compile(r"^[a-z0-9_-]+$")
# World names become file and folder names.
_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# train: the model is fitted on these worlds; ``evaluate`` cross-validates them.
# val:   scored by ``evaluate`` with a model fitted on the train worlds.
# test:  scored only by ``evaluate --test``, kept aside for the final check.
# none:  prepared and labelable, not used by the model (e.g. while labeling).
ROLES = ("train", "val", "test", "none")
UNLABELED = ("natural", "ignore")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class RenderLayer:
    """One MCA Selector render preset, e.g. surface or a Y-limited view."""

    name: str
    title: str
    args: tuple[str, ...] = ()


DEFAULT_LAYERS = (
    RenderLayer("surface", "Surface"),
    RenderLayer("below-60", "Below Y=60", ("--render-height", "60")),
)


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
    # Default for worlds that do not set their own: "natural" counts
    # unlabeled chunks as natural, so only player-made chunks need labeling;
    # "ignore" uses only explicitly labeled chunks.
    unlabeled: str = "natural"
    folds: int = 5
    # None keeps the defaults in chunktriager.model.train.
    rounds: int | None = None
    params: dict = field(default_factory=dict)


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
class World:
    name: str
    path: Path
    role: str
    unlabeled: str
    labels_path: Path
    cache_dir: Path

    @property
    def tiles_dir(self) -> Path:
        return self.cache_dir / "tiles"


@dataclass(frozen=True)
class Config:
    workdir: Path
    model_dir: Path
    game_version: str
    dimension: str
    mcaselector: McaSelectorConfig | None
    layers: tuple[RenderLayer, ...]
    server: ServerConfig
    training: TrainingConfig
    protection: ProtectionConfig
    worlds: tuple[World, ...]

    def layer(self, name: str) -> RenderLayer:
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    @property
    def path(self) -> Path:
        return self.workdir / CONFIG_NAME

    def world(self, name: str) -> World:
        for w in self.worlds:
            if w.name == name:
                return w
        known = ", ".join(w.name for w in self.worlds) or "none"
        raise ConfigError(f"no world named {name!r} in {self.path}; configured: {known}")

    def select(self, names: list[str] | None = None, roles: tuple[str, ...] = ROLES) -> list[World]:
        """Worlds by name (all named ones, any role) or else every world with one of ``roles``."""
        if names:
            return [self.world(n) for n in names]
        return [w for w in self.worlds if w.role in roles]


def _resolve(base: Path, value: object, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _str_tuple(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{key} must be a list of strings")
    return tuple(value)


def _choice(value: object, options: tuple[str, ...], key: str) -> str:
    if value not in options:
        raise ConfigError(f"{key} must be one of {', '.join(map(repr, options))}, got {value!r}")
    return value


def _worlds(raw: list, base: Path, unlabeled: str) -> tuple[World, ...]:
    worlds = []
    for i, entry in enumerate(raw):
        key = f"worlds[{i}]"
        if "path" not in entry:
            raise ConfigError(f"{key} needs a path")
        path = _resolve(base, entry["path"], f"{key}.path")
        name = entry.get("name", path.name)
        if not isinstance(name, str) or not name.strip() or _BAD_NAME.search(name) or name in (".", ".."):
            raise ConfigError(f"{key}.name {name!r} cannot be used as a file name")
        worlds.append(World(
            name=name,
            path=path,
            role=_choice(entry.get("role", "train"), ROLES, f"{key}.role"),
            unlabeled=_choice(entry.get("unlabeled", unlabeled), UNLABELED, f"{key}.unlabeled"),
            labels_path=base / "labels" / f"{name}.sqlite3",
            cache_dir=base / "cache" / name,
        ))
    names = [w.name.casefold() for w in worlds]
    if len(set(names)) != len(names):
        raise ConfigError("world names must be unique (case-insensitive); set `name` on duplicates")
    return tuple(worlds)


def load_config(workdir: Path) -> Config:
    base = workdir.resolve()
    path = base / CONFIG_NAME
    if not path.is_file():
        raise ConfigError(f"{base} is not a working directory (no {CONFIG_NAME}); "
                          "create one with `chunktriager -w <folder> init`, or pass -w")
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    model_dir = _resolve(base, raw["model"], "model") if "model" in raw else base / "models" / "current"

    mca = raw.get("mcaselector")
    mcaselector = None
    if mca is not None:
        if "command" not in mca:
            raise ConfigError("mcaselector.command is required when [mcaselector] is present")
        mcaselector = McaSelectorConfig(
            command=_str_tuple(mca["command"], "mcaselector.command"),
            block_regions=int(mca.get("block_regions", 8)),
            timeout=int(mca.get("timeout", 3600)),
        )

    layers = []
    for entry in raw.get("layers", []):
        name = entry.get("name", "")
        if not _LAYER_NAME.match(name):
            raise ConfigError(f"invalid layer name {name!r}")
        args = _str_tuple(entry.get("args", []), f"layers.{name}.args")
        layers.append(RenderLayer(name, entry.get("title", name), args))
    if len({layer.name for layer in layers}) != len(layers):
        raise ConfigError("layer names must be unique")

    server = raw.get("server", {})
    training = raw.get("training", {})
    unlabeled = _choice(training.get("unlabeled", "natural"), UNLABELED, "training.unlabeled")
    params = training.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("training.params must be a table")
    prot = raw.get("protection", {})
    return Config(
        workdir=base,
        model_dir=model_dir,
        game_version=str(raw.get("game_version", "java-26.2")),
        dimension=raw.get("dimension", "minecraft:overworld"),
        mcaselector=mcaselector,
        layers=tuple(layers) or DEFAULT_LAYERS,
        server=ServerConfig(
            host=server.get("host", "127.0.0.1"),
            port=int(server.get("port", 8765)),
        ),
        training=TrainingConfig(
            unlabeled=unlabeled,
            folds=int(training.get("folds", 5)),
            rounds=int(training["rounds"]) if "rounds" in training else None,
            params=dict(params),
        ),
        protection=ProtectionConfig(
            radius=int(prot.get("radius", 8)),
            short_radius=int(prot.get("short_radius", 1)),
            short_radius_blocks=_str_tuple(prot.get("short_radius_blocks", list(RAILS)),
                                           "protection.short_radius_blocks"),
            threshold=float(prot.get("threshold", 0.01)),
        ),
        worlds=_worlds(raw.get("worlds", []), base, unlabeled),
    )
