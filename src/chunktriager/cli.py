"""Command line entry point: ``chunktriager [-c config] <command>``.

Every command reads one configuration file (``-c``, default
``chunktriager.toml`` in the current directory) that lists the worlds and
their roles; see :mod:`chunktriager.config`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONFIG_NAME, Config, ConfigError, World, default_config_path, load_config
from .render import RenderError, render_layer
from .store import Store
from .versions import UnknownVersionError, get_profile, supported_versions
from .worldio import list_regions

_TEMPLATE = """\
# chunktriager configuration. Relative paths are relative to this file.

# Labels, per-world caches and models are stored here. Labels are the only
# part that cannot be rebuilt.
data_dir = "data"
# Model written by `train` and read by `predict` (default: <data_dir>/models/current).
# model = "data/models/current"

# One of: {versions}
game_version = "java-26.2"
dimension = "minecraft:overworld"

# MCA Selector draws the base map of the labeling page (`render`).
# [mcaselector]
# command = ["java", "-jar", "C:/path/to/mcaselector.jar"]

# One entry per world. Use force-upgraded copies, never a world you play on.
# name:      used for file names and on the command line (default: folder name)
# role:      train | val | test | none
#            train: fitted on and cross-validated; val: scored by `evaluate`;
#            test: scored only by `evaluate --test`; none: labeling only
# unlabeled: natural (unlabeled chunks are natural) | ignore (default: [training])
{worlds}
[training]
unlabeled = "natural"
folds = 5
# rounds = 400
# [training.params]        # LightGBM parameters overriding the defaults
# learning_rate = 0.05

[protection]
# Chunks within `radius` of a chunk scoring at least `threshold` are kept.
# Evidence that comes only from rails protects `short_radius`.
radius = 8
short_radius = 1
threshold = 0.01

[server]
host = "127.0.0.1"
port = 8765
"""

_EXAMPLE_WORLD = """\
# [[worlds]]
# name = "main"
# path = "C:/path/to/world-copy"
# role = "train"
"""


def _toml_str(value: str) -> str:
    # JSON string escapes are valid TOML basic-string escapes.
    return json.dumps(value, ensure_ascii=False)


def _find_worlds(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ConfigError(f"{folder} is not a folder")
    return sorted(p for p in folder.iterdir() if (p / "level.dat").is_file())


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def cmd_init(args: argparse.Namespace) -> int:
    path: Path = args.config
    if path.exists():
        print(f"{path} already exists; not overwriting", file=sys.stderr)
        return 1
    entries = []
    for folder in args.worlds or []:
        for world in _find_worlds(folder):
            entries.append(f"[[worlds]]\nname = {_toml_str(world.name)}\n"
                           f"path = {_toml_str(_relative(world, path.parent))}\nrole = \"train\"\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.format(
        versions=", ".join(supported_versions()),
        worlds="\n".join(entries) if entries else _EXAMPLE_WORLD,
    ), encoding="utf-8")
    print(f"wrote {path} with {len(entries)} worlds")
    print("set each world's role, then run `chunktriager prepare`")
    return 0


# World context -----------------------------------------------------------------

def _dimension_dir(cfg: Config, world: World) -> Path:
    profile = get_profile(cfg.game_version)
    path = profile.dimension_dir(world.path, cfg.dimension)
    if not (path / "region").is_dir():
        raise ConfigError(f"world {world.name!r}: no region folder at {path / 'region'}")
    return path


def _features_dir(cfg: Config, world: World, required: bool = True) -> Path:
    from .features.schema import build_layout
    from .features.store import table_dir
    from .versions.vocab import load_vocabulary

    layout = build_layout(load_vocabulary(get_profile(cfg.game_version).vocabulary))
    path = table_dir(world.cache_dir, layout.schema)
    if required and not (path / "manifest.json").is_file():
        raise ConfigError(f"world {world.name!r} is not prepared; run `chunktriager prepare {world.name}`")
    return path


def _load_world(cfg: Config, world: World, labelled: bool = True):
    from .model.dataset import load_world
    from .store import read_labels

    labels = read_labels(world.labels_path, cfg.dimension) if labelled else None
    w = load_world(_features_dir(cfg, world), labels, world.unlabeled, name=world.name)
    if labelled:
        print(f"  {w.name} ({world.role}): {int((w.y >= 0).sum())} chunks, {int((w.y == 1).sum())} player-made")
    return w


def _model_dir(cfg: Config, override: Path | None) -> Path:
    return override.resolve() if override else cfg.model_dir


# Commands ----------------------------------------------------------------------

def cmd_worlds(args: argparse.Namespace) -> int:
    from .store import read_labels

    cfg = load_config(args.config)
    profile = get_profile(cfg.game_version)
    if not cfg.worlds:
        print(f"no worlds in {cfg.path}")
        return 0
    for w in cfg.worlds:
        feats = _features_dir(cfg, w, required=False)
        prepared = "prepared" if (feats / "manifest.json").is_file() else "not prepared"
        gen = profile.read_generator(w.path, cfg.dimension) if w.path.is_dir() else None
        labels = read_labels(w.labels_path, cfg.dimension)
        made = sum(1 for v in labels.values() if v == "keep")
        tiles = "tiles" if any(w.tiles_dir.glob("*/0")) else "no tiles"
        terrain = gen.describe() if gen else ("missing" if not w.path.is_dir() else "unknown terrain")
        print(f"{w.name}  [{w.role}]  {terrain}; "
              f"{prepared}, {tiles}, {made} labelled player-made")
        print(f"    {w.path}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .features.pipeline import extract_dimension

    cfg = load_config(args.config)
    worlds = cfg.select(args.worlds)
    if not worlds:
        raise ConfigError(f"no worlds in {cfg.path}")
    failed = 0
    for world in worlds:
        print(f"{world.name}")
        dimension_dir = _dimension_dir(cfg, world)
        errors = []

        def progress(done: int, total: int, r) -> None:
            if r.error:
                errors.append(r)
            print(f"\r  regions {done}/{total}", end="", flush=True)

        out_dir, results = extract_dimension(
            world.cache_dir, dimension_dir, cfg.game_version,
            workers=args.workers, force=args.force, progress=progress,
        )
        print()
        done = [r for r in results if not r.skipped and not r.error]
        skipped = sum(r.skipped for r in results)
        print(f"  {sum(r.chunks for r in done)} chunks in {len(done)} regions extracted"
              + (f", {skipped} regions unchanged" if skipped else ""))
        rejected = sum(r.rejected for r in results)
        if rejected:
            print(f"  {rejected} chunks skipped: DataVersion outside {cfg.game_version} "
                  "(run forceUpgrade on the world copy to include them)")
        for r in errors:
            print(f"  {r.region.filename}: {r.error}", file=sys.stderr)
        failed += bool(errors)
    return 1 if failed else 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if cfg.mcaselector is None:
        raise ConfigError(f"set [mcaselector] command in {cfg.path} to render base maps")
    names = args.layer or [layer.name for layer in cfg.layers]
    try:
        layers = [cfg.layer(n) for n in names]
    except KeyError as e:
        raise ConfigError(f"unknown layer {e.args[0]!r}") from None
    for world in cfg.select(args.worlds):
        dimension_dir = _dimension_dir(cfg, world)
        regions = list_regions(dimension_dir / "region")
        for layer in layers:
            print(f"{world.name}: rendering layer {layer.name}")
            render_layer(
                cfg.mcaselector, dimension_dir, regions, layer, world.tiles_dir,
                lambda done, total: print(f"\r  block {done}/{total}", end="", flush=True),
            )
            print()
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    cfg = load_config(args.config)
    world = cfg.world(args.world)
    profile = get_profile(cfg.game_version)
    app = create_app(cfg, world, profile, _dimension_dir(cfg, world), _features_dir(cfg, world))
    port = args.port or cfg.server.port
    print(f"{world.name}: open http://{cfg.server.host}:{port}/")
    uvicorn.run(app, host=cfg.server.host, port=port, log_level="warning")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .model.train import train_worlds

    cfg = load_config(args.config)
    selected = cfg.select(roles=("train",))
    if not selected:
        raise ConfigError(f"no worlds with role = \"train\" in {cfg.path}")
    print("loading worlds")
    worlds = [_load_world(cfg, w) for w in selected]
    try:
        bundle = train_worlds(worlds, cfg.game_version, cfg.training.rounds, cfg.training.params)
    except ValueError as e:
        raise ConfigError(str(e)) from None
    out = _model_dir(cfg, args.out)
    bundle.save(out)
    print(f"model saved to {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .model.evaluate import report_grouped_cv, report_held_out

    cfg = load_config(args.config)
    train = cfg.select(roles=("train",))
    held = cfg.select(roles=("val", "test") if args.test else ("val",))
    if not train:
        raise ConfigError(f"no worlds with role = \"train\" in {cfg.path}")
    print("loading worlds")
    train_data = [_load_world(cfg, w) for w in train]
    held_data = [_load_world(cfg, w) for w in held]
    report_grouped_cv(train_data, cfg.training, cfg.protection)
    if held_data:
        report_held_out(train_data, held_data, cfg.training, cfg.protection)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .model.predict import predict_world, save_predictions
    from .model.train import ModelBundle

    cfg = load_config(args.config)
    model_dir = _model_dir(cfg, args.model)
    if not (model_dir / "meta.json").is_file():
        raise ConfigError(f"no model at {model_dir}; run `chunktriager train` or pass --model")
    bundle = ModelBundle.load(model_dir)
    for world in cfg.select(args.worlds):
        data = _load_world(cfg, world, labelled=False)
        try:
            p = predict_world(bundle, data, cfg.protection)
        except ValueError as e:
            raise ConfigError(f"world {world.name!r}: {e}") from None
        save_predictions(world.cache_dir, p)
        print(f"{world.name}: {p.meta['chunks']} chunks, {p.meta['flagged']} likely player-made, "
              f"{p.meta['protected']} protected ({p.meta['protected'] / max(1, p.meta['chunks']):.1%})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chunktriager")
    parser.add_argument("-c", "--config", type=Path, default=None,
                        help=f"configuration file (default: ./{CONFIG_NAME})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a configuration template")
    p.add_argument("--worlds", type=Path, action="append",
                   help="add every save in this folder as a world (repeatable)")
    p.set_defaults(func=cmd_init)

    sub.add_parser("worlds", help="list configured worlds and their state").set_defaults(func=cmd_worlds)

    p = sub.add_parser("prepare", help="extract per-chunk features (all worlds by default)")
    p.add_argument("worlds", nargs="*", help="world names")
    p.add_argument("--workers", type=int, help="worker processes (default: CPU count - 1)")
    p.add_argument("--force", action="store_true", help="recompute regions that are up to date")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("render", help="render base-map tiles with MCA Selector")
    p.add_argument("worlds", nargs="*", help="world names (default: all)")
    p.add_argument("--layer", action="append", help="layer name (repeatable; default: all)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("label", help="open the labeling page for one world")
    p.add_argument("world", help="world name")
    p.add_argument("--port", type=int, help="override [server] port")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="train a model on the worlds with role train")
    p.add_argument("--out", type=Path, help="model folder (default: `model` in the config)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="cross-validate train worlds and score val worlds")
    p.add_argument("--test", action="store_true", help="also score worlds with role test")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="score chunks with a trained model (all worlds by default)")
    p.add_argument("worlds", nargs="*", help="world names")
    p.add_argument("--model", type=Path, help="model folder (default: `model` in the config)")
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    args.config = args.config or default_config_path()
    try:
        return args.func(args)
    except (ConfigError, UnknownVersionError, RenderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
