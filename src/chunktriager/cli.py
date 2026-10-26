"""Command line entry point: ``chunktriager <command>``.

The workspace comes from ``-w``, ``CHUNKTRIAGER_WORKSPACE`` or the default
set with ``chunktriager use``; see :mod:`chunktriager.workspace`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import workspace as workspace_mod
from .config import CONFIG_NAME, ConfigError, load_config
from .render import RenderError, render_layer
from .store import Store
from .versions import UnknownVersionError, get_profile, supported_versions
from .worldio import RegionFile, RegionFormatError, list_regions

_TEMPLATE = """\
# Path to the force-upgraded COPY of the world (not the world you play on).
world = {world}
dimension = "minecraft:overworld"
# One of: {versions}
game_version = "java-26.2"

[mcaselector]
# Full command used to start MCA Selector's CLI.
command = ["java", "-jar", "C:/path/to/mcaselector.jar"]
block_regions = 8

# Base-map layers rendered by MCA Selector; `args` are passed to --mode image.
[[layers]]
name = "surface"
title = "Surface"
args = []

[[layers]]
name = "below-60"
title = "Below Y=60"
args = ["--render-height", "60"]

[server]
host = "127.0.0.1"
port = 8765

[training]
# "natural": unlabeled chunks count as natural, so only player-made chunks
# need labeling. "ignore": train only on explicitly labeled chunks.
unlabeled = "natural"

[protection]
# Chunks within `radius` of a likely player-made chunk are protected.
# Evidence that comes only from rails protects `short_radius`.
radius = 8
short_radius = 1
threshold = 0.01
"""


def _context(workspace: Path):
    cfg = load_config(workspace)
    profile = get_profile(cfg.game_version)
    dimension_dir = profile.dimension_dir(cfg.world, cfg.dimension)
    if not (dimension_dir / "region").is_dir():
        raise ConfigError(f"no region folder at {dimension_dir / 'region'}")
    return cfg, profile, dimension_dir


def cmd_init(args: argparse.Namespace) -> int:
    path = args.workspace / CONFIG_NAME
    if path.exists():
        print(f"{path} already exists; not overwriting", file=sys.stderr)
        return 1
    args.workspace.mkdir(parents=True, exist_ok=True)
    world = args.world.resolve().as_posix() if args.world else "C:/path/to/world-copy"
    path.write_text(
        _TEMPLATE.format(world=json.dumps(world), versions=", ".join(supported_versions())),
        encoding="utf-8",
    )
    workspace_mod.save_default(args.workspace)
    print(f"wrote {path} and made it the default workspace")
    print("edit it, then run `chunktriager scan`")
    return 0


def cmd_use(args: argparse.Namespace) -> int:
    if args.folder is None:
        current = workspace_mod.saved_default()
        print(current if current else "no default workspace set")
        return 0
    if not (args.folder / CONFIG_NAME).is_file():
        print(f"{args.folder / CONFIG_NAME} not found; run `chunktriager -w {args.folder} init` first",
              file=sys.stderr)
        return 1
    workspace_mod.save_default(args.folder)
    print(f"default workspace: {args.folder.resolve()}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    cfg, profile, dimension_dir = _context(args.workspace)
    region_dir = dimension_dir / "region"
    regions = list_regions(region_dir)
    positions: list[tuple[int, int]] = []
    problems = 0
    for i, rpos in enumerate(regions, 1):
        try:
            region = RegionFile(region_dir / rpos.filename)
            for cx, cz, root in region.iter_chunks():
                try:
                    profile.parse_chunk(root)
                except Exception as e:  # report and keep scanning
                    problems += 1
                    if problems <= 20:
                        print(f"  {rpos.filename} ({cx}, {cz}): {e}", file=sys.stderr)
                    continue
                positions.append((cx, cz))
        except (RegionFormatError, OSError) as e:
            problems += 1
            print(f"  {rpos.filename}: {e}", file=sys.stderr)
        print(f"\rscanned {i}/{len(regions)} regions", end="", flush=True)
    print()
    with_issues = f", {problems} problems (first 20 shown above)" if problems else ""
    print(f"{len(positions)} chunks in {len(regions)} regions{with_issues}")
    if problems and not args.allow_problems:
        print("index NOT updated; fix the world copy or pass --allow-problems", file=sys.stderr)
        return 1
    store = Store(cfg.db_path)
    try:
        store.replace_chunks(cfg.dimension, positions)
    finally:
        store.close()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    cfg, _, dimension_dir = _context(args.workspace)
    regions = list_regions(dimension_dir / "region")
    names = args.layers or [layer.name for layer in cfg.layers]
    for name in names:
        try:
            layer = cfg.layer(name)
        except KeyError:
            print(f"unknown layer {name!r}", file=sys.stderr)
            return 1
        print(f"rendering layer {name}")
        render_layer(
            cfg, dimension_dir, regions, layer,
            lambda done, total: print(f"\r  block {done}/{total}", end="", flush=True),
        )
        print()
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from .features.pipeline import extract_dimension

    cfg, _, dimension_dir = _context(args.workspace)
    errors = []

    def progress(done: int, total: int, r) -> None:
        if r.error:
            errors.append(r)
        print(f"\rregions {done}/{total}", end="", flush=True)

    out_dir, results = extract_dimension(
        cfg.workspace, dimension_dir, cfg.game_version,
        workers=args.workers, force=args.force, progress=progress,
    )
    print()
    done = [r for r in results if not r.skipped and not r.error]
    print(f"{sum(r.chunks for r in done)} chunks in {len(done)} regions written to {out_dir}")
    skipped = sum(r.skipped for r in results)
    if skipped:
        print(f"{skipped} regions unchanged, skipped (use --force to redo)")
    rejected = sum(r.rejected for r in results)
    if rejected:
        print(f"{rejected} chunks rejected by the {cfg.game_version} profile (not upgraded)")
    for r in errors:
        print(f"  {r.region.filename}: {r.error}", file=sys.stderr)
    return 1 if errors else 0


def _features_dir(cfg) -> Path:
    from .features.schema import build_layout
    from .features.store import table_dir
    from .versions.vocab import load_vocabulary

    layout = build_layout(load_vocabulary(get_profile(cfg.game_version).vocabulary))
    path = table_dir(cfg.workspace, layout.schema)
    if not (path / "manifest.json").is_file():
        raise ConfigError(f"no features for {cfg.workspace}; run `chunktriager -w {cfg.workspace} features`")
    return path


def _load_worlds(paths: list[Path]):
    from .model.dataset import load_world
    from .store import read_labels

    worlds, cfgs = [], []
    for ws in paths:
        cfg = load_config(ws)
        labels = read_labels(cfg.db_path, cfg.dimension)
        w = load_world(_features_dir(cfg), labels, cfg.training.unlabeled, name=cfg.workspace.name)
        print(f"  {w.name}: {int((w.y >= 0).sum())} chunks, {int((w.y == 1).sum())} player-made")
        worlds.append(w)
        cfgs.append(cfg)
    return worlds, cfgs


def cmd_train(args: argparse.Namespace) -> int:
    from .model.train import train_worlds

    paths = [args.workspace] + list(args.with_ or [])
    print("loading worlds")
    worlds, cfgs = _load_worlds(paths)
    if args.cv:
        _report_cv(worlds)
    bundle = train_worlds(worlds, cfgs[0].game_version)
    out = args.out or (cfgs[0].models_dir / "current")
    bundle.save(out)
    print(f"model saved to {out}")
    return 0


def _report_cv(worlds) -> None:
    import numpy as np
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score

    from .model.evaluate import protect_two_radii
    from .model.train import assign_folds, fit, site_groups

    # Folds are drawn per world; a world with no player-made chunks joins
    # every fold's training set and is tested in its own right.
    print("cross-validation (grouped by site)")
    labelled = [w for w in worlds if (w.y == 1).any()]
    natural_only = [w for w in worlds if not (w.y == 1).any()]
    for w in labelled:
        use = np.flatnonzero(w.y >= 0)
        y = w.y[use].astype(np.int32)
        folds = assign_folds(site_groups(w.positions[use], y), y, 5)
        oof = np.zeros(len(use))
        extra_X = [o.X[o.y >= 0] for o in worlds if o is not w]
        extra_y = [o.y[o.y >= 0].astype(np.int32) for o in worlds if o is not w]
        for f in range(5):
            te = folds == f
            model = fit(sp.vstack([w.X[use][~te], *extra_X], format="csr"), np.concatenate([y[~te], *extra_y]))
            oof[te] = model.predict(w.X[use][te])
        print(f"  {w.name}: ROC AUC {roc_auc_score(y, oof):.4f}")
        for t in (3e-2, 1e-2, 3e-3):
            hit = oof >= t
            kept = protect_two_radii(w.positions[use], hit.astype(np.float32), w.rail_only[use] & hit, 8, 1) > 0
            print(f"    t={t:.0e}: player-made protected {kept[y == 1].mean():.3f}, deletable {(~kept).mean():.3f}, "
                  f"natural flagged {int((hit & (y == 0)).sum())}")
    for w in natural_only:
        others = [o for o in worlds if o is not w]
        if not others:
            continue
        model = fit(sp.vstack([o.X[o.y >= 0] for o in others], format="csr"),
                    np.concatenate([o.y[o.y >= 0].astype(np.int32) for o in others]))
        p = model.predict(w.X[w.y >= 0])
        print(f"  {w.name} (all natural, held out): " + ", ".join(
            f"t={t:.0e} flagged {(p >= t).mean():.4f}" for t in (3e-2, 1e-2, 3e-3)))


def cmd_predict(args: argparse.Namespace) -> int:
    from .model.dataset import load_world
    from .model.predict import predict_world, save_predictions
    from .model.train import ModelBundle

    cfg, _, _ = _context(args.workspace)
    model_dir = args.model or (cfg.models_dir / "current")
    if not (model_dir / "meta.json").is_file():
        raise ConfigError(f"no model at {model_dir}; run `chunktriager train` or pass --model")
    bundle = ModelBundle.load(model_dir)
    world = load_world(_features_dir(cfg), None, name=cfg.workspace.name)
    try:
        p = predict_world(bundle, world, cfg.protection)
    except ValueError as e:
        raise ConfigError(str(e)) from None
    path = save_predictions(cfg.workspace, p)
    print(f"{p.meta['chunks']} chunks: {p.meta['flagged']} likely player-made, "
          f"{p.meta['protected']} protected ({p.meta['protected'] / p.meta['chunks']:.1%}); written to {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    cfg, profile, dimension_dir = _context(args.workspace)
    app = create_app(cfg, profile, dimension_dir)
    print(f"open http://{cfg.server.host}:{cfg.server.port}/")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chunktriager")
    parser.add_argument("-w", "--workspace", type=Path, default=None,
                        help="workspace folder holding triage.toml (default: set with `use`)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="create a workspace config and make it the default")
    p.add_argument("--world", type=Path, help="force-upgraded world copy")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("use", help="set or show the default workspace")
    p.add_argument("folder", type=Path, nargs="?", help="workspace folder (omit to show)")
    p.set_defaults(func=cmd_use)
    p = sub.add_parser("scan", help="build the chunk index from region files")
    p.add_argument("--allow-problems", action="store_true",
                   help="update the index even if some chunks could not be read")
    p.set_defaults(func=cmd_scan)
    p = sub.add_parser("render", help="render base-map tiles with MCA Selector")
    p.add_argument("layers", nargs="*", help="layer names (default: all)")
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("features", help="extract per-chunk features for training")
    p.add_argument("--workers", type=int, help="worker processes (default: CPU count - 1)")
    p.add_argument("--force", action="store_true", help="recompute regions that are up to date")
    p.set_defaults(func=cmd_features)
    p = sub.add_parser("train", help="train a model on this workspace and others")
    p.add_argument("--with", dest="with_", type=Path, action="append",
                   help="another workspace to train on (repeatable)")
    p.add_argument("--cv", action="store_true", help="report grouped cross-validation first")
    p.add_argument("--out", type=Path, help="model folder (default: <workspace>/models/current)")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("predict", help="score every chunk with a trained model")
    p.add_argument("--model", type=Path, help="model folder (default: <workspace>/models/current)")
    p.set_defaults(func=cmd_predict)
    sub.add_parser("serve", help="start the labeling UI").set_defaults(func=cmd_serve)
    args = parser.parse_args(argv)
    try:
        if args.func is not cmd_use:
            args.workspace = workspace_mod.resolve(args.workspace, for_init=args.func is cmd_init)
        return args.func(args)
    except (ConfigError, UnknownVersionError, RenderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
