"""Command line entry point: ``chunktriager <command> --workspace DIR``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import export as export_mod
from .config import CONFIG_NAME, ConfigError, load_config
from .render import RenderError, render_layer
from .store import Store
from .versions import UnknownVersionError, get_profile, supported_versions
from .worldio import RegionFile, RegionFormatError, list_regions

_TEMPLATE = """\
# Path to the force-upgraded COPY of the world (not the world you play on).
world = {world!r}
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

[export]
buffer = 1
delete_unlabeled = false
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
    world = str(args.world.resolve()) if args.world else "C:/path/to/world-copy"
    path.write_text(
        _TEMPLATE.format(world=world, versions=", ".join(supported_versions())),
        encoding="utf-8",
    )
    print(f"wrote {path}; edit it, then run `chunktriager scan`")
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


def cmd_export(args: argparse.Namespace) -> int:
    cfg, _, _ = _context(args.workspace)
    store = Store(cfg.db_path)
    try:
        path = export_mod.export_delete_selection(cfg, store)
    finally:
        store.close()
    print(f"wrote {path}")
    print("load it in MCA Selector on the ORIGINAL world (Selection > Import) and review before deleting")
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
    parser.add_argument("-w", "--workspace", type=Path, default=Path.cwd(),
                        help="workspace directory holding triage.toml (default: cwd)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="create a workspace config")
    p.add_argument("--world", type=Path, help="force-upgraded world copy")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("scan", help="build the chunk index from region files")
    p.add_argument("--allow-problems", action="store_true",
                   help="update the index even if some chunks could not be read")
    p.set_defaults(func=cmd_scan)
    p = sub.add_parser("render", help="render base-map tiles with MCA Selector")
    p.add_argument("layers", nargs="*", help="layer names (default: all)")
    p.set_defaults(func=cmd_render)
    sub.add_parser("serve", help="start the labeling UI").set_defaults(func=cmd_serve)
    sub.add_parser("export", help="write the delete selection CSV").set_defaults(func=cmd_export)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, UnknownVersionError, RenderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
