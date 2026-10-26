"""Base-map rendering through the MCA Selector CLI, cut into Leaflet tiles.

Tile pyramid: zoom 0 is native resolution (1 px per block) with one
512 px tile per region, so tile ``(x, y)`` at zoom 0 is region ``(x, z)``.
Zoom ``-k`` tiles cover ``2^k`` regions per side.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image

from .config import McaSelectorConfig, RenderLayer
from .worldio import RegionPos

TILE = 512
MAX_LEVELS = 10


class RenderError(Exception):
    pass


def tile_path(root: Path, layer: str, z: int, x: int, y: int) -> Path:
    return root / layer / str(z) / str(x) / f"{y}.png"


def group_blocks(regions: Iterable[RegionPos], size: int) -> list[list[RegionPos]]:
    """Group regions into ``size``-by-``size`` blocks, one MCA Selector call each."""
    blocks: dict[tuple[int, int], list[RegionPos]] = {}
    for r in regions:
        blocks.setdefault((r.x // size, r.z // size), []).append(r)
    return [sorted(v) for _, v in sorted(blocks.items())]


def write_region_selection(path: Path, regions: Iterable[RegionPos]) -> None:
    # MCA Selector selection CSV: one "rx;rz" line selects a whole region.
    path.write_text("".join(f"{r.x};{r.z}\n" for r in regions), encoding="utf-8")


def _isolated_env(home: Path) -> dict[str, str]:
    # MCA Selector reuses cached region images from its cache directory, and
    # those may have been rendered with other settings (e.g. by the GUI).
    # Pointing its per-user directories at an empty temp dir forces a fresh
    # render with the arguments we pass.
    env = dict(os.environ)
    for key in ("LOCALAPPDATA", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        env[key] = str(home)
    return env


def run_mcaselector(
    mca: McaSelectorConfig, dimension_dir: Path, regions: list[RegionPos], layer: RenderLayer, work: Path
) -> Path:
    selection = work / "selection.csv"
    output = work / "out.png"
    write_region_selection(selection, regions)
    home = work / "home"
    home.mkdir(exist_ok=True)
    cmd = [
        *mca.command,
        "--mode", "image",
        "--world", str(dimension_dir),
        "--selection", str(selection),
        "--output", str(output),
        *layer.args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=_isolated_env(home),
            capture_output=True,
            text=True,
            timeout=mca.timeout,
        )
    except FileNotFoundError:
        raise RenderError(f"cannot start MCA Selector: {cmd[0]!r} not found") from None
    except subprocess.TimeoutExpired:
        raise RenderError(f"MCA Selector timed out after {mca.timeout}s") from None
    # The CLI reports argument errors on stdout and still exits with 0, so
    # the output file is the only reliable success signal.
    if proc.returncode != 0 or not output.is_file():
        log = (proc.stdout + proc.stderr).strip()[-2000:]
        raise RenderError(f"MCA Selector produced no image (exit {proc.returncode}):\n{log}")
    return output


def cut_block_image(image_path: Path, regions: list[RegionPos], out_root: Path, layer: str) -> None:
    """Split one MCA Selector image into zoom-0 region tiles.

    The image covers the bounding box of the selected regions, starting at
    the smallest region x and z, with 512 px per region.
    """
    min_x = min(r.x for r in regions)
    min_z = min(r.z for r in regions)
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        for r in regions:
            left, top = (r.x - min_x) * TILE, (r.z - min_z) * TILE
            if left + TILE > img.width or top + TILE > img.height:
                raise RenderError(f"image {img.size} does not cover region {r.filename}")
            tile = img.crop((left, top, left + TILE, top + TILE))
            path = tile_path(out_root, layer, 0, r.x, r.z)
            path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(path, optimize=False)


def build_pyramid(out_root: Path, layer: str, levels: int) -> None:
    """Build zoom -1 .. -levels by 2x downsampling the level above."""
    for z in range(0, -levels, -1):
        src_dir = out_root / layer / str(z)
        if not src_dir.is_dir():
            return
        parents: set[tuple[int, int]] = set()
        for x_dir in src_dir.iterdir():
            for f in x_dir.glob("*.png"):
                parents.add((int(x_dir.name) // 2, int(f.stem) // 2))
        for px, py in parents:
            canvas = Image.new("RGBA", (TILE * 2, TILE * 2), (0, 0, 0, 0))
            for dx in (0, 1):
                for dy in (0, 1):
                    child = tile_path(out_root, layer, z, px * 2 + dx, py * 2 + dy)
                    if child.is_file():
                        with Image.open(child) as c:
                            canvas.paste(c.convert("RGBA"), (dx * TILE, dy * TILE))
            path = tile_path(out_root, layer, z - 1, px, py)
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.resize((TILE, TILE), Image.Resampling.BOX).save(path)


def pyramid_levels(regions: Iterable[RegionPos]) -> int:
    """Levels needed until the whole map fits in a few tiles."""
    regions = list(regions)
    if not regions:
        return 0
    span = max(
        max(r.x for r in regions) - min(r.x for r in regions),
        max(r.z for r in regions) - min(r.z for r in regions),
    ) + 1
    levels = 0
    while span > 2 and levels < MAX_LEVELS:
        span = (span + 1) // 2
        levels += 1
    return levels


def render_layer(
    mca: McaSelectorConfig,
    dimension_dir: Path,
    regions: list[RegionPos],
    layer: RenderLayer,
    out_root: Path,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> None:
    staging = out_root / f".{layer.name}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    blocks = group_blocks(regions, mca.block_regions)
    for i, block in enumerate(blocks):
        with tempfile.TemporaryDirectory(prefix="chunktriager-") as tmp:
            image = run_mcaselector(mca, dimension_dir, block, layer, Path(tmp))
            cut_block_image(image, block, staging, layer.name)
        progress(i + 1, len(blocks))
    build_pyramid(staging, layer.name, pyramid_levels(regions))
    # Swap in the finished layer only after every step succeeded.
    final = out_root / layer.name
    shutil.rmtree(final, ignore_errors=True)
    (staging / layer.name).rename(final)
    shutil.rmtree(staging, ignore_errors=True)
