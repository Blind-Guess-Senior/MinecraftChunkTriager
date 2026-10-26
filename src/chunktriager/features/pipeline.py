"""Feature extraction for a whole dimension, one process per region file.

A region is skipped when its output file is newer than both its region and
entities files, so re-running only redoes what changed. Chunks the version
profile rejects (e.g. not upgraded) are skipped and counted.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..versions import UnsupportedChunkError, get_profile
from ..versions.vocab import load_vocabulary
from ..worldio import RegionFile, RegionFormatError, RegionPos, list_regions
from .extract import Extractor
from .schema import build_layout
from .store import table_dir, write_manifest, write_region


@dataclass(frozen=True)
class RegionResult:
    region: RegionPos
    chunks: int = 0
    rejected: int = 0
    skipped: bool = False
    error: str | None = None


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def _entities_by_chunk(profile, path: Path) -> dict[tuple[int, int], list]:
    if not path.is_file() or path.stat().st_size <= 2 * 4096:
        return {}
    result = {}
    for cx, cz, root in RegionFile(path).iter_chunks():
        result[(cx, cz)] = profile.read_entities(root)
    return result


# Worker state, built once per process.
_worker = None


def _init_worker(game_version: str) -> None:
    global _worker
    profile = get_profile(game_version)
    vocab = load_vocabulary(profile.vocabulary)
    _worker = (profile, Extractor(vocab, build_layout(vocab)))


def _process_region(dimension_dir: Path, out_dir: Path, rpos: RegionPos, force: bool) -> RegionResult:
    profile, extractor = _worker
    region_path = dimension_dir / "region" / rpos.filename
    entity_path = dimension_dir / "entities" / rpos.filename
    out_path = out_dir / f"r.{rpos.x}.{rpos.z}.parquet"
    if not force and _mtime(out_path) > max(_mtime(region_path), _mtime(entity_path)):
        return RegionResult(rpos, skipped=True)
    try:
        entities = _entities_by_chunk(profile, entity_path)
        positions, rows, rejected = [], [], 0
        for cx, cz, root in RegionFile(region_path).iter_chunks():
            try:
                chunk = profile.read_chunk(root)
            except UnsupportedChunkError:
                rejected += 1
                continue
            positions.append((cx, cz))
            rows.append(extractor.row(chunk, entities.get((cx, cz), [])))
        write_region(out_path, positions, rows)
        return RegionResult(rpos, chunks=len(rows), rejected=rejected)
    except (RegionFormatError, OSError, KeyError, ValueError) as e:
        return RegionResult(rpos, error=f"{type(e).__name__}: {e}")


def extract_dimension(
    workspace: Path,
    dimension_dir: Path,
    game_version: str,
    workers: int | None = None,
    force: bool = False,
    progress: Callable[[int, int, RegionResult], None] = lambda done, total, r: None,
) -> tuple[Path, list[RegionResult]]:
    profile = get_profile(game_version)
    vocab = load_vocabulary(profile.vocabulary)
    layout = build_layout(vocab)
    out_dir = table_dir(workspace, layout.schema)
    write_manifest(out_dir, layout.schema, game_version)
    regions = list_regions(dimension_dir / "region")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    results = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(game_version,)) as pool:
        futures = [pool.submit(_process_region, dimension_dir, out_dir, r, force) for r in regions]
        for f in as_completed(futures):
            results.append(f.result())
            progress(len(results), len(regions), results[-1])
    return out_dir, sorted(results, key=lambda r: (r.region.x, r.region.z))
