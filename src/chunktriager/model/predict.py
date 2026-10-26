"""Predictions for one world, stored in its cache folder.

``<cache>/predictions.parquet`` holds, per full chunk: the model
probability that the chunk itself is player-made, whether its only evidence
is rails, and whether protection keeps it. Chunks not in the file (not
generated to completion) are never player-made and never protected by
themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ProtectionConfig
from .dataset import World
from .evaluate import protect_two_radii
from .train import ModelBundle

FILE = "predictions.parquet"


@dataclass
class Predictions:
    positions: np.ndarray
    prob: np.ndarray
    rail_only: np.ndarray
    protected: np.ndarray
    meta: dict


def predict_world(bundle: ModelBundle, world: World, prot: ProtectionConfig) -> Predictions:
    bundle.check(world)
    prob = bundle.booster.predict(world.X).astype(np.float32)
    hit = prob >= prot.threshold
    short = world.rail_only & hit
    protected = protect_two_radii(world.positions, hit.astype(np.float32), short,
                                  prot.radius, prot.short_radius) > 0
    meta = {
        "model": bundle.meta,
        "protection": {"radius": prot.radius, "short_radius": prot.short_radius,
                       "threshold": prot.threshold},
        "chunks": len(prob),
        "flagged": int(hit.sum()),
        "protected": int(protected.sum()),
    }
    return Predictions(world.positions, prob, world.rail_only, protected, meta)


def save_predictions(cache_dir: Path, p: Predictions) -> Path:
    table = pa.table({
        "x": pa.array(p.positions[:, 0], pa.int32()),
        "z": pa.array(p.positions[:, 1], pa.int32()),
        "prob": pa.array(p.prob, pa.float32()),
        "rail_only": pa.array(p.rail_only, pa.bool_()),
        "protected": pa.array(p.protected, pa.bool_()),
    }).replace_schema_metadata({"meta": json.dumps(p.meta)})
    path = cache_dir / FILE
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)
    return path


def load_predictions(cache_dir: Path) -> Predictions | None:
    path = cache_dir / FILE
    if not path.is_file():
        return None
    t = pq.read_table(path)
    meta = json.loads(t.schema.metadata[b"meta"])
    pos = np.stack([t.column("x").to_numpy(), t.column("z").to_numpy()], axis=1)
    return Predictions(pos, t.column("prob").to_numpy(), t.column("rail_only").to_numpy(),
                       t.column("protected").to_numpy(), meta)
