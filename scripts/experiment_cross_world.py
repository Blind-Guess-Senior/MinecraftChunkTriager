"""Cross-world experiment: train on the main world, test on an all-natural world.

Usage: uv run python scripts/experiment_cross_world.py <working directory> <main world> <natural world>

Experimental code: rarity features here are not yet part of the feature schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from chunktriager.config import load_config
from chunktriager.model.dataset import FeatureSet
from chunktriager.model.evaluate import protect
from chunktriager.model.train import fit
from chunktriager.store import read_labels

BANDS = ("above", "surface", "shallow", "deep")


def load(world):
    fs = FeatureSet.load(next((world.cache_dir / "features").iterdir()))
    rows = np.flatnonzero(fs.column("chunk.full") == 1)
    keep = {p for p, label in read_labels(world.labels_path, "minecraft:overworld").items() if label == "keep"}
    pos = fs.positions[rows]
    y = np.array([tuple(p) in keep for p in pos], dtype=int)
    X, _ = fs.matrix()
    return fs, rows, pos, y, X[rows].tocsr()


def rarity_basis(fs: FeatureSet, rows: np.ndarray) -> np.ndarray:
    """Per-column surprise (-log10 presence frequency) estimated on one world."""
    freq = np.asarray((fs.own[rows] > 0).mean(axis=0)).ravel()
    ev = np.array([c.startswith(("block.", "be.", "entity.")) for c in fs.columns])
    return np.where(ev, -np.log10(np.maximum(freq, 1 / len(rows))), 0).astype(np.float32)


def rarity(fs: FeatureSet, rows: np.ndarray, sur: np.ndarray) -> sp.csr_matrix:
    masks = [np.array([c.startswith(f"block.{b}.") for c in fs.columns]) for b in BANDS]
    masks.append(np.array([c.startswith(("be.", "entity.")) for c in fs.columns]))
    out = []
    for M in (fs.own[rows].tocsr(), fs.neighbour_sum()[rows].tocsr()):
        P = (M > 0).astype(np.float32)
        for m in masks:
            for cut in (3, 4):
                sel = (m & (sur >= cut)).astype(np.float32)
                out += [np.asarray(P @ sel).ravel(), np.asarray(M @ sel).ravel()]
            out.append(np.asarray(P @ (sur * m)).ravel())
    return sp.csr_matrix(np.stack(out, axis=1))


def report(name: str, pos: np.ndarray, p: np.ndarray, biome_of: np.ndarray, names: list[str]) -> None:
    print(f"  {name}: {len(p)} natural chunks")
    for t in (3e-2, 1e-2, 3e-3):
        fp = p >= t
        kept = protect(pos, fp.astype(np.float32), 8) > 0
        print(f"    t={t:.0e}  false positives {int(fp.sum()):5d} ({fp.mean():.4f})  deletable {(~kept).mean():.3f}")
    fp = p >= 1e-2
    if fp.any():
        worst = np.bincount(biome_of[fp], minlength=len(names))
        print("    FP by dominant biome (t=1e-2):",
              ", ".join(f"{names[i]} {worst[i]}" for i in np.argsort(-worst)[:6] if worst[i]))


def main() -> None:
    cfg = load_config(Path(sys.argv[1]))
    fa, ra, pa, ya, Xa = load(cfg.world(sys.argv[2]))
    fb, rb, pb, yb, Xb = load(cfg.world(sys.argv[3]))
    assert fa.fingerprint == fb.fingerprint, "feature schemas differ"
    assert yb.sum() == 0, "natural world has player-made labels"
    bio = [i for i, c in enumerate(fa.columns) if c.startswith("biome.")]
    names = [fa.columns[i].split(":")[1] for i in bio]
    biome_b = fb.own[rb][:, bio].toarray().argmax(axis=1)
    print(f"main: {len(ya)} chunks, {ya.sum()} player-made; natural world: {len(yb)} chunks")

    # Rarity estimated on each world itself, as it would be at prediction time.
    for label, use_rarity in (("A. block counts only", False), ("B. + rarity", True)):
        Xa_ = sp.hstack([Xa, rarity(fa, ra, rarity_basis(fa, ra))], format="csr") if use_rarity else Xa
        Xb_ = sp.hstack([Xb, rarity(fb, rb, rarity_basis(fb, rb))], format="csr") if use_rarity else Xb
        print(label)
        model = fit(Xa_, ya)
        report("natural world, trained on main only", pb, model.predict(Xb_), biome_b, names)
    # C: add half the natural world to training (split by region), test on the other half.
    region_b = (pb[:, 0] // 32) * 100_000 + (pb[:, 1] // 32)
    test = (np.unique(region_b, return_inverse=True)[1] % 2) == 1
    Xa_ = sp.hstack([Xa, rarity(fa, ra, rarity_basis(fa, ra))], format="csr")
    Xb_ = sp.hstack([Xb, rarity(fb, rb, rarity_basis(fb, rb))], format="csr")
    model = fit(sp.vstack([Xa_, Xb_[~test]], format="csr"), np.concatenate([ya, yb[~test]]))
    print("C. + rarity, training adds half of the natural world (other half tested)")
    report("held-out half of natural world", pb[test], model.predict(Xb_[test]), biome_b[test], names)


if __name__ == "__main__":
    main()
