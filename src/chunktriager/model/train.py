"""Model training and spatially grouped cross-validation.

The model predicts whether a chunk itself contains player-made content.
Labelled chunks near each other are strongly correlated, so validation
groups them into "sites": chunks within ``GROUP_RADIUS`` of each other end up
in the same fold, together with the natural chunks around them. This keeps
a building from appearing on both sides of a split.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

GROUP_RADIUS = 16      # chunks; natural chunks join the nearest site within this range
CELL = 64              # chunks; natural chunks far from any site are grouped by grid cell

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 0,
}
ROUNDS = 400


def site_groups(positions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Group id per labelled chunk; positive chunks connected within GROUP_RADIUS share a site."""
    pos = [tuple(p) for p in positions]
    positive = [i for i in range(len(pos)) if y[i] == 1]
    group = np.full(len(pos), -1, dtype=np.int64)
    sid = 0
    for i in positive:
        if group[i] >= 0:
            continue
        group[i] = sid
        stack = [i]
        while stack:
            a = stack.pop()
            for b in positive:
                if group[b] < 0 and max(abs(pos[a][0] - pos[b][0]), abs(pos[a][1] - pos[b][1])) <= GROUP_RADIUS:
                    group[b] = sid
                    stack.append(b)
        sid += 1
    site_points = positions[y == 1]
    site_ids = group[y == 1]
    for i in np.flatnonzero(group < 0):
        if len(site_points):
            d = np.abs(site_points - positions[i]).max(axis=1)
            j = int(np.argmin(d))
            if d[j] <= GROUP_RADIUS:
                group[i] = site_ids[j]
                continue
        cx, cz = positions[i] // CELL
        group[i] = 1_000_000 + (int(cx) + 10_000) * 20_000 + (int(cz) + 10_000)
    return group


def assign_folds(groups: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Fold per row; groups with positives are spread first so every fold gets some."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    pos_count = {g: int(y[groups == g].sum()) for g in ids}
    with_pos = sorted((g for g in ids if pos_count[g]), key=lambda g: -pos_count[g])
    without = [g for g in ids if not pos_count[g]]
    rng.shuffle(without)
    fold_of, load = {}, np.zeros(k)
    for g in with_pos:                          # greedy balance of positives
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += pos_count[g]
    for i, g in enumerate(without):
        fold_of[g] = i % k
    return np.array([fold_of[g] for g in groups])


def fit(X: sp.csr_matrix, y: np.ndarray, rounds: int = ROUNDS) -> lgb.Booster:
    # Natural chunks vastly outnumber player-made ones; weight positives so
    # both classes carry equal total weight.
    w = np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, free_raw_data=False), rounds)


@dataclass
class CVResult:
    oof: np.ndarray          # out-of-fold probability per labelled row
    folds: np.ndarray
    groups: np.ndarray


def cross_validate(X: sp.csr_matrix, y: np.ndarray, positions: np.ndarray, k: int = 5) -> CVResult:
    groups = site_groups(positions, y)
    folds = assign_folds(groups, y, k)
    oof = np.zeros(len(y))
    for f in range(k):
        test = folds == f
        if not test.any() or y[~test].sum() == 0:
            continue
        model = fit(X[~test], y[~test])
        oof[test] = model.predict(X[test])
    return CVResult(oof, folds, groups)


# Model files --------------------------------------------------------------------

MODEL_FORMAT = 1


@dataclass
class ModelBundle:
    booster: lgb.Booster
    meta: dict

    def save(self, directory) -> None:
        import json
        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.txt"))
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory) -> "ModelBundle":
        import json
        from pathlib import Path

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != MODEL_FORMAT:
            raise ValueError(f"model format {meta.get('format')} is not supported")
        return cls(lgb.Booster(model_file=str(directory / "model.txt")), meta)

    def check(self, world) -> None:
        """Refuse to predict on features the model was not trained on."""
        from .derived import DERIVED_VERSION

        if world.fingerprint != self.meta["feature_fingerprint"]:
            raise ValueError(f"feature schema {world.fingerprint} does not match the model's "
                             f"{self.meta['feature_fingerprint']}; re-run `features` or retrain")
        if self.meta["derived_version"] != DERIVED_VERSION or len(world.columns) != self.meta["n_features"]:
            raise ValueError("derived feature version does not match the model; retrain")


def train_worlds(worlds, game_version: str) -> ModelBundle:
    import datetime

    from .derived import DERIVED_VERSION

    fingerprints = {w.fingerprint for w in worlds}
    if len(fingerprints) != 1:
        raise ValueError(f"worlds use different feature schemas: {sorted(fingerprints)}")
    Xs, ys = [], []
    for w in worlds:
        use = w.y >= 0
        Xs.append(w.X[use])
        ys.append(w.y[use].astype(np.int32))
    X, y = sp.vstack(Xs, format="csr"), np.concatenate(ys)
    if y.sum() == 0:
        raise ValueError("no player-made chunks in the training worlds")
    booster = fit(X, y)
    meta = {
        "format": MODEL_FORMAT,
        "game_version": game_version,
        "feature_fingerprint": worlds[0].fingerprint,
        "derived_version": DERIVED_VERSION,
        "n_features": X.shape[1],
        "params": PARAMS, "rounds": ROUNDS,
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "worlds": [{"name": w.name, "chunks": int((w.y >= 0).sum()), "player_made": int((w.y == 1).sum())}
                   for w in worlds],
    }
    return ModelBundle(booster, meta)
