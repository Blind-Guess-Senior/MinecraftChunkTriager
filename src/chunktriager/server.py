"""Local labeling server. Binds to loopback by default and has no auth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Config, World
from .features.store import read_positions
from .model.predict import FILE, load_predictions
from .proposals import EXTRA, KINDS, from_features
from .render import tile_path
from .store import Store
from .versions import UnsupportedChunkError, VersionProfile
from .worldio import REGION_CHUNKS, RegionFile, RegionFormatError, RegionPos

WEB_DIR = Path(__file__).parent / "web"
MAX_RECT_CHUNKS = 1_000_000
# Region state string alphabet: absent, unlabeled, keep, delete.
_STATE_CHAR = {None: "u", "keep": "k", "delete": "d"}


class AcceptRequest(BaseModel):
    # Proposal kinds to write as player-made; chunks with a label are left alone.
    kinds: list[Literal["e", "s"]] = Field(default_factory=lambda: [EXTRA])


class LabelRequest(BaseModel):
    label: Literal["keep", "delete"] | None
    source: Literal["single", "area", "model"] = "single"
    chunks: list[tuple[int, int]] = Field(default_factory=list)
    # Inclusive chunk rectangle [x0, z0, x1, z1]; combined with ``chunks``.
    rect: tuple[int, int, int, int] | None = None


def _region_state(chunks: set[tuple[int, int]], labels: dict, rx: int, rz: int) -> str:
    out = []
    for lz in range(REGION_CHUNKS):
        for lx in range(REGION_CHUNKS):
            p = (rx * REGION_CHUNKS + lx, rz * REGION_CHUNKS + lz)
            if p not in chunks:
                out.append(".")
            else:
                entry = labels.get(p)
                out.append(_STATE_CHAR[entry[0] if entry else None])
    return "".join(out)


class _PredictionCache:
    """Predictions file, reloaded when ``chunktriager predict`` rewrites it."""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / FILE
        self.cache_dir = cache_dir
        self.mtime = None
        self.p = None
        self.by_region: dict[tuple[int, int], tuple[bytearray, bytearray]] = {}

    def get(self):
        m = self.path.stat().st_mtime if self.path.is_file() else None
        if m != self.mtime:
            self.mtime = m
            self.p = load_predictions(self.cache_dir) if m else None
            self.by_region = {}
            if self.p is not None:
                for (x, z), prob, prot in zip(self.p.positions.tolist(), self.p.prob.tolist(),
                                              self.p.protected.tolist()):
                    key = (x // REGION_CHUNKS, z // REGION_CHUNKS)
                    if key not in self.by_region:
                        self.by_region[key] = (bytearray(REGION_CHUNKS ** 2), bytearray(b"." * REGION_CHUNKS ** 2))
                    i = (z % REGION_CHUNKS) * REGION_CHUNKS + x % REGION_CHUNKS
                    self.by_region[key][0][i] = min(255, round(prob * 255))
                    self.by_region[key][1][i] = ord("p" if prot else "-")
        return self.p

    def summary(self) -> dict | None:
        p = self.get()
        if p is None:
            return None
        m = p.meta
        return {"threshold": m["protection"]["threshold"], "flagged": m["flagged"],
                "protected": m["protected"], "chunks": m["chunks"],
                "trained_at": m["model"]["trained_at"]}

    def regions(self, rx0, rz0, rx1, rz1) -> dict:
        self.get()
        return {f"{rx},{rz}": {"prob": v[0].hex(), "protected": v[1].decode()}
                for (rx, rz), v in self.by_region.items() if rx0 <= rx <= rx1 and rz0 <= rz <= rz1}


def create_app(cfg: Config, world: World, profile: VersionProfile, dimension_dir: Path,
               features_dir: Path) -> FastAPI:
    store = Store(world.labels_path)
    dim = cfg.dimension
    region_dir = dimension_dir / "region"
    tiles_dir = world.tiles_dir
    # The chunk index is the set of chunks with a feature row. It only
    # changes through `chunktriager prepare`, so it is loaded once here.
    chunks = read_positions(features_dir)
    present = {(x // REGION_CHUNKS, z // REGION_CHUNKS) for x, z in chunks}
    app = FastAPI(title="chunktriager")
    preds = _PredictionCache(world.cache_dir)
    proposals = from_features(features_dir, profile.read_generator(world.path, dim))
    proposals_by_region: dict[tuple[int, int], bytearray] = {}
    if proposals is not None:
        for (x, z), kind in proposals.kind.items():
            key = (x // REGION_CHUNKS, z // REGION_CHUNKS)
            row = proposals_by_region.setdefault(key, bytearray(b"." * REGION_CHUNKS ** 2))
            row[(z % REGION_CHUNKS) * REGION_CHUNKS + x % REGION_CHUNKS] = ord(kind)

    def proposal_summary() -> dict | None:
        if proposals is None:
            return None
        labels = store.labels(dim)
        pending = dict.fromkeys(KINDS, 0)
        for p, kind in proposals.kind.items():
            if p not in labels:
                pending[kind] += 1
        return {"baseline": proposals.baseline, "counts": proposals.counts(), "pending": pending}

    @app.get("/api/meta")
    def meta() -> dict:
        regions = sorted(present)
        rendered = [l.name for l in cfg.layers if (tiles_dir / l.name).is_dir()]
        zooms = [
            int(p.name) for name in rendered[:1]
            for p in (tiles_dir / name).iterdir() if p.is_dir()
        ]
        return {
            "world": world.name,
            "dimension": dim,
            "game_version": profile.name,
            "layers": [{"name": l.name, "title": l.title} for l in cfg.layers if l.name in rendered],
            "min_zoom": min(zooms, default=0),
            "regions": regions,
            "chunk_count": len(chunks),
            "counts": store.label_counts(dim),
            "predictions": preds.summary(),
            "proposals": proposal_summary(),
        }

    @app.get("/api/proposals")
    def proposal_regions(rx0: int, rz0: int, rx1: int, rz1: int) -> dict:
        """Per region: 1024 chars, "." none, "e" extra content, "s" extra content in a structure."""
        if (rx1 - rx0 + 1) * (rz1 - rz0 + 1) > 4096:
            raise HTTPException(400, "region range too large")
        return {f"{rx},{rz}": v.decode() for (rx, rz), v in proposals_by_region.items()
                if rx0 <= rx <= rx1 and rz0 <= rz <= rz1}

    @app.post("/api/proposals/accept")
    def accept_proposals(req: AcceptRequest) -> dict:
        """Write unlabeled proposals of the given kinds as player-made, as one undoable batch."""
        if proposals is None:
            raise HTTPException(404, "this world has no generator baseline")
        labels = store.labels(dim)
        targets = [p for p, kind in proposals.kind.items() if kind in req.kinds and p not in labels]
        changed = store.set_labels(dim, targets, "keep", "proposal", known=chunks)
        return {"changed": changed, "counts": store.label_counts(dim), "proposals": proposal_summary()}

    @app.get("/api/predictions")
    def predictions(rx0: int, rz0: int, rx1: int, rz1: int) -> dict:
        """Per region: 1024 probability bytes (0-255, hex) and a protection string."""
        if (rx1 - rx0 + 1) * (rz1 - rz0 + 1) > 4096:
            raise HTTPException(400, "region range too large")
        return preds.regions(rx0, rz0, rx1, rz1)

    @app.get("/api/review")
    def review(kind: str, limit: int = 200) -> list:
        """Chunks where the model and the labels disagree, worst first.

        ``missed``: labelled player-made, low probability, not protected.
        ``flagged``: not labelled player-made, probability at or above the threshold.
        """
        p = preds.get()
        if p is None:
            return []
        labels = store.labels(dim)
        rows = []
        for (x, z), prob, prot in zip(map(tuple, p.positions.tolist()), p.prob.tolist(), p.protected.tolist()):
            human = labels.get((x, z), (None,))[0] == "keep"
            if kind == "missed" and human and not prot:
                rows.append((prob, x, z))
            elif kind == "flagged" and not human and prob >= p.meta["protection"]["threshold"]:
                rows.append((-prob, x, z))
        rows.sort()
        return [{"x": x, "z": z, "prob": abs(v)} for v, x, z in rows[:limit]]

    @app.get("/api/state")
    def state(rx0: int, rz0: int, rx1: int, rz1: int) -> dict:
        if (rx1 - rx0 + 1) * (rz1 - rz0 + 1) > 4096:
            raise HTTPException(400, "region range too large")
        bbox = (rx0 * REGION_CHUNKS, rz0 * REGION_CHUNKS,
                (rx1 + 1) * REGION_CHUNKS - 1, (rz1 + 1) * REGION_CHUNKS - 1)
        labels = store.labels(dim, bbox)
        return {
            f"{rx},{rz}": _region_state(chunks, labels, rx, rz)
            for rx in range(rx0, rx1 + 1) for rz in range(rz0, rz1 + 1)
            if (rx, rz) in present
        }

    @app.get("/api/chunk/{x}/{z}")
    def chunk(x: int, z: int) -> dict:
        if (x, z) not in chunks:
            raise HTTPException(404, "chunk not in index")
        rpos = RegionPos(x // REGION_CHUNKS, z // REGION_CHUNKS)
        try:
            root = RegionFile(region_dir / rpos.filename).read_chunk(x, z)
            if root is None:
                raise HTTPException(404, "chunk no longer present in region file")
            info = profile.parse_chunk(root)
        except (RegionFormatError, UnsupportedChunkError, OSError) as e:
            raise HTTPException(422, str(e)) from None
        entry = store.labels(dim, (x, z, x, z)).get((x, z))
        counts = Counter(be.id for be in info.block_entities)
        return {
            "x": x, "z": z,
            "region": rpos.filename,
            "data_version": info.data_version,
            "label": entry[0] if entry else None,
            "source": entry[1] if entry else None,
            "block_entities": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
            "tp": f"/tp @s {x * 16 + 8} ~ {z * 16 + 8}",
        }

    @app.post("/api/labels")
    def set_labels(req: LabelRequest) -> dict:
        positions = list(req.chunks)
        if req.rect is not None:
            x0, z0, x1, z1 = req.rect
            x0, x1 = sorted((x0, x1))
            z0, z1 = sorted((z0, z1))
            if (x1 - x0 + 1) * (z1 - z0 + 1) > MAX_RECT_CHUNKS:
                raise HTTPException(400, "rectangle too large")
            positions += [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)]
        changed = store.set_labels(dim, positions, req.label, req.source, known=chunks)
        return {"changed": changed, "counts": store.label_counts(dim)}

    @app.post("/api/labels/clear")
    def clear_labels() -> dict:
        return {"changed": store.clear_labels(dim), "counts": store.label_counts(dim)}

    @app.post("/api/undo")
    def undo() -> dict:
        return {"reverted": store.undo(dim), "counts": store.label_counts(dim)}

    @app.get("/tiles/{layer}/{z}/{x}/{y}.png")
    def tile(layer: str, z: int, x: int, y: int) -> Response:
        try:
            cfg.layer(layer)
        except KeyError:
            raise HTTPException(404) from None
        path = tile_path(tiles_dir, layer, z, x, y)
        if not path.is_file():
            return Response(status_code=204)
        return FileResponse(path, media_type="image/png")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
