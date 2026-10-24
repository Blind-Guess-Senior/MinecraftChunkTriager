"""Local labeling server. Binds to loopback by default and has no auth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import export as export_mod
from .config import Config
from .render import tile_path
from .store import Store
from .versions import UnsupportedChunkError, VersionProfile
from .worldio import REGION_CHUNKS, RegionFile, RegionFormatError, RegionPos

WEB_DIR = Path(__file__).parent / "web"
MAX_RECT_CHUNKS = 1_000_000
# Region state string alphabet: absent, unlabeled, keep, delete.
_STATE_CHAR = {None: "u", "keep": "k", "delete": "d"}


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


def create_app(cfg: Config, profile: VersionProfile, dimension_dir: Path) -> FastAPI:
    store = Store(cfg.db_path)
    dim = cfg.dimension
    region_dir = dimension_dir / "region"
    # The chunk index only changes through `chunktriager scan`, which is run
    # while the server is stopped, so it is loaded once here.
    chunks = store.chunks(dim)
    present = {(x // REGION_CHUNKS, z // REGION_CHUNKS) for x, z in chunks}
    app = FastAPI(title="chunktriager")

    @app.get("/api/meta")
    def meta() -> dict:
        regions = sorted(present)
        rendered = [l.name for l in cfg.layers if (cfg.tiles_dir / l.name).is_dir()]
        zooms = [
            int(p.name) for name in rendered[:1]
            for p in (cfg.tiles_dir / name).iterdir() if p.is_dir()
        ]
        return {
            "dimension": dim,
            "game_version": profile.name,
            "layers": [{"name": l.name, "title": l.title} for l in cfg.layers if l.name in rendered],
            "min_zoom": min(zooms, default=0),
            "regions": regions,
            "chunk_count": len(chunks),
            "counts": store.label_counts(dim),
        }

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

    @app.post("/api/undo")
    def undo() -> dict:
        return {"reverted": store.undo(dim), "counts": store.label_counts(dim)}

    @app.get("/api/export/preview")
    def export_preview() -> dict:
        plan = _plan()
        return {
            "delete": len(plan.delete),
            "kept_by_label": plan.kept_by_label,
            "kept_by_buffer": plan.kept_by_buffer,
            "kept_unlabeled": plan.kept_unlabeled,
            "buffer": cfg.export.buffer,
        }

    @app.post("/api/export")
    def export() -> dict:
        path = export_mod.export_delete_selection(cfg, store)
        return {"path": str(path), "delete": len(_plan().delete)}

    def _plan() -> export_mod.ExportPlan:
        labels = {p: v[0] for p, v in store.labels(dim).items()}
        return export_mod.plan_export(chunks, labels, cfg.export.buffer, cfg.export.delete_unlabeled)

    @app.get("/tiles/{layer}/{z}/{x}/{y}.png")
    def tile(layer: str, z: int, x: int, y: int) -> Response:
        try:
            cfg.layer(layer)
        except KeyError:
            raise HTTPException(404) from None
        path = tile_path(cfg.tiles_dir, layer, z, x, y)
        if not path.is_file():
            return Response(status_code=204)
        return FileResponse(path, media_type="image/png")

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
