// Map coordinates: CRS.Simple at zoom 0 is 1 px per block, and a block at
// (x, z) sits at latLng(-z, x). Zoom-0 tile (x, y) is region (x, z).
"use strict";

const REGION = 32;          // chunks per region side
const CHUNK = 16;           // blocks per chunk side
const RGB = { k: [46, 139, 87], d: [192, 57, 43] };

const state = {
  tool: "keep",
  grid: true,
  opacity: 0.45,
  regions: new Map(),       // "rx,rz" -> 1024-char state string
  pending: new Set(),
};

const api = async (path, opts = {}) => {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
};

const toLatLng = (bx, bz) => L.latLng(-bz, bx);
const toBlock = (ll) => [Math.floor(ll.lng), Math.floor(-ll.lat)];
const toChunk = (ll) => toBlock(ll).map((b) => Math.floor(b / CHUNK));
const regionOf = (c) => Math.floor(c / REGION);
const mod = (a, n) => ((a % n) + n) % n;

function chunkState(cx, cz) {
  const s = state.regions.get(`${regionOf(cx)},${regionOf(cz)}`);
  return s ? s[mod(cz, REGION) * REGION + mod(cx, REGION)] : undefined;
}

async function loadRegions(rx0, rz0, rx1, rz1, force = false) {
  const missing = [];
  for (let rx = rx0; rx <= rx1; rx++)
    for (let rz = rz0; rz <= rz1; rz++) {
      const key = `${rx},${rz}`;
      if (force || !state.regions.has(key)) missing.push(key);
    }
  if (!missing.length) return;
  const data = await api(`/api/state?rx0=${rx0}&rz0=${rz0}&rx1=${rx1}&rz1=${rz1}`);
  for (const key of missing) state.regions.set(key, data[key] || "");
}

// Canvas grid layer that draws labels and chunk/region lines for each tile.
const LabelLayer = L.GridLayer.extend({
  createTile(coords, done) {
    const tile = document.createElement("canvas");
    const size = this.getTileSize();
    tile.width = size.x;
    tile.height = size.y;
    const scale = Math.pow(2, coords.z);              // px per block
    const span = REGION / scale;                      // chunks per tile side
    const cx0 = coords.x * span, cz0 = coords.y * span;
    const rx0 = regionOf(cx0), rz0 = regionOf(cz0);
    const rx1 = regionOf(cx0 + span - 1), rz1 = regionOf(cz0 + span - 1);
    loadRegions(rx0, rz0, rx1, rz1)
      .then(() => { this.paint(tile, cx0, cz0, span, scale); done(null, tile); })
      .catch((e) => done(e, tile));
    return tile;
  },

  paint(tile, cx0, cz0, span, scale) {
    const ctx = tile.getContext("2d");
    ctx.clearRect(0, 0, tile.width, tile.height);
    const px = CHUNK * scale;                         // px per chunk
    // One pixel per chunk, then scaled up without smoothing.
    const bitmap = new ImageData(span, span);
    const alpha = Math.round(state.opacity * 255);
    for (let dz = 0; dz < span; dz++)
      for (let dx = 0; dx < span; dx++) {
        const c = RGB[chunkState(cx0 + dx, cz0 + dz)];
        if (!c) continue;
        const i = (dz * span + dx) * 4;
        bitmap.data[i] = c[0]; bitmap.data[i + 1] = c[1]; bitmap.data[i + 2] = c[2];
        bitmap.data[i + 3] = alpha;
      }
    const small = document.createElement("canvas");
    small.width = small.height = span;
    small.getContext("2d").putImageData(bitmap, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(small, 0, 0, tile.width, tile.height);
    if (!state.grid) return;
    ctx.lineWidth = 1;
    if (px >= 6) {
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.beginPath();
      for (let i = 0; i <= span; i++) {
        ctx.moveTo(i * px + 0.5, 0); ctx.lineTo(i * px + 0.5, tile.height);
        ctx.moveTo(0, i * px + 0.5); ctx.lineTo(tile.width, i * px + 0.5);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath();
    const rpx = REGION * px;
    for (let i = 0; i * rpx <= tile.width; i++) {
      const off = mod(-cx0, REGION) * px + i * rpx;
      ctx.moveTo(off + 0.5, 0); ctx.lineTo(off + 0.5, tile.height);
    }
    for (let i = 0; i * rpx <= tile.height; i++) {
      const off = mod(-cz0, REGION) * px + i * rpx;
      ctx.moveTo(0, off + 0.5); ctx.lineTo(tile.width, off + 0.5);
    }
    ctx.stroke();
  },
});

const map = L.map("map", {
  crs: L.CRS.Simple,
  zoomSnap: 1,
  boxZoom: false,            // Shift+drag is used for area labeling
  doubleClickZoom: false,
  attributionControl: false,
});
// Zoom 5 is one chunk per 512 px tile; below LABEL_MIN_ZOOM a tile would
// cover too many chunks for a per-chunk bitmap.
const MAX_ZOOM = 5;
const LABEL_MIN_ZOOM = -5;
const labels = new LabelLayer({ tileSize: 512, maxZoom: MAX_ZOOM, zIndex: 10 });
let baseLayer = null;

function setBaseLayer(name, meta) {
  if (baseLayer) map.removeLayer(baseLayer);
  baseLayer = L.tileLayer(`/tiles/${name}/{z}/{x}/{y}.png`, {
    tileSize: 512,
    minZoom: meta.min_zoom,
    minNativeZoom: meta.min_zoom,
    maxNativeZoom: 0,
    maxZoom: MAX_ZOOM,
    noWrap: true,
  }).addTo(map);
  baseLayer.bringToBack();
}

function redrawLabels() {
  labels.redraw();
}

function renderCounts(counts) {
  document.getElementById("counts").innerHTML =
    `<span>保留 ${counts.keep}</span><span>删除 ${counts.delete}</span>` +
    `<span>未标注 ${state.total - counts.keep - counts.delete}</span>`;
}

async function refreshExportPreview() {
  const p = await api("/api/export/preview");
  document.getElementById("export-preview").textContent =
    `将删除 ${p.delete} 个区块；保留标签 ${p.kept_by_label}，缓冲区(${p.buffer}) ${p.kept_by_buffer}，未标注 ${p.kept_unlabeled}`;
}

async function applyLabel(body) {
  const label = state.tool === "clear" ? null : state.tool;
  const res = await api("/api/labels", { method: "POST", body: { label, ...body } });
  await afterChange(res.counts, body);
}

async function afterChange(counts, body) {
  // Reload label state for the affected regions (or everything after undo).
  if (body && body.rect) {
    const [x0, z0, x1, z1] = body.rect;
    await loadRegions(regionOf(Math.min(x0, x1)), regionOf(Math.min(z0, z1)),
                      regionOf(Math.max(x0, x1)), regionOf(Math.max(z0, z1)), true);
  } else if (body && body.chunks) {
    for (const [x, z] of body.chunks) await loadRegions(regionOf(x), regionOf(z), regionOf(x), regionOf(z), true);
  } else {
    state.regions.clear();
  }
  renderCounts(counts);
  redrawLabels();
  refreshExportPreview();
}

function setTool(tool) {
  state.tool = tool;
  for (const b of document.querySelectorAll("[data-label]"))
    b.classList.toggle("active", b.dataset.label === tool);
}

async function showDetail(cx, cz) {
  const el = document.getElementById("detail-body");
  if (chunkState(cx, cz) === "." || chunkState(cx, cz) === undefined) {
    el.innerHTML = `<div class="muted">(${cx}, ${cz}) 未生成</div>`;
    return;
  }
  try {
    const d = await api(`/api/chunk/${cx}/${cz}`);
    const rows = d.block_entities.map(([id, n]) => `<tr><td>${id}</td><td>${n}</td></tr>`).join("");
    el.innerHTML =
      `<div>区块 (${d.x}, ${d.z}) · ${d.region}</div>` +
      `<div>标签：${d.label ?? "未标注"}${d.source ? ` (${d.source})` : ""}</div>` +
      `<div><code>${d.tp}</code></div>` +
      `<h2>方块实体</h2>` + (rows ? `<table>${rows}</table>` : `<div class="muted">无</div>`);
  } catch (e) {
    el.textContent = String(e);
  }
}

// Shift + drag: rectangle selection in chunk units.
// Capture phase, so Leaflet's own pan drag never sees a Shift+mousedown.
let drag = null;
map.getContainer().addEventListener("mousedown", (ev) => {
  if (!ev.shiftKey || ev.button !== 0) return;
  ev.stopPropagation();
  ev.preventDefault();
  const ll = map.mouseEventToLatLng(ev);
  drag = { start: toChunk(ll), rect: L.rectangle([ll, ll], { className: "box-select", interactive: false }).addTo(map) };
}, true);
map.on("mousemove", (e) => {
  const [bx, bz] = toBlock(e.latlng);
  const [cx, cz] = toChunk(e.latlng);
  document.getElementById("cursor").textContent = `方块 ${bx}, ${bz} · 区块 ${cx}, ${cz} · r.${regionOf(cx)}.${regionOf(cz)}`;
  if (!drag) return;
  const [x0, z0] = drag.start;
  const [x1, z1] = [cx, cz];
  drag.end = [x1, z1];
  drag.rect.setBounds([
    toLatLng(Math.min(x0, x1) * CHUNK, Math.min(z0, z1) * CHUNK),
    toLatLng((Math.max(x0, x1) + 1) * CHUNK, (Math.max(z0, z1) + 1) * CHUNK),
  ]);
});
document.addEventListener("mouseup", async () => {
  if (!drag) return;
  const { start, end, rect } = drag;
  drag = null;
  map.removeLayer(rect);
  const [x1, z1] = end || start;
  await applyLabel({ source: "area", rect: [start[0], start[1], x1, z1] });
});
map.on("click", async (e) => {
  if (e.originalEvent.shiftKey) return;
  const [cx, cz] = toChunk(e.latlng);
  if (e.originalEvent.altKey) return showDetail(cx, cz);
  await applyLabel({ source: "single", chunks: [[cx, cz]] });
  showDetail(cx, cz);
});

document.addEventListener("keydown", async (e) => {
  if (e.target.tagName === "INPUT") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    const res = await api("/api/undo", { method: "POST" });
    return afterChange(res.counts, null);
  }
  const keys = { k: "keep", d: "delete", c: "clear" };
  if (keys[e.key.toLowerCase()]) setTool(keys[e.key.toLowerCase()]);
  if (e.key.toLowerCase() === "g") { state.grid = !state.grid; redrawLabels(); }
});

for (const b of document.querySelectorAll("[data-label]"))
  b.addEventListener("click", () => setTool(b.dataset.label));
document.getElementById("opacity").addEventListener("input", (e) => {
  state.opacity = Number(e.target.value);
  redrawLabels();
});
document.getElementById("export").addEventListener("click", async () => {
  const res = await api("/api/export", { method: "POST" });
  document.getElementById("export-result").innerHTML =
    `已写入 ${res.delete} 个区块：<code>${res.path}</code><br>在 MCA Selector 中对原存档导入此选区，检查后再删除。`;
});

async function init() {
  const meta = await api("/api/meta");
  state.total = meta.chunk_count;
  const box = document.getElementById("layers");
  if (!meta.layers.length) box.innerHTML = `<div class="muted">尚未渲染底图，运行 chunktriager render</div>`;
  meta.layers.forEach((layer, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="layer" ${i === 0 ? "checked" : ""}> ${layer.title}`;
    label.querySelector("input").addEventListener("change", () => setBaseLayer(layer.name, meta));
    box.appendChild(label);
  });
  if (meta.layers.length) setBaseLayer(meta.layers[0].name, meta);
  labels.options.minZoom = Math.max(meta.min_zoom, LABEL_MIN_ZOOM);
  map.setMaxZoom(MAX_ZOOM);
  labels.addTo(map);
  map.setMinZoom(meta.min_zoom);
  if (meta.regions.length) {
    const xs = meta.regions.map((r) => r[0]), zs = meta.regions.map((r) => r[1]);
    const bounds = L.latLngBounds(
      toLatLng(Math.min(...xs) * 512, Math.min(...zs) * 512),
      toLatLng((Math.max(...xs) + 1) * 512, (Math.max(...zs) + 1) * 512),
    );
    map.fitBounds(bounds);
  } else {
    map.setView([0, 0], 0);
  }
  setTool("keep");
  renderCounts(meta.counts);
  refreshExportPreview();
}

init().catch((e) => {
  document.getElementById("detail-body").textContent = `加载失败：${e}`;
});
