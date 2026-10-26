// Map coordinates: CRS.Simple at zoom 0 is 1 px per block, and a block at
// (x, z) sits at latLng(-z, x). Zoom-0 tile (x, y) is region (x, z).
"use strict";

const REGION = 32;          // chunks per region side
const CHUNK = 16;           // blocks per chunk side
const RGB = { k: [46, 139, 87], d: [192, 57, 43] };
const CHAR = { keep: "k", delete: "d", clear: "u" };
// Zoom 5 is one chunk per 512 px tile; below LABEL_MIN_ZOOM a tile would
// cover too many chunks for a per-chunk bitmap.
const MAX_ZOOM = 5;
const LABEL_MIN_ZOOM = -5;
const START_ZOOM = -1;
// As in MCA Selector, zoomed-in views work in chunks (chunk grid, chunk
// marking) and zoomed-out views in regions. Compared on the rounded zoom,
// which is also the zoom Leaflet renders tiles at.
const CHUNK_MODE_MIN_ZOOM = -1;
const MOUSE_KEY = "chunktriager.mouse";
const VIEW_KEY = "chunktriager.view";
let viewKey = VIEW_KEY;

const state = {
  grid: true,
  onlyLabeled: false,
  // "labels": your labels. "prob": model probability. "protect": chunks
  // protection keeps, coloured by why.
  view: "labels",
  // "rx,rz" -> { prob: Uint8Array(1024), protected: string } from the last predict.
  preds: new Map(),
  counts: { keep: 0, delete: 0 },
  opacity: 0.45,
  // "rx,rz" -> array of 1024 state chars ("." absent, "u", "k", "d"), or
  // null for a region without chunks.
  regions: new Map(),
  // Scheme "a": left keep, right aRight. Scheme "b": left bLeft, right clear.
  mouse: { scheme: "a", aRight: "delete", bLeft: "keep",
           ...JSON.parse(localStorage.getItem(MOUSE_KEY) || "{}") },
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
const chunkMode = () => Math.round(map.getZoom()) >= CHUNK_MODE_MIN_ZOOM;

function actionFor(button) {
  const m = state.mouse;
  if (m.scheme === "a") return button === 0 ? "keep" : m.aRight;
  return button === 0 ? m.bLeft : "clear";
}

// Region state cache ---------------------------------------------------------

// Bumped on every local edit, so a fetch that started before the edit
// cannot overwrite it with stale data.
const seq = new Map();
const bump = (key) => seq.set(key, (seq.get(key) || 0) + 1);

function chunkState(cx, cz) {
  const s = state.regions.get(`${regionOf(cx)},${regionOf(cz)}`);
  return s ? s[mod(cz, REGION) * REGION + mod(cx, REGION)] : undefined;
}

// Load region states for an inclusive region range, in blocks the server accepts.
async function loadRegions(rx0, rz0, rx1, rz1, force = false) {
  const STEP = 64;
  const jobs = [];
  for (let bx = rx0; bx <= rx1; bx += STEP)
    for (let bz = rz0; bz <= rz1; bz += STEP)
      jobs.push(loadBlock(bx, bz, Math.min(bx + STEP - 1, rx1), Math.min(bz + STEP - 1, rz1), force));
  await Promise.all(jobs);
}

async function loadBlock(rx0, rz0, rx1, rz1, force) {
  const wanted = new Map();
  for (let rx = rx0; rx <= rx1; rx++)
    for (let rz = rz0; rz <= rz1; rz++) {
      const key = `${rx},${rz}`;
      if (force || !state.regions.has(key)) wanted.set(key, seq.get(key) || 0);
    }
  if (!wanted.size) return;
  const data = await api(`/api/state?rx0=${rx0}&rz0=${rz0}&rx1=${rx1}&rz1=${rz1}`);
  for (const [key, s] of wanted)
    if ((seq.get(key) || 0) === s) state.regions.set(key, data[key] ? data[key].split("") : null);
}

async function loadPredictions(rx0, rz0, rx1, rz1) {
  const missing = [];
  for (let rx = rx0; rx <= rx1; rx++)
    for (let rz = rz0; rz <= rz1; rz++)
      if (!state.preds.has(`${rx},${rz}`)) missing.push(`${rx},${rz}`);
  if (!missing.length) return;
  const data = await api(`/api/predictions?rx0=${rx0}&rz0=${rz0}&rx1=${rx1}&rz1=${rz1}`);
  for (const key of missing) {
    const d = data[key];
    if (!d) { state.preds.set(key, null); continue; }
    const prob = new Uint8Array(1024);
    for (let i = 0; i < 1024; i++) prob[i] = parseInt(d.prob.substr(i * 2, 2), 16);
    state.preds.set(key, { prob, protected: d.protected });
  }
}

function predAt(cx, cz) {
  const p = state.preds.get(`${regionOf(cx)},${regionOf(cz)}`);
  if (!p) return null;
  const i = mod(cz, REGION) * REGION + mod(cx, REGION);
  return { prob: p.prob[i] / 255, protected: p.protected[i] === "p" };
}

// Colour of one chunk in the current view, or null for transparent.
function chunkColour(cx, cz) {
  const s = chunkState(cx, cz);
  if (state.view === "labels") return RGB[s] || null;
  const p = predAt(cx, cz);
  if (!p) return null;
  if (state.view === "prob") {
    if (p.prob < state.threshold) return null;
    // Threshold .. 1 mapped from yellow to red on a log scale.
    const t = Math.min(1, Math.log(p.prob / state.threshold) / Math.log(1 / state.threshold));
    return [255, Math.round(220 * (1 - t)), 0];
  }
  // protect: player-made labels the model missed are the important case.
  if (s === "k" && !p.protected) return [255, 0, 255];
  if (!p.protected) return null;
  return p.prob >= state.threshold ? [255, 140, 0] : [70, 130, 230];
}

// Apply a label locally; returns true if any chunk changed.
function setLocal(cx, cz, action) {
  const key = `${regionOf(cx)},${regionOf(cz)}`;
  const s = state.regions.get(key);
  if (!s) return false;
  const i = mod(cz, REGION) * REGION + mod(cx, REGION);
  if (s[i] === "." || s[i] === CHAR[action]) return false;
  s[i] = CHAR[action];
  bump(key);
  return true;
}

// Label layer ------------------------------------------------------------------

// Chunk and region range covered by a tile; tile zooms are integers.
function tileRange(coords) {
  const span = REGION / Math.pow(2, coords.z);        // chunks per tile side
  const cx0 = coords.x * span, cz0 = coords.y * span;
  return {
    z: coords.z, span, cx0, cz0,
    rx0: regionOf(cx0), rz0: regionOf(cz0),
    rx1: regionOf(cx0 + span - 1), rz1: regionOf(cz0 + span - 1),
  };
}

function paintTile(tile, r) {
  const ctx = tile.getContext("2d");
  ctx.clearRect(0, 0, tile.width, tile.height);
  // One pixel per chunk, then scaled up without smoothing. With "only
  // labeled" on, unlabeled chunks are masked so the base map shows only
  // under labeled ones.
  const bitmap = new ImageData(r.span, r.span);
  const alpha = Math.round(state.opacity * 255);
  for (let dz = 0; dz < r.span; dz++)
    for (let dx = 0; dx < r.span; dx++) {
      const s = chunkState(r.cx0 + dx, r.cz0 + dz);
      const c = chunkColour(r.cx0 + dx, r.cz0 + dz);
      const i = (dz * r.span + dx) * 4;
      if (c && (!state.onlyLabeled || RGB[s])) {
        bitmap.data[i] = c[0]; bitmap.data[i + 1] = c[1]; bitmap.data[i + 2] = c[2];
        bitmap.data[i + 3] = alpha;
      } else if (state.onlyLabeled && !RGB[s]) {
        bitmap.data[i + 3] = 255;     // opaque black
      }
    }
  const small = document.createElement("canvas");
  small.width = small.height = r.span;
  small.getContext("2d").putImageData(bitmap, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, tile.width, tile.height);
  if (!state.grid) return;
  // Chunk lines in chunk mode, region lines otherwise; never both.
  const chunks = r.z >= CHUNK_MODE_MIN_ZOOM;
  const step = (chunks ? 1 : REGION) * tile.width / r.span;
  ctx.lineWidth = 1;
  ctx.strokeStyle = chunks ? "rgba(255,255,255,0.18)" : "rgba(255,255,255,0.35)";
  ctx.beginPath();
  for (let off = 0; off <= tile.width; off += step) {
    ctx.moveTo(off + 0.5, 0); ctx.lineTo(off + 0.5, tile.height);
    ctx.moveTo(0, off + 0.5); ctx.lineTo(tile.width, off + 0.5);
  }
  ctx.stroke();
}

const LabelLayer = L.GridLayer.extend({
  createTile(coords, done) {
    const tile = document.createElement("canvas");
    tile.width = tile.height = this.getTileSize().x;
    const r = tileRange(coords);
    Promise.all([loadRegions(r.rx0, r.rz0, r.rx1, r.rz1),
                 state.hasPreds ? loadPredictions(r.rx0, r.rz0, r.rx1, r.rz1) : null])
      .then(() => { paintTile(tile, r); tile._range = r; done(null, tile); })
      .catch((e) => done(e, tile));
    return tile;
  },

  // Repaint loaded tiles in place. GridLayer.redraw() is not used: at a
  // fractional zoom it rebuilds tiles at that fractional zoom, which leaves
  // the layer blank until the next zoom change.
  repaint(dirty = null) {
    for (const key in this._tiles) {
      const tile = this._tiles[key].el;
      const r = tile._range;
      if (!r) continue;
      if (dirty && !touches(r, dirty)) continue;
      paintTile(tile, r);
    }
  },

  // Refetch state for every loaded tile, then repaint (after undo).
  async reload() {
    state.regions.clear();
    const jobs = [];
    for (const key in this._tiles) {
      const r = this._tiles[key].el._range;
      if (r) jobs.push(loadRegions(r.rx0, r.rz0, r.rx1, r.rz1));
    }
    await Promise.all(jobs);
    this.repaint();
  },
});

function touches(r, dirty) {
  for (const key of dirty) {
    const [rx, rz] = key.split(",").map(Number);
    if (rx >= r.rx0 && rx <= r.rx1 && rz >= r.rz0 && rz <= r.rz1) return true;
  }
  return false;
}

// Map ------------------------------------------------------------------------

const map = L.map("map", {
  crs: L.CRS.Simple,
  // Finer zoom steps: each wheel notch or +/- press changes zoom by 0.5.
  zoomSnap: 0.25,
  zoomDelta: 0.5,
  wheelPxPerZoomLevel: 120,
  dragging: false,           // panning is done with the middle button below
  boxZoom: false,
  doubleClickZoom: false,
  attributionControl: false,
});
const labels = new LabelLayer({ tileSize: 512, maxZoom: MAX_ZOOM, zIndex: 10 });
let baseLayer = null;
map.on("zoomend", () => labels.repaint());

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

function renderCounts(counts) {
  state.counts = counts;
  const unlabeled = state.total - counts.keep - counts.delete;
  const row = (color, name, n) =>
    `<span><i class="swatch" style="background:${color}"></i>${name}</span><span>${n}</span>`;
  document.getElementById("counts").innerHTML =
    row("#2e8b57", "人工", counts.keep) + row("#c0392b", "自然", counts.delete) +
    row("#444", "未标注", unlabeled);
}

function showError(e) {
  document.getElementById("detail-body").innerHTML = `<span style="color:#e88">出错：${e}</span>`;
}

// Marking ----------------------------------------------------------------------
//
// Dragging paints the cells the cursor passes over. With Shift held, every
// move marks the rectangle from the anchor to the cursor instead (cells stay
// marked if the rectangle shrinks again); the anchor is the press point, or
// the last painted cell if Shift is pressed mid-drag.
// Cells are chunks when zoomed in and whole regions when zoomed out.
// Ctrl forces region cells at any zoom.
//
// Changes are drawn immediately and sent to the server when the button is
// released, as one undoable batch.

let stroke = null;   // { button, action, regionCells, anchor, chunks: Set, dirty: Set }
let flushing = Promise.resolve();

function cellAt(ll, regionCells) {
  const [cx, cz] = toChunk(ll);
  return regionCells ? [regionOf(cx), regionOf(cz)] : [cx, cz];
}

function markRect(a, b) {
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]);
  const z0 = Math.min(a[1], b[1]), z1 = Math.max(a[1], b[1]);
  const scale = stroke.regionCells ? REGION : 1;
  for (let x = x0 * scale; x < (x1 + 1) * scale; x++)
    for (let z = z0 * scale; z < (z1 + 1) * scale; z++) {
      if (!setLocal(x, z, stroke.action)) continue;
      stroke.chunks.add(`${x},${z}`);
      stroke.dirty.add(`${regionOf(x)},${regionOf(z)}`);
    }
}

function strokeMove(ev) {
  const cell = cellAt(map.mouseEventToLatLng(ev), stroke.regionCells);
  const before = stroke.chunks.size;
  if (ev.shiftKey) {
    markRect(stroke.anchor, cell);
  } else {
    // Walk from the previous cell so fast moves leave no gaps.
    const [px, pz] = stroke.prev || cell;
    const steps = Math.max(Math.abs(cell[0] - px), Math.abs(cell[1] - pz), 1);
    for (let i = 0; i <= steps; i++) {
      const p = [Math.round(px + (cell[0] - px) * i / steps), Math.round(pz + (cell[1] - pz) * i / steps)];
      markRect(p, p);
    }
    stroke.anchor = cell;
  }
  stroke.prev = cell;
  if (stroke.chunks.size !== before) labels.repaint(stroke.dirty);
}

async function strokeEnd() {
  const { action, chunks } = stroke;
  stroke = null;
  if (!chunks.size) return;
  const list = [...chunks].map((k) => k.split(",").map(Number));
  const label = action === "clear" ? null : action;
  const send = () => api("/api/labels", {
    method: "POST",
    body: { label, source: list.length === 1 ? "single" : "area", chunks: list },
  });
  // Keep batches in order, so a later stroke cannot land before an earlier one.
  flushing = flushing.then(send).then((res) => {
    renderCounts(res.counts);
  }).catch(async (e) => {
    showError(e);
    await labels.reload();          // drop optimistic changes the server rejected
  });
  await flushing;
  if (list.length === 1) showDetail(...list[0]);
}

async function showDetail(cx, cz) {
  const el = document.getElementById("detail-body");
  const s = chunkState(cx, cz);
  if (s === "." || s === undefined) {
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

// Mouse: left/right mark, middle drag pans, Alt+click only inspects.
const container = map.getContainer();
let pan = null;      // { x, y }

container.addEventListener("contextmenu", (ev) => ev.preventDefault());
container.addEventListener("mousedown", (ev) => {
  if (ev.target.closest(".leaflet-control")) return;
  if (ev.button === 1) {
    ev.preventDefault();   // also stops Windows middle-click autoscroll
    pan = { x: ev.clientX, y: ev.clientY };
    container.classList.add("panning");
    return;
  }
  if ((ev.button !== 0 && ev.button !== 2) || stroke) return;
  ev.preventDefault();
  const ll = map.mouseEventToLatLng(ev);
  if (ev.altKey) { showDetail(...toChunk(ll)); return; }
  const regionCells = ev.ctrlKey || ev.metaKey || !chunkMode();
  stroke = {
    button: ev.button, action: actionFor(ev.button), regionCells,
    anchor: cellAt(ll, regionCells), chunks: new Set(), dirty: new Set(),
  };
  strokeMove(ev);
});

document.addEventListener("mousemove", (ev) => {
  if (pan) {
    map.panBy([pan.x - ev.clientX, pan.y - ev.clientY], { animate: false });
    pan = { x: ev.clientX, y: ev.clientY };
    return;
  }
  if (stroke) strokeMove(ev);
  if (!container.contains(ev.target)) return;
  const ll = map.mouseEventToLatLng(ev);
  const [bx, bz] = toBlock(ll);
  const [cx, cz] = toChunk(ll);
  document.getElementById("cursor").textContent =
    `方块 ${bx}, ${bz} · 区块 ${cx}, ${cz} · r.${regionOf(cx)}.${regionOf(cz)}`;
});

document.addEventListener("mouseup", (ev) => {
  if (ev.button === 1 && pan) {
    pan = null;
    container.classList.remove("panning");
    return;
  }
  if (stroke && ev.button === stroke.button) strokeEnd();
});

document.addEventListener("keydown", async (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    await flushing;
    try {
      const res = await api("/api/undo", { method: "POST" });
      renderCounts(res.counts);
      } catch (err) { showError(err); }
    document.getElementById("detail-body").innerHTML = `<div class="muted">已撤销</div>`;
    return labels.reload();
  }
  if (e.key.toLowerCase() === "g") {
    state.grid = !state.grid;
    document.getElementById("grid").checked = state.grid;
    labels.repaint();
  }
});

// Mouse settings: one segmented control per button, built from state.mouse
// so the panel and the map badge always show what each button does.
// Stored labels stay "keep"/"delete"; they mean "contains player-made
// content" and "natural".
const ACTION_NAME = { keep: "人工", delete: "自然", clear: "清除" };

function renderMouse() {
  const m = state.mouse;
  for (const b of document.querySelectorAll("#scheme button"))
    b.classList.toggle("on", b.dataset.v === m.scheme);
  // Scheme 1: left is fixed to keep, right chooses. Scheme 2: the reverse.
  const left = m.scheme === "a" ? ["keep"] : ["keep", "delete"];
  const right = m.scheme === "a" ? ["delete", "clear"] : ["clear"];
  const fill = (id, options, current, onPick) => {
    const box = document.getElementById(id);
    box.innerHTML = "";
    for (const a of options) {
      const b = document.createElement("button");
      b.textContent = ACTION_NAME[a];
      b.className = a === current ? `on ${a}` : "";
      b.disabled = options.length === 1;
      b.addEventListener("click", () => { onPick(a); saveMouse(); });
      box.appendChild(b);
    }
  };
  fill("bind-left", left, actionFor(0), (a) => { m.bLeft = a; });
  fill("bind-right", right, actionFor(2), (a) => { m.aRight = a; });
  document.getElementById("mouse-badge").innerHTML =
    `<span class="${actionFor(0)}">左键 ${ACTION_NAME[actionFor(0)]}</span>` +
    `<span class="${actionFor(2)}">右键 ${ACTION_NAME[actionFor(2)]}</span>`;
}

function saveMouse() {
  localStorage.setItem(MOUSE_KEY, JSON.stringify(state.mouse));
  renderMouse();
}

for (const b of document.querySelectorAll("#scheme button"))
  b.addEventListener("click", () => { state.mouse.scheme = b.dataset.v; saveMouse(); });
renderMouse();

document.getElementById("opacity").addEventListener("input", (e) => {
  state.opacity = Number(e.target.value);
  labels.repaint();
});
document.getElementById("grid").addEventListener("change", (e) => {
  state.grid = e.target.checked;
  labels.repaint();
});
document.getElementById("only-labeled").addEventListener("change", (e) => {
  state.onlyLabeled = e.target.checked;
  labels.repaint();
});
document.getElementById("clear-all").addEventListener("click", async () => {
  const n = state.counts.keep + state.counts.delete;
  if (!n) return;
  if (!confirm(`清除全部 ${n} 个标注？可以用 Ctrl+Z 撤销。`)) return;
  await flushing;
  try {
    const res = await api("/api/labels/clear", { method: "POST" });
    renderCounts(res.counts);
  } catch (err) { showError(err); }
  await labels.reload();
});
map.on("moveend", () => {
  const c = map.getCenter();
  localStorage.setItem(viewKey, JSON.stringify({ lat: c.lat, lng: c.lng, zoom: map.getZoom() }));
});

// Model views -----------------------------------------------------------------

const LEGEND = {
  labels: [["#2e8b57", "人工"], ["#c0392b", "自然"]],
  prob: [["ramp", `≥ 阈值（黄）到 1（红）`]],
  protect: [["#ff8c00", "判为人工"], ["#4682e6", "人工区块附近，被保护"],
            ["#ff00ff", "标为人工但未被保护"]],
};

function renderView() {
  for (const b of document.querySelectorAll("#view button"))
    b.classList.toggle("on", b.dataset.v === state.view);
  document.getElementById("legend").innerHTML = LEGEND[state.view].map(([c, text]) =>
    (c === "ramp" ? `<i class="ramp"></i>` : `<i class="swatch" style="background:${c}"></i>`) +
    `<span>${text}</span>`).join("");
  labels.repaint();
}

async function renderReview(kind) {
  for (const b of document.querySelectorAll("#review-kind button"))
    b.classList.toggle("on", b.dataset.v === kind);
  const list = document.getElementById("review");
  list.innerHTML = "<li class='muted'>加载中</li>";
  const rows = await api(`/api/review?kind=${kind}`);
  list.innerHTML = rows.length ? "" : "<li class='muted'>无</li>";
  for (const r of rows) {
    const li = document.createElement("li");
    li.innerHTML = `(${r.x}, ${r.z})<span>${r.prob < 0.01 ? r.prob.toExponential(0) : r.prob.toFixed(2)}</span>`;
    li.addEventListener("click", () => {
      map.setView(toLatLng(r.x * CHUNK + 8, r.z * CHUNK + 8), Math.max(map.getZoom(), 2));
      showDetail(r.x, r.z);
    });
    list.appendChild(li);
  }
}

for (const b of document.querySelectorAll("#view button"))
  b.addEventListener("click", () => { state.view = b.dataset.v; renderView(); });
for (const b of document.querySelectorAll("#review-kind button"))
  b.addEventListener("click", () => renderReview(b.dataset.v));

function setupModel(summary) {
  state.hasPreds = !!summary;
  document.getElementById("model-none").hidden = state.hasPreds;
  document.getElementById("model-body").hidden = !state.hasPreds;
  if (!summary) return;
  state.threshold = summary.threshold;
  document.getElementById("model-info").textContent =
    `模型训练于 ${summary.trained_at.replace("T", " ")}。阈值 ${summary.threshold}：` +
    `${summary.flagged} 个区块判为人工，${summary.protected} / ${summary.chunks} ` +
    `（${(summary.protected / summary.chunks * 100).toFixed(1)}%）被保护。`;
  renderView();
  renderReview("missed");
}

async function init() {
  const meta = await api("/api/meta");
  state.total = meta.chunk_count;
  // Several worlds can be labeled on the same address; keep views apart.
  viewKey = `${VIEW_KEY}.${meta.world}`;
  document.title = `${meta.world} - Chunk Triager`;
  const box = document.getElementById("layers");
  if (!meta.layers.length) box.innerHTML = `<div class="muted">尚未渲染底图，运行 chunktriager render ${meta.world}</div>`;
  meta.layers.forEach((layer, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="layer" ${i === 0 ? "checked" : ""}> ${layer.title}`;
    label.querySelector("input").addEventListener("change", () => setBaseLayer(layer.name, meta));
    box.appendChild(label);
  });
  if (meta.layers.length) setBaseLayer(meta.layers[0].name, meta);
  labels.options.minZoom = Math.max(meta.min_zoom, LABEL_MIN_ZOOM);
  map.setMinZoom(meta.min_zoom);
  map.setMaxZoom(MAX_ZOOM);

  // Restore the last view; otherwise start at a chunk-readable zoom on the
  // center of the explored area.
  const saved = JSON.parse(localStorage.getItem(viewKey) || "null");
  if (saved) {
    map.setView([saved.lat, saved.lng], saved.zoom);
  } else if (meta.regions.length) {
    const xs = meta.regions.map((r) => r[0]), zs = meta.regions.map((r) => r[1]);
    const cx = (Math.min(...xs) + Math.max(...xs) + 1) / 2 * 512;
    const cz = (Math.min(...zs) + Math.max(...zs) + 1) / 2 * 512;
    map.setView(toLatLng(cx, cz), START_ZOOM);
  } else {
    map.setView([0, 0], START_ZOOM);
  }
  setupModel(meta.predictions);
  labels.addTo(map);
  renderCounts(meta.counts);
}

init().catch(showError);
