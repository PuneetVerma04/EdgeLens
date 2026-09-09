/* EdgeLens web demo - NEU-DET surface defect detection, entirely in the browser.
 *
 * Model: yolo26n_int8.onnx (2.80 MB). Its head is end-to-end / NMS-free, so the graph
 * emits [1, 300, 6] = 300 detections of [x1, y1, x2, y2, confidence, class_id], already
 * sorted by confidence. There is deliberately NO non-maximum suppression in this file --
 * see README.md. Exporting yolov8n instead would give [1, 10, 8400] raw anchors and NMS
 * here would be mandatory.
 *
 * The preprocessing below mirrors ultralytics.data.augment.LetterBox exactly. It is
 * verified against the Python pipeline by web-demo/verify_parity.py; the numbers that
 * check produced are in README.md. Do not "simplify" it without re-running that check --
 * a wrong letterbox does not throw, it just moves every box a few pixels.
 */

const MODEL_URL = "model/yolo26n_int8.onnx";
const IMGSZ = 640;
const PAD_VALUE = 114; // ultralytics LetterBox padding_value
const CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface",
                     "rolled-in_scale", "scratches"];
const CLASS_COLORS = ["#e86a33", "#3ca0dc", "#78c864", "#c878d2", "#f0c83c", "#5ad2c8"];
const EXAMPLES = [
  { file: "patches_155.jpg",        label: "patches",         inDomain: true },
  { file: "scratches_141.jpg",      label: "scratches",       inDomain: true },
  { file: "pitted_surface_237.jpg", label: "pitted surface",  inDomain: true },
  { file: "cast_def_0_127.jpeg",    label: "casting (def)",   inDomain: false },
  { file: "cast_def_0_240.jpeg",    label: "casting (def)",   inDomain: false },
  { file: "cast_ok_0_119.jpeg",     label: "casting (ok)",    inDomain: false },
];

let session = null;
let confThreshold = 0.25;
let lastResult = null; // { imageData, width, height, detections, ms }

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ preprocessing */

/** Reproduce LetterBox.get_params for a fixed-size export (auto=False, scaleup=True,
 *  center=True). The -0.1/+0.1 pushes an odd padding pixel to the bottom/right. */
function letterboxParams(width, height, imgsz = IMGSZ) {
  const r = Math.min(imgsz / height, imgsz / width);
  const newW = Math.round(width * r);
  const newH = Math.round(height * r);
  const dw = (imgsz - newW) / 2;
  const dh = (imgsz - newH) / 2;
  return {
    r, newW, newH,
    left: Math.round(dw - 0.1), top: Math.round(dh - 0.1),
    right: Math.round(dw + 0.1), bottom: Math.round(dh + 0.1),
  };
}

/** Bilinear resize matching cv2.INTER_LINEAR's pixel-centre convention:
 *  src = (dst + 0.5) * scale - 0.5, clamped at the edges.
 *
 *  Canvas drawImage is NOT used for the resize. Its scaling quality is
 *  implementation-defined -- browsers may pick bicubic or Lanczos -- so it would put a
 *  browser-dependent difference between this and the Python pipeline before inference
 *  even starts. Doing the interpolation by hand costs ~20 lines and removes that variable. */
function resizeBilinear(src, srcW, srcH, dstW, dstH) {
  const out = new Uint8ClampedArray(dstW * dstH * 4);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    let fy = (y + 0.5) * scaleY - 0.5;
    if (fy < 0) fy = 0;
    const y0 = Math.min(Math.floor(fy), srcH - 1);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = fy - y0;
    for (let x = 0; x < dstW; x++) {
      let fx = (x + 0.5) * scaleX - 0.5;
      if (fx < 0) fx = 0;
      const x0 = Math.min(Math.floor(fx), srcW - 1);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = fx - x0;
      const i00 = (y0 * srcW + x0) * 4, i01 = (y0 * srcW + x1) * 4;
      const i10 = (y1 * srcW + x0) * 4, i11 = (y1 * srcW + x1) * 4;
      const o = (y * dstW + x) * 4;
      for (let c = 0; c < 3; c++) {
        const top = src[i00 + c] * (1 - wx) + src[i01 + c] * wx;
        const bot = src[i10 + c] * (1 - wx) + src[i11 + c] * wx;
        out[o + c] = top * (1 - wy) + bot * wy;
      }
      out[o + 3] = 255;
    }
  }
  return out;
}

/** ImageData -> Float32Array (1,3,640,640), RGB in [0,1], letterboxed with 114 padding.
 *  Canvas gives RGBA directly, so unlike the Python path (which reads BGR via cv2 and
 *  swaps) there is no channel swap to do here -- the tensor is RGB either way. */
function preprocess(imageData) {
  const { width, height, data } = imageData;
  const p = letterboxParams(width, height);
  const resized = resizeBilinear(data, width, height, p.newW, p.newH);

  const tensor = new Float32Array(3 * IMGSZ * IMGSZ).fill(PAD_VALUE / 255);
  const plane = IMGSZ * IMGSZ;
  for (let y = 0; y < p.newH; y++) {
    const dy = y + p.top;
    if (dy < 0 || dy >= IMGSZ) continue;
    for (let x = 0; x < p.newW; x++) {
      const dx = x + p.left;
      if (dx < 0 || dx >= IMGSZ) continue;
      const s = (y * p.newW + x) * 4;
      const d = dy * IMGSZ + dx;
      tensor[d] = resized[s] / 255;                    // R
      tensor[plane + d] = resized[s + 1] / 255;        // G
      tensor[2 * plane + d] = resized[s + 2] / 255;    // B
    }
  }
  return { tensor, params: p };
}

/** Map boxes from 640x640 letterbox space back to original image pixels. */
function undoLetterbox(box, p, width, height) {
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  return [
    clamp((box[0] - p.left) / p.r, 0, width),
    clamp((box[1] - p.top) / p.r, 0, height),
    clamp((box[2] - p.left) / p.r, 0, width),
    clamp((box[3] - p.top) / p.r, 0, height),
  ];
}

/* ---------------------------------------------------------------------- inference */

async function loadModel() {
  const bar = $("progressBar");
  const status = $("loadStatus");
  const wrap = $("loadWrap");
  wrap.hidden = false;

  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`model fetch failed: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 2938750;

  // Stream so the progress bar reflects real bytes rather than guessing. On a cold load
  // this file is ~2.8 MB and the wait is otherwise unexplained.
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = Math.min(100, (received / total) * 100);
    bar.style.width = pct + "%";
    status.textContent = `Downloading model ${(received / 1048576).toFixed(2)} / ${(total / 1048576).toFixed(2)} MB`;
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }

  status.textContent = "Initialising ONNX Runtime…";
  bar.style.width = "100%";
  session = await ort.InferenceSession.create(bytes.buffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  wrap.hidden = true;
  $("modelInfo").textContent =
    `yolo26n INT8 · ${(received / 1048576).toFixed(2)} MB · ` +
    `${session.outputNames[0]} ${JSON.stringify(session.outputMetadata?.[0]?.dims ?? [1, 300, 6])} · NMS-free`;
  setBusy(false);
}

async function detect(imageData) {
  const { tensor, params } = preprocess(imageData);
  const input = new ort.Tensor("float32", tensor, [1, 3, IMGSZ, IMGSZ]);

  const t0 = performance.now();
  const output = await session.run({ [session.inputNames[0]]: input });
  const ms = performance.now() - t0;

  // [1, 300, 6] -> [x1, y1, x2, y2, conf, cls], already sorted, already NMS-free.
  const raw = output[session.outputNames[0]].data;
  const detections = [];
  for (let i = 0; i < 300; i++) {
    const o = i * 6;
    const conf = raw[o + 4];
    if (conf < confThreshold) continue;   // sorted desc, but filter defensively
    detections.push({
      class_id: Math.round(raw[o + 5]),
      confidence: conf,
      box: undoLetterbox([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]],
                         params, imageData.width, imageData.height),
    });
  }
  return { detections, ms, params };
}

/* ------------------------------------------------------------------------ drawing */

function render() {
  if (!lastResult) return;
  const { imageData, width, height, detections, ms } = lastResult;

  const canvas = $("canvas");
  const maxW = Math.min(canvas.parentElement.clientWidth, 760);
  const scale = Math.max(1, Math.min(4, maxW / width));
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;   // NEU-DET is 200px; keep the upscale crisp
  const off = document.createElement("canvas");
  off.width = width; off.height = height;
  off.getContext("2d").putImageData(imageData, 0, 0);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  const visible = detections.filter((d) => d.confidence >= confThreshold);

  // Label size is deliberately independent of `scale`. NEU-DET images are 200 px, so the
  // canvas is upscaled ~3.5x; scaling the font with it produced 39 px text that buried the
  // image it was annotating. Boxes scale, chrome does not.
  const LINE = 2;
  const FONT = 13;
  const LABEL_H = 18;
  ctx.lineWidth = LINE;
  ctx.font = `${FONT}px ui-monospace, monospace`;
  ctx.textBaseline = "top";

  // Draw thin boxes first so labels are never hidden behind a neighbouring box.
  for (const d of visible) {
    const [x1, y1, x2, y2] = d.box.map((v) => v * scale);
    ctx.strokeStyle = CLASS_COLORS[d.class_id % CLASS_COLORS.length];
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  for (const d of visible) {
    const colour = CLASS_COLORS[d.class_id % CLASS_COLORS.length];
    const [x1, y1] = [d.box[0] * scale, d.box[1] * scale];
    const label = `${CLASS_NAMES[d.class_id]} ${d.confidence.toFixed(2)}`;
    const w = ctx.measureText(label).width + 8;

    // Keep the label inside the canvas on all four edges.
    const lx = Math.min(Math.max(0, x1 - LINE / 2), canvas.width - w);
    const ly = y1 - LABEL_H < 0 ? y1 : y1 - LABEL_H;

    ctx.fillStyle = colour;
    ctx.fillRect(lx, ly, w, LABEL_H);
    ctx.fillStyle = "#0b0d10";
    ctx.fillText(label, lx + 4, ly + 3);
  }

  $("stats").hidden = false;
  $("statCount").textContent = visible.length;
  $("statMs").textContent = ms.toFixed(1);
  $("statSize").textContent = `${width}×${height}`;

  const list = $("detections");
  list.innerHTML = "";
  if (!visible.length) {
    list.innerHTML = '<li class="empty">No defects above the confidence threshold.</li>';
  }
  for (const d of visible) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="swatch" style="background:${CLASS_COLORS[d.class_id % 6]}"></span>` +
      `<span class="cls">${CLASS_NAMES[d.class_id]}</span>` +
      `<span class="conf">${(d.confidence * 100).toFixed(1)}%</span>` +
      `<span class="box">${d.box.map((v) => v.toFixed(0)).join(", ")}</span>`;
    list.appendChild(li);
  }
}

/* -------------------------------------------------------------------------- input */

function setBusy(busy, message) {
  $("busy").hidden = !busy;
  if (message) $("busyText").textContent = message;
  document.querySelectorAll("button, input").forEach((el) => { el.disabled = busy; });
}

async function imageDataFrom(source) {
  const bitmap = await createImageBitmap(source);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

async function runOn(source, name) {
  if (!session) return;
  setBusy(true, "Running inference…");
  try {
    const imageData = await imageDataFrom(source);
    const { detections, ms } = await detect(imageData);
    lastResult = { imageData, width: imageData.width, height: imageData.height,
                   detections, ms, name };
    $("placeholder").hidden = true;
    $("canvas").hidden = false;
    render();
  } catch (err) {
    alert(`Inference failed: ${err.message}`);
    console.error(err);
  } finally {
    setBusy(false);
  }
}

async function runOnUrl(url, name) {
  const blob = await (await fetch(url)).blob();
  await runOn(blob, name);
}

/* --------------------------------------------------------------- parity harness
 * Exposed for web-demo/verify_parity.py --check. Runs every bundled example through
 * the exact user-facing path and returns detections in original-image pixels, which is
 * the same space reference.json uses. */
window.__runParity = async function () {
  const images = {};
  for (const ex of EXAMPLES) {
    const blob = await (await fetch(`examples/${ex.file}`)).blob();
    const imageData = await imageDataFrom(blob);
    const { detections } = await detect(imageData);
    images[ex.file] = {
      width: imageData.width,
      height: imageData.height,
      detections: detections.map((d) => ({
        class_id: d.class_id,
        confidence: Number(d.confidence.toFixed(6)),
        box: d.box.map((v) => Number(v.toFixed(3))),
      })),
    };
  }
  return { source: "browser", userAgent: navigator.userAgent, images };
};

/* --------------------------------------------------------------------------- init */

function buildExamples() {
  const wrap = $("examples");
  for (const ex of EXAMPLES) {
    const btn = document.createElement("button");
    btn.className = "example" + (ex.inDomain ? "" : " ood");
    btn.innerHTML = `<img src="examples/${ex.file}" alt="${ex.label}" loading="lazy">` +
                    `<span>${ex.label}</span>`;
    btn.title = ex.inDomain
      ? `NEU-DET test split · ${ex.file}`
      : `Out of distribution: a casting part, not steel surface. ${ex.file}`;
    btn.addEventListener("click", () => runOnUrl(`examples/${ex.file}`, ex.file));
    wrap.appendChild(btn);
  }
}

function wireInput() {
  const drop = $("drop");
  $("filePicker").addEventListener("change", (e) => {
    if (e.target.files?.[0]) runOn(e.target.files[0], e.target.files[0].name);
  });
  drop.addEventListener("click", () => $("filePicker").click());
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) runOn(file, file.name);
  });

  const slider = $("confSlider");
  slider.addEventListener("input", () => {
    confThreshold = Number(slider.value);
    $("confValue").textContent = confThreshold.toFixed(2);
    render();   // re-filter without re-running the model
  });
}

(async function init() {
  buildExamples();
  wireInput();
  setBusy(true, "Loading model…");
  try {
    // Match the CDN the <script> tag pulled ort from, so the .wasm binaries agree with it.
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;   // no cross-origin isolation on GitHub Pages
    await loadModel();
  } catch (err) {
    $("loadStatus").textContent = `Failed to load model: ${err.message}`;
    $("progressBar").style.background = "#e05252";
    console.error(err);
  }
})();
