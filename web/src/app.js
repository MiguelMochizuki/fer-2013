/**
 * Page wiring: file picking, the worker, and rendering the result.
 * All visible text comes from strings.js; the only network traffic after the
 * models are loaded is none (see the CSP in index.html).
 */

import { S } from "./strings.js";
import { rankProbabilities, validateFile } from "./view.js";

const $ = (id) => document.getElementById(id);
const app = $("app");
const els = {
  drop: $("drop"), choose: $("choose"), example: $("example"), file: $("file"),
  photo: $("photo"), canvas: $("canvas"), boxes: $("boxes"),
  progress: $("progress"), progressFill: $("progressFill"), status: $("status"),
  error: $("error"), errorText: $("errorText"), retry: $("retry"),
  strip: $("strip"), faces: $("faces"), timings: $("timings"), plate: $("plate"),
};

// ---- static text ----
for (const node of document.querySelectorAll("[data-s]")) node.textContent = S[node.dataset.s];
for (const node of document.querySelectorAll("[data-s-label]")) node.setAttribute("aria-label", S[node.dataset.sLabel]);
document.title = S.title;

function h(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

// ---- feature check: the runtime needs WebAssembly SIMD ----
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
const supported = typeof WebAssembly === "object" && WebAssembly.validate(SIMD_PROBE);

// ---- worker ----
let worker = null;
let ready = false;
let initInFlight = false;
let readyDeferred = null;
let resultDeferred = null;
let lastProgress = null;

const deferred = () => {
  const d = {};
  d.promise = new Promise((resolve, reject) => Object.assign(d, { resolve, reject }));
  d.promise.catch(() => {}); // avoid unhandled rejections when nobody is waiting
  return d;
};

function onMessage({ data }) {
  if (data.type === "progress") {
    lastProgress = data;
    renderProgress();
  } else if (data.type === "ready") {
    ready = true;
    initInFlight = false;
    readyDeferred.resolve();
  } else if (data.type === "result") {
    resultDeferred?.resolve(data.result);
  } else if (data.type === "error") {
    const error = { code: data.code };
    if (!ready) {
      // ort keeps a module-level "failed" flag, so a retry needs a fresh worker.
      worker?.terminate();
      worker = null;
      initInFlight = false;
      readyDeferred?.reject(error);
    }
    resultDeferred?.reject(error);
  }
}

function startWorker() {
  if (worker) return;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = onMessage;
  worker.onerror = () =>
    onMessage({ data: { type: "error", code: ready ? "inference" : "download" } });
}

/** Start (or join) the model download; safe to call repeatedly. */
function ensureReady() {
  startWorker();
  if (!ready && !initInFlight) {
    initInFlight = true;
    readyDeferred = deferred();
    worker.postMessage({ type: "init" });
  }
  return readyDeferred.promise;
}

function prefetch() {
  if (supported && !ready) ensureReady().catch(() => {});
}

// ---- state and small UI helpers ----
let busy = false;
let lastBlob = null;

function setState(state) {
  app.dataset.state = state;
}
function setBusy(value) {
  busy = value;
  els.choose.disabled = els.example.disabled = els.plate.disabled = value || !supported;
}
function setStatus(text) {
  els.status.textContent = text;
}
function renderProgress() {
  if (!lastProgress || ready || els.progress.hidden) return;
  const { loaded, total } = lastProgress;
  els.progressFill.style.transform = `scaleX(${total ? loaded / total : 0})`;
  setStatus(S.loadingModel(loaded, total));
}
function showError(code) {
  els.errorText.textContent = S.errors[code] ?? S.errors.inference;
  const retryable = code === "integrity" || code === "download";
  els.retry.hidden = !retryable;
  els.error.hidden = false;
  setStatus("");
}
function clearResults() {
  els.strip.replaceChildren();
  els.strip.hidden = true;
  els.faces.replaceChildren();
  els.boxes.replaceChildren();
  els.boxes.classList.remove("in", "multi");
  els.timings.hidden = true;
  els.timings.replaceChildren();
  els.error.hidden = true;
  els.photo.hidden = true;
  els.photo.classList.remove("in");
}

// ---- drawing ----
const MAX_CANVAS_WIDTH = 1600;
function showPhoto(bitmap) {
  const scale = Math.min(1, MAX_CANVAS_WIDTH / bitmap.width);
  els.canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  els.canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  els.canvas.getContext("2d").drawImage(bitmap, 0, 0, els.canvas.width, els.canvas.height);
  els.canvas.setAttribute("aria-label", S.stageLabel(0));
  els.photo.hidden = false;
  els.photo.classList.remove("in");
  requestAnimationFrame(() => els.photo.classList.add("in"));
}

function grayToImageData(gray, w, h) {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) rgba.set([gray[i], gray[i], gray[i], 255], i * 4);
  return new ImageData(rgba, w, h);
}
function paint(canvas, imageData) {
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  canvas.getContext("2d").putImageData(imageData, 0, 0);
}

function buildCard(face, i) {
  const index = i + 1;
  const card = h("li", "face");
  card.style.setProperty("--i", String(Math.min(i, 5)));
  card.setAttribute("aria-label", `${S.faceLabel(index)}: ${S.emotions[face.emotion]}, ${Math.round(face.confidence * 100)}%`);

  const views = h("div", "views");
  const figure = (canvas, caption) => {
    const fig = h("figure", "view");
    fig.append(canvas, h("figcaption", "", caption));
    return fig;
  };
  const orig = h("canvas");
  orig.setAttribute("role", "img");
  orig.setAttribute("aria-label", S.faceLabel(index));
  paint(orig, grayToImageData(face.crop.gray, face.crop.w, face.crop.h));
  views.append(figure(orig, S.original));
  if (face.heatmap) {
    const heat = h("canvas");
    heat.setAttribute("role", "img");
    heat.setAttribute("aria-label", S.heatmapAlt(index, S.emotions[face.emotion]));
    paint(heat, new ImageData(face.heatmap, 128, 128));
    views.append(figure(heat, S.heatmap));
  }

  const words = h("ol", "words");
  const faceTag = h("li", "face-tag", String(index));
  faceTag.setAttribute("aria-hidden", "true");
  words.append(faceTag);
  rankProbabilities(face.probabilities).forEach(({ label, p }, rank) => {
    const word = h("li", rank === 0 ? "word top" : "word");
    word.style.setProperty("--p", String(p));
    word.append(h("span", "", S.emotions[label]), h("span", "pct", `${Math.round(p * 100)}%`));
    words.append(word);
  });
  card.append(words, views);

  return card;
}

function renderResult(result) {
  const { width, height } = result.image;
  const many = result.faces.length > 1;
  const cards = [];
  const boxes = [];
  const picks = [];
  const select = (index) => {
    cards.forEach((card, k) => {
      card.hidden = k !== index;
      boxes[k].classList.toggle("active", k === index);
      picks[k]?.setAttribute("aria-pressed", String(k === index));
    });
  };
  result.faces.forEach((face, i) => {
    const box = h("div", "box");
    box.style.left = `${(face.box.x / width) * 100}%`;
    box.style.top = `${(face.box.y / height) * 100}%`;
    box.style.width = `${(face.box.w / width) * 100}%`;
    box.style.height = `${(face.box.h / height) * 100}%`;
    box.append(h("span", "tag", String(i + 1)));
    els.boxes.append(box);
    boxes.push(box);
    const card = buildCard(face, i);
    els.faces.append(card);
    cards.push(card);
    if (many) {
      const pick = h("button", "pick");
      pick.type = "button";
      pick.setAttribute("aria-label", `${S.faceLabel(i + 1)}: ${S.emotions[face.emotion]}`);
      const thumb = h("canvas");
      paint(thumb, grayToImageData(face.crop.gray, face.crop.w, face.crop.h));
      pick.append(thumb, h("span", "pick-n", String(i + 1)));
      pick.addEventListener("click", () => select(i));
      box.addEventListener("click", () => select(i));
      els.strip.append(pick);
      picks.push(pick);
    }
  });
  els.strip.hidden = !many;
  els.boxes.classList.toggle("multi", many);
  select(0);
  els.canvas.setAttribute("aria-label", S.stageLabel(result.faces.length));

  const t = result.timings;
  const items = [[S.timings.detect, `${Math.round(t.detect)} ms`], [S.timings.classify, `${Math.round(t.classify)} ms`], [S.timings.gradcam, `${Math.round(t.gradcam)} ms`]];
  for (const [label, value] of items) {
    const cell = h("div");
    cell.append(h("dt", "", label), h("dd", "", value));
    els.timings.append(cell);
  }
  const backend = h("div");
  backend.append(h("dt", "", S.timings.runtime), h("dd", "", S.backend));
  els.timings.append(backend);
  els.timings.hidden = false;

  setStatus(result.faces.length === 0 ? `${S.noFace} ${S.noFaceTips}` : S.result(result.faces.length, t.total));
  return nextFrame().then(() => {
    els.boxes.classList.add("in");
    cards.forEach((c) => c.classList.add("in"));
  });
}

// ---- the main flow ----
async function run(blob) {
  if (!blob) return;
  if (busy) {
    setStatus(S.busy);
    return;
  }
  setBusy(true);
  lastBlob = blob;
  clearResults();
  let bitmap = null;
  try {
    const first = validateFile(blob);
    if (!first.ok) throw { code: first.code };
    try {
      bitmap = await createImageBitmap(blob, { imageOrientation: "from-image", colorSpaceConversion: "none", premultiplyAlpha: "none" });
    } catch {
      throw { code: "decode" };
    }
    const second = validateFile(blob, { width: bitmap.width, height: bitmap.height });
    if (!second.ok) {
      bitmap.close();
      throw { code: second.code };
    }
    showPhoto(bitmap);

    if (!ready) {
      setState("loading");
      els.progress.hidden = false;
      renderProgress();
      await ensureReady();
      els.progress.hidden = true;
    }
    setState("analyzing");
    setStatus(S.detecting);
    resultDeferred = deferred();
    worker.postMessage({ type: "analyze", bitmap, explain: true }, [bitmap]);
    bitmap = null; // transferred: the worker closes it
    const result = await resultDeferred.promise;
    await renderResult(result);
    setState("result");
  } catch (error) {
    bitmap?.close(); // never reached the worker (download failed, say): free the pixels
    els.progress.hidden = true;
    setState("error");
    showError(error?.code ?? "inference");
  } finally {
    setBusy(false);
  }
}

// ---- events ----
els.choose.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => {
  const file = els.file.files[0];
  els.file.value = "";
  run(file);
});
async function runExample() {
  try {
    const response = await fetch("examples/astronaut.jpg");
    if (!response.ok) throw new Error(String(response.status));
    run(await response.blob());
  } catch {
    setState("error");
    showError("download");
  }
}
els.example.addEventListener("click", runExample);
els.plate.addEventListener("click", runExample);
els.retry.addEventListener("click", () => run(lastBlob));
for (const button of [els.choose, els.example, els.plate]) {
  button.addEventListener("pointerenter", prefetch);
  button.addEventListener("focus", prefetch);
}
// the whole page is the drop target; always cancel the default so a miss never navigates away
for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (e) => {
    e.preventDefault();
    document.body.classList.add("over");
  });
}
document.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("over");
});
document.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("over");
  run(e.dataTransfer?.files?.[0]);
});

if (!supported) {
  setBusy(false);
  showError("unsupported");
  els.choose.disabled = els.example.disabled = els.plate.disabled = true;
}
setState("idle");
