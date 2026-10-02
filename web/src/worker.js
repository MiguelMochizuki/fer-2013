/**
 * Module worker: loads the models and runs the pipeline off the main thread.
 *
 * Messages in:  { type: "init" } | { type: "analyze", bitmap: ImageBitmap, explain?: boolean }
 * Messages out: { type: "progress", loaded, total } | { type: "ready" }
 *               { type: "result", result } | { type: "error", code }
 *   code: "unsupported" | "integrity" | "download" | "decode" | "inference"
 */

import * as ort from "../ort/ort.wasm.min.mjs";
import { ModelDownloadError, ModelIntegrityError, loadModels } from "./models.js";
import { parseNpy } from "./npy.js";
import { createPipeline } from "./pipeline.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;

const CACHE_NAME = "fer-models";
let pipelinePromise = null;

async function openCache() {
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null; // private mode or blocked storage: just don't cache
  }
}

async function buildPipeline() {
  const models = await loadModels({
    manifestUrl: new URL("../manifest.json", import.meta.url).href,
    cache: await openCache(),
    onProgress: (loaded, total) => postMessage({ type: "progress", loaded, total }),
  });
  const create = (buffer) => ort.InferenceSession.create(new Uint8Array(buffer), { executionProviders: ["wasm"] });
  const pipeline = createPipeline({
    ort,
    detectorSession: await create(models.detector),
    classifierSession: await create(models.classifier),
    fcWeight: parseNpy(models.fcWeight).data,
  });
  // warm-up: the first run pays for WASM compilation and memory growth
  await pipeline.analyze(new Uint8Array(64 * 64 * 3).fill(128), 64, 64, { explain: false });
  return pipeline;
}

function getPipeline() {
  pipelinePromise ??= buildPipeline().catch((error) => {
    pipelinePromise = null; // allow a retry
    throw error;
  });
  return pipelinePromise;
}

function codeFor(error, fallback) {
  if (error instanceof ModelIntegrityError) return "integrity";
  if (error instanceof ModelDownloadError) return "download";
  return fallback;
}

self.onmessage = async ({ data }) => {
  if (data.type === "init") {
    try {
      await getPipeline();
      postMessage({ type: "ready" });
    } catch (error) {
      postMessage({ type: "error", code: codeFor(error, "download") });
    }
  } else if (data.type === "analyze") {
    let pixels;
    let width;
    let height;
    try {
      width = data.bitmap.width;
      height = data.bitmap.height;
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(data.bitmap, 0, 0);
      data.bitmap.close();
      pixels = ctx.getImageData(0, 0, width, height).data;
    } catch {
      postMessage({ type: "error", code: "decode" });
      return;
    }
    try {
      const pipeline = await getPipeline();
      const result = await pipeline.analyze(pixels, width, height, { explain: data.explain !== false, stride: 4 });
      postMessage({ type: "result", result });
    } catch (error) {
      postMessage({ type: "error", code: codeFor(error, "inference") });
    }
  }
};
