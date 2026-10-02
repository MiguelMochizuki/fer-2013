import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { createPipeline } from "../src/pipeline.js";
import { parseNpy } from "../src/npy.js";
import { resizeU8 } from "../src/resample.js";

const ROOT = new URL("../../models/", import.meta.url);
const FILES = ["fer_resnet18.onnx", "fer_fc_weight.npy", "face_detection_yunet_2026may.onnx"];
const skip = !FILES.every((f) => existsSync(new URL(f, ROOT)));
const golden = JSON.parse(readFileSync(new URL("./golden/analyze.json", import.meta.url), "utf8"));

function fixture(name) {
  const png = PNG.sync.read(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
  const rgb = new Uint8Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) rgb.set(png.data.subarray(i, i + 3), j);
  return { rgb, rgba: Uint8Array.from(png.data), w: png.width, h: png.height };
}
const iou = (a, b) => {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (w * h) / (a.w * a.h + b.w * b.h - w * h);
};

let cached;
async function pipeline() {
  if (cached) return cached;
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const make = async (f) => ort.InferenceSession.create(readFileSync(new URL(f, ROOT)), { executionProviders: ["wasm"] });
  const buf = readFileSync(new URL("fer_fc_weight.npy", ROOT));
  const fcWeight = parseNpy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).data;
  cached = createPipeline({
    ort,
    detectorSession: await make("face_detection_yunet_2026may.onnx"),
    classifierSession: await make("fer_resnet18.onnx"),
    fcWeight,
  });
  return cached;
}

test("analyze reproduces Python on the three fixtures", { skip }, async () => {
  const p = await pipeline();
  for (const [name, want] of Object.entries(golden.fixtures)) {
    const { rgb, w, h } = fixture(name);
    const got = await p.analyze(rgb, w, h);
    assert.deepEqual(got.image, { width: w, height: h });
    assert.equal(got.faces.length, want.faces.length, `${name}: faces`);
    got.faces.forEach((f, i) => {
      const t = want.faces[i];
      assert.equal(f.emotion, t.emotion, `${name}[${i}] emotion`);
      assert.ok(Math.abs(f.confidence - t.confidence) <= 0.01, `${name}[${i}] confidence ${f.confidence} vs ${t.confidence}`);
      assert.ok(iou(f.box, t.box) >= 0.99, `${name}[${i}] IoU ${iou(f.box, t.box)}`);
      assert.ok(Number.isInteger(f.box.x) && Number.isInteger(f.box.w), "integer boxes as in Python");
      const maxCam = t.cam.reduce((m, v, k) => Math.max(m, Math.abs(v - f.cam[k])), 0);
      assert.ok(maxCam <= 2e-2, `${name}[${i}] cam ${maxCam}`); // int8 features differ a little between WASM and native
      assert.equal(f.heatmap.length, 128 * 128 * 4);
      assert.deepEqual(Object.keys(f.probabilities), ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]);
    });
  }
});

test("explain false does not compute a heatmap", { skip }, async () => {
  const p = await pipeline();
  const { rgb, w, h } = fixture("face.png");
  const got = await p.analyze(rgb, w, h, { explain: false });
  assert.equal(got.faces.length, 1);
  assert.equal(got.faces[0].heatmap, null);
  assert.equal(got.faces[0].cam, null);
  assert.equal(got.timings.gradcam, 0);
});

test("many faces: all of them come back, in more than one classifier batch", { skip }, async () => {
  const p = await pipeline();
  const { rgb: face, w: fw, h: fh } = fixture("face.png");
  const tile = 150;
  const cols = 5;
  const rows = 4;
  const small = resizeU8(face, fw, fh, 3, tile, tile, "bicubic");
  const W = cols * tile;
  const H = rows * tile;
  const canvas = new Uint8Array(W * H * 3).fill(128);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      for (let y = 0; y < tile; y++) {
        canvas.set(small.subarray(y * tile * 3, (y + 1) * tile * 3), ((r * tile + y) * W + c * tile) * 3);
      }
    }
  }
  const got = await p.analyze(canvas, W, H, { explain: true });
  assert.equal(got.faces.length, cols * rows);
  for (const f of got.faces) {
    const sum = Object.values(f.probabilities).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-4);
    assert.equal(f.cam.length, 49);
  }
});

test("RGBA with varying alpha gives the same result as RGB", { skip }, async () => {
  const p = await pipeline();
  const { rgb, rgba, w, h } = fixture("face.png");
  for (let i = 3; i < rgba.length; i += 4) rgba[i] = (i >> 2) % 256; // varying alpha
  const a = await p.analyze(rgb, w, h, { explain: false });
  const b = await p.analyze(rgba, w, h, { explain: false, stride: 4 });
  assert.equal(b.faces.length, a.faces.length);
  assert.equal(b.faces[0].emotion, a.faces[0].emotion);
  assert.ok(Math.abs(b.faces[0].confidence - a.faces[0].confidence) < 1e-6);
});

test("an image without a face returns no faces", { skip }, async () => {
  const p = await pipeline();
  const got = await p.analyze(new Uint8Array(300 * 300 * 3).fill(128), 300, 300);
  assert.deepEqual(got.faces, []);
});

test("timings are non-negative numbers and total >= detect", { skip }, async () => {
  const p = await pipeline();
  const { rgb, w, h } = fixture("face.png");
  const { timings } = await p.analyze(rgb, w, h);
  for (const v of Object.values(timings)) assert.ok(Number.isFinite(v) && v >= 0);
  assert.ok(timings.total >= timings.detect);
});
