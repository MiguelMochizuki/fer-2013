import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { createYuNetDetector, decodeYunet, nms } from "../src/detector.js";

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
const iou = (a, b) => {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return (w * h) / (a.w * a.h + b.w * b.h - w * h);
};

/** Zeroed YuNet head outputs for a padded H x W input. */
function emptyOutputs(H, W) {
  const out = {};
  for (const s of [8, 16, 32]) {
    const n = (H / s) * (W / s);
    out[`cls_${s}`] = new Float32Array(n);
    out[`obj_${s}`] = new Float32Array(n);
    out[`bbox_${s}`] = new Float32Array(n * 4);
    out[`kps_${s}`] = new Float32Array(n * 10);
  }
  return out;
}

test("decodeYunet decodes a known cell", () => {
  const out = emptyOutputs(32, 64); // 4x8 grid at stride 8
  const idx = 2 * 8 + 3; // row 2, column 3
  out.cls_8[idx] = 0.81;
  out.obj_8[idx] = 0.81;
  out.bbox_8.set([0.5, 0.25, Math.log(2), Math.log(3)], idx * 4);
  const boxes = decodeYunet(out, 32, 64, 0.6);
  assert.equal(boxes.length, 1);
  const b = boxes[0];
  near(b.x, 20, 1e-4, "x");
  near(b.y, 6, 1e-4, "y");
  near(b.w, 16, 1e-4, "w");
  near(b.h, 24, 1e-4, "h");
  near(b.score, 0.81, 1e-6, "score");
});

test("decodeYunet descarta scores abaixo do limiar", () => {
  const out = emptyOutputs(32, 64);
  out.cls_8[19] = 0.5;
  out.obj_8[19] = 0.5; // score 0.5 < 0.6
  assert.equal(decodeYunet(out, 32, 64, 0.6).length, 0);
});

test("nms keeps the highest score and drops IoU > 0.3", () => {
  const a = { x: 0, y: 0, w: 10, h: 10, score: 0.9 };
  const b = { x: 3, y: 0, w: 10, h: 10, score: 0.8 }; // IoU 0.54 with a
  const c = { x: 100, y: 100, w: 10, h: 10, score: 0.7 };
  assert.deepEqual(nms([b, c, a], 0.3), [a, c]);
});

/** A session that records the input tensor and returns empty heads. */
function stubOrt() {
  const calls = [];
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  }
  const session = {
    async run(feeds) {
      calls.push(feeds.input);
      const [, , H, W] = feeds.input.dims;
      return Object.fromEntries(Object.entries(emptyOutputs(H, W)).map(([k, v]) => [k, { data: v }]));
    },
  };
  return { ort: { Tensor }, session, calls };
}

test("pads to a multiple of 32 without scaling up", async () => {
  const { ort, session, calls } = stubOrt();
  await createYuNetDetector(ort, session).detect(new Uint8Array(100 * 70 * 3), 100, 70);
  assert.deepEqual(calls[0].dims, [1, 3, 96, 128]);
});

test("reduz o maior lado a 640 antes de detectar", async () => {
  const { ort, session, calls } = stubOrt();
  await createYuNetDetector(ort, session).detect(new Uint8Array(1200 * 600 * 3), 1200, 600);
  assert.deepEqual(calls[0].dims, [1, 3, 320, 640]);
});

test("the input is BGR 0 to 255 with zeros in the padding", async () => {
  const { ort, session, calls } = stubOrt();
  const rgb = new Uint8Array(2 * 1 * 3);
  rgb.set([10, 20, 30, 40, 50, 60]); // two RGB pixels
  await createYuNetDetector(ort, session).detect(rgb, 2, 1);
  const t = calls[0];
  assert.deepEqual(t.dims, [1, 3, 32, 32]);
  const plane = 32 * 32;
  assert.deepEqual([t.data[0], t.data[plane], t.data[2 * plane]], [30, 20, 10]); // B, G, R of pixel 0
  assert.deepEqual([t.data[1], t.data[plane + 1], t.data[2 * plane + 1]], [60, 50, 40]);
  assert.equal(t.data[2], 0);
});

test("scales the boxes back to the original image", async () => {
  const calls = [];
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  }
  const session = {
    async run(feeds) {
      calls.push(feeds.input.dims);
      const out = emptyOutputs(320, 640);
      const idx = 2 * (640 / 8) + 3; // row 2, column 3 in an 80-column grid
      out.cls_8[idx] = 0.81;
      out.obj_8[idx] = 0.81;
      out.bbox_8.set([0.5, 0.25, Math.log(2), Math.log(3)], idx * 4);
      return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, { data: v }]));
    },
  };
  // 1200x600 -> 640x320 (scale 0.5333...): the box (20, 6, 16, 24) comes back multiplied by 1/scale
  const [box] = await createYuNetDetector({ Tensor }, session).detect(new Uint8Array(1200 * 600 * 3), 1200, 600);
  const scale = 640 / 1200;
  near(box.x, 20 / scale, 1e-3, "x");
  near(box.w, 16 / scale, 1e-3, "w");
});

const MODEL = new URL("../../models/face_detection_yunet_2026may.onnx", import.meta.url);
const golden = JSON.parse(readFileSync(new URL("./golden/detector.json", import.meta.url), "utf8"));
function fixtureRgb(name) {
  const png = PNG.sync.read(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
  const rgb = new Uint8Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) rgb.set(png.data.subarray(i, i + 3), j);
  return { rgb, w: png.width, h: png.height };
}
async function realDetector() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(readFileSync(MODEL), { executionProviders: ["wasm"] });
  return createYuNetDetector(ort, session);
}

test("detect: count, boxes and scores match OpenCV", { skip: !existsSync(MODEL) }, async () => {
  const detector = await realDetector();
  for (const [name, want] of Object.entries(golden.fixtures)) {
    const { rgb, w, h } = fixtureRgb(name);
    const got = await detector.detect(rgb, w, h);
    assert.equal(got.length, want.faces.length, `${name}: number of faces`);
    assert.ok(got.length >= 1, `${name}: the fixture should contain a face`);
    got.forEach((g, i) => {
      assert.ok(iou(g, want.faces[i]) >= 0.999, `${name}[${i}] IoU ${iou(g, want.faces[i])}`);
      near(g.score, want.faces[i].score, 1e-3, `${name}[${i}] score`);
    });
  }
});

test("a thin or tiny image returns an empty list", { skip: !existsSync(MODEL) }, async () => {
  const detector = await realDetector();
  for (const [w, h] of [[5000, 2], [1, 1], [3000, 5], [2, 5000]]) {
    assert.deepEqual(await detector.detect(new Uint8Array(w * h * 3), w, h), [], `${w}x${h}`);
  }
});
