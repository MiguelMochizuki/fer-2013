import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { EMOTIONS, createClassifier } from "../src/classifier.js";

const MODEL = new URL("../../models/fer_resnet18.onnx", import.meta.url);
const pre = JSON.parse(readFileSync(new URL("./golden/preprocess.json", import.meta.url), "utf8"));
const gold = JSON.parse(readFileSync(new URL("./golden/classifier.json", import.meta.url), "utf8"));
const floats = (b64) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** Normalized CHW input built from a golden 224x224 plane. */
function inputFromPlane(b64) {
  const plane = floats(b64);
  const out = new Float32Array(3 * plane.length);
  for (let c = 0; c < 3; c++) for (let i = 0; i < plane.length; i++) out[c * plane.length + i] = (plane[i] - MEAN[c]) / STD[c];
  return out;
}
const allFaces = [];
for (const name of Object.keys(pre.fixtures)) {
  pre.fixtures[name].faces.forEach((f, i) => allFaces.push({ name, i, plane: f.plane224_f32_b64, probs: gold.fixtures[name].faces[i].probs }));
}
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

async function realClassifier() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(readFileSync(MODEL), { executionProviders: ["wasm"] });
  return createClassifier(ort, session);
}
const skip = !existsSync(MODEL);

test("EMOTIONS na ordem do treino", () => {
  assert.deepEqual([...EMOTIONS], ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]);
});

test("WASM probs match Python within 1e-4", { skip }, async () => {
  const clf = await realClassifier();
  assert.ok(allFaces.length >= 3);
  for (const f of allFaces) {
    const { probs } = await clf.classify(inputFromPlane(f.plane), 1);
    assert.ok(maxDiff([...probs], f.probs) <= 1e-4, `${f.name}[${f.i}] ${maxDiff([...probs], f.probs)}`);
  }
});

test("features match Python within 1e-3", { skip }, async () => {
  const clf = await realClassifier();
  const f = allFaces.find((x) => x.name === gold.features_fixture);
  const { features } = await clf.classify(inputFromPlane(f.plane), 1);
  const want = floats(gold.features_f32_b64);
  assert.equal(features.length, 512 * 49);
  assert.ok(maxDiff([...features], [...want]) <= 1e-3);
});

test("a batch of 3 gives the same as three calls of 1 (within 1e-5)", { skip }, async () => {
  const clf = await realClassifier();
  const three = allFaces.slice(0, 3);
  const batch = new Float32Array(3 * 3 * 224 * 224);
  three.forEach((f, k) => batch.set(inputFromPlane(f.plane), k * 3 * 224 * 224));
  const { probs, features } = await clf.classify(batch, 3);
  assert.equal(probs.length, 21);
  assert.equal(features.length, 3 * 512 * 49);
  for (let k = 0; k < 3; k++) {
    const single = await clf.classify(inputFromPlane(three[k].plane), 1);
    assert.ok(maxDiff([...probs.subarray(k * 7, k * 7 + 7)], [...single.probs]) <= 1e-5, `face ${k}`);
  }
});

test("probs sum to 1 per face", { skip }, async () => {
  const clf = await realClassifier();
  const { probs } = await clf.classify(inputFromPlane(allFaces[0].plane), 1);
  assert.ok(Math.abs([...probs].reduce((a, b) => a + b, 0) - 1) < 1e-5);
});
