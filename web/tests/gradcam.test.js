import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { gradcamPP, renderOverlay } from "../src/gradcam.js";
import { parseNpy } from "../src/npy.js";

const FC = new URL("../../models/fer_fc_weight.npy", import.meta.url);
const skip = !existsSync(FC);
const gold = JSON.parse(readFileSync(new URL("./golden/gradcam.json", import.meta.url), "utf8"));
const cls = JSON.parse(readFileSync(new URL("./golden/classifier.json", import.meta.url), "utf8"));
const pre = JSON.parse(readFileSync(new URL("./golden/preprocess.json", import.meta.url), "utf8"));
const floats = (b64) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const loadFc = () => {
  const buf = readFileSync(FC);
  return parseNpy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)).data;
};

test("mapa 7x7 igual ao Python nas 7 classes até 1e-4", { skip }, () => {
  const fc = loadFc();
  const features = floats(cls.features_f32_b64);
  gold.cams.forEach((want, c) => {
    const got = gradcamPP(features, fc, c);
    assert.equal(got.length, 49);
    const max = want.reduce((m, v, i) => Math.max(m, Math.abs(v - got[i])), 0);
    assert.ok(max <= 1e-4, `classe ${c}: ${max}`);
  });
});

test("features zeradas dão zeros sem NaN", { skip }, () => {
  const cam = gradcamPP(new Float32Array(512 * 49), loadFc(), 3);
  assert.ok(cam.every((v) => v === 0));
});

test("mapa fica em [0,1]", { skip }, () => {
  const rng = (i) => Math.abs(Math.sin(i * 12.9898) * 43758.5453) % 1;
  const features = Float32Array.from({ length: 512 * 49 }, (_, i) => rng(i));
  const cam = gradcamPP(features, loadFc(), 2);
  assert.ok(cam.every((v) => v >= 0 && v <= 1));
  assert.ok(cam.some((v) => v > 0));
});

test("overlay RGBA 128x128, alfa 255, até 1 nível do Python", () => {
  const face = pre.fixtures[gold.fixture].faces[0];
  const gray = Uint8Array.from(Buffer.from(face.crop_b64, "base64"));
  const cam = Float32Array.from(gold.cams[gold.overlay_class]);
  const rgba = renderOverlay(gray, face.crop_w, face.crop_h, cam, gold.overlay_size);
  assert.equal(rgba.length, 128 * 128 * 4);
  const want = Uint8Array.from(Buffer.from(gold.overlay_rgb_b64, "base64"));
  let max = 0;
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    assert.equal(rgba[i + 3], 255);
    for (let c = 0; c < 3; c++) max = Math.max(max, Math.abs(rgba[i + c] - want[j + c]));
  }
  assert.ok(max <= 1, `diferença máxima ${max}`);
});
