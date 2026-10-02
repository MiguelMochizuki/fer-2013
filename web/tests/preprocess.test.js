import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { cropSquareGray, toInput } from "../src/preprocess.js";

const golden = JSON.parse(readFileSync(new URL("./golden/preprocess.json", import.meta.url), "utf8"));
const bytes = (b64) => Uint8Array.from(Buffer.from(b64, "base64"));
const floats = (b64) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
function fixtureRgb(name) {
  const png = PNG.sync.read(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
  const rgb = new Uint8Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) rgb.set(png.data.subarray(i, i + 3), j);
  return { rgb, rgba: Uint8Array.from(png.data), w: png.width, h: png.height };
}
const blank = (w, h) => new Uint8Array(w * h * 3).fill(120);
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

test("square crop of a tall box", () => {
  const out = cropSquareGray(blank(300, 300), 300, 300, { x: 100, y: 80, w: 90, h: 113 });
  assert.equal(out.w, 135); // trunc(113 * 1.2)
  assert.equal(out.h, 135);
  assert.equal(out.gray.length, 135 * 135);
});

test("the crop slides inside the edge", () => {
  const out = cropSquareGray(blank(300, 300), 300, 300, { x: 0, y: 50, w: 60, h: 80 });
  assert.deepEqual([out.w, out.h], [96, 96]);
});

test("an empty box, outside the image or without area returns null", () => {
  const img = blank(100, 100);
  for (const box of [{ x: 5, y: 5, w: 0, h: 10 }, { x: 500, y: 500, w: 10, h: 10 }, { x: 150, y: 10, w: 20, h: 20 }, { x: -30, y: 10, w: 20, h: 20 }]) {
    assert.equal(cropSquareGray(img, 100, 100, box), null, JSON.stringify(box));
  }
});

test("a negative or edge box is clamped as in Python", () => {
  const a = cropSquareGray(blank(100, 80), 100, 80, { x: -10, y: -10, w: 30, h: 30 });
  assert.deepEqual([a.w, a.h], [36, 36]);
  const b = cropSquareGray(blank(100, 80), 100, 80, { x: 90, y: 70, w: 40, h: 40 });
  assert.deepEqual([b.w, b.h], [48, 48]);
});

test("a square larger than the image gives a non-square crop, as in Python", () => {
  const out = cropSquareGray(blank(100, 20), 100, 20, { x: 10, y: 2, w: 18, h: 18 });
  assert.deepEqual([out.w, out.h], [21, 20]);
});

test("crop, 48x48 and 224 plane match Python", () => {
  for (const [name, fx] of Object.entries(golden.fixtures)) {
    const { rgb, w, h } = fixtureRgb(name);
    assert.ok(fx.faces.length >= 1, name);
    fx.faces.forEach((f, i) => {
      const crop = cropSquareGray(rgb, w, h, f.box);
      assert.deepEqual([crop.w, crop.h], [f.crop_w, f.crop_h], `${name}[${i}] size`);
      assert.deepEqual(crop.gray, bytes(f.crop_b64), `${name}[${i}] recorte cinza`);
      const input = toInput(crop.gray, crop.w, crop.h);
      assert.equal(input.length, 3 * 224 * 224);
      const plane = floats(f.plane224_f32_b64);
      let max = 0;
      for (let c = 0; c < 3; c++) {
        for (let p = 0; p < plane.length; p++) {
          max = Math.max(max, Math.abs(input[c * plane.length + p] - (plane[p] - MEAN[c]) / STD[c]));
        }
      }
      assert.ok(max <= 1e-3, `${name}[${i}] tensor ${max}`);
    });
  }
});

test("RGBA gives the same crop as RGB", () => {
  const { rgb, rgba, w, h } = fixtureRgb("face.png");
  const box = golden.fixtures["face.png"].faces[0].box;
  const a = cropSquareGray(rgb, w, h, box, 0.1, 3);
  const b = cropSquareGray(rgba, w, h, box, 0.1, 4);
  assert.deepEqual(a.gray, b.gray);
});
