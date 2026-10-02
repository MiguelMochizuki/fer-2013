import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { computeCoefficients, resizeF32, resizeU8, rgbToGray } from "../src/resample.js";

const golden = JSON.parse(readFileSync(new URL("./golden/resample.json", import.meta.url), "utf8"));
const bytes = (b64) => Uint8Array.from(Buffer.from(b64, "base64"));
const floats = (b64) => {
  const b = Buffer.from(b64, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
/** RGB bytes of a PNG fixture, alpha dropped. */
function fixtureRgb(name) {
  const png = PNG.sync.read(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
  const rgb = new Uint8Array(png.width * png.height * 3);
  for (let i = 0, j = 0; i < png.data.length; i += 4, j += 3) rgb.set(png.data.subarray(i, i + 3), j);
  return rgb;
}
const byKind = (kind) => golden.cases.filter((c) => c.kind === kind);

test("resizeU8 is byte-identical to Pillow", () => {
  for (const c of byKind("u8")) {
    const input = c.input_fixture ? fixtureRgb(c.input_fixture) : bytes(c.input_b64);
    const out = resizeU8(input, c.w, c.h, c.channels, c.out_w, c.out_h, c.filter);
    assert.equal(out.length, c.out_w * c.out_h * c.channels, c.name);
    if (c.output_sha256) {
      assert.equal(createHash("sha256").update(out).digest("hex"), c.output_sha256, c.name);
    } else {
      assert.deepEqual(out, bytes(c.output_b64), c.name);
    }
  }
});

test("resizeF32 within 1e-4 of Pillow", () => {
  for (const c of byKind("f32")) {
    const out = resizeF32(floats(c.input_f32_b64), c.w, c.h, c.out_w, c.out_h, c.filter);
    const want = floats(c.output_f32_b64);
    assert.equal(out.length, want.length);
    let max = 0;
    for (let i = 0; i < want.length; i++) max = Math.max(max, Math.abs(out[i] - want[i]));
    assert.ok(max <= 1e-4, `${c.name}: max diff ${max}`);
  }
});

test("rgbToGray matches Pillow and ignores alpha", () => {
  for (const c of byKind("gray")) {
    const out = rgbToGray(bytes(c.input_b64), c.w, c.h, c.stride);
    assert.deepEqual(out, bytes(c.output_b64), c.name);
  }
});

test("the weights of each output pixel sum to 1", () => {
  for (const [inSize, outSize] of [[260, 130], [135, 48], [24, 48], [48, 224], [800, 640], [7, 128]]) {
    for (const filter of ["bilinear", "bicubic", "lanczos"]) {
      const { count, ksize, weights } = computeCoefficients(inSize, outSize, filter);
      for (let i = 0; i < outSize; i++) {
        let sum = 0;
        for (let k = 0; k < count[i]; k++) sum += weights[i * ksize + k];
        assert.ok(Math.abs(sum - 1) < 1e-12, `${filter} ${inSize}->${outSize} pixel ${i}: ${sum}`);
      }
    }
  }
});

test("the same size returns a copy", () => {
  const src = Uint8Array.from([1, 2, 3, 4, 5, 6]);
  const out = resizeU8(src, 2, 1, 3, 2, 1, "lanczos");
  assert.deepEqual(out, src);
  assert.notEqual(out.buffer, src.buffer);
});
