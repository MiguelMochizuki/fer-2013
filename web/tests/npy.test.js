import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { parseNpy } from "../src/npy.js";

/** Build an .npy v1.0 file in memory. */
function makeNpy(shape, values, { descr = "<f4", fortran = false } = {}) {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header = `{'descr': '${descr}', 'fortran_order': ${fortran ? "True" : "False"}, 'shape': (${dims}), }`;
  const total = 10 + header.length + 1;
  header += " ".repeat((64 - (total % 64)) % 64) + "\n";
  const out = new Uint8Array(10 + header.length + values.length * 4);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), 10);
  new Float32Array(out.buffer, 10 + header.length, values.length).set(values);
  return out.buffer;
}

test("reads little-endian float32 in C order", () => {
  const values = Float32Array.from({ length: 7 * 512 }, (_, i) => i / 100);
  const { shape, data } = parseNpy(makeNpy([7, 512], values));
  assert.deepEqual(shape, [7, 512]);
  assert.equal(data.length, 7 * 512);
  assert.equal(data[0], 0);
  assert.equal(data[513], Math.fround(5.13));
});

test("accepts a one-dimension shape with a trailing comma", () => {
  const { shape, data } = parseNpy(makeNpy([3], Float32Array.of(1, 2, 3)));
  assert.deepEqual(shape, [3]);
  assert.deepEqual([...data], [1, 2, 3]);
});

test("rejeita fortran_order e dtype diferente de float32", () => {
  assert.throws(() => parseNpy(makeNpy([2, 2], new Float32Array(4), { fortran: true })), Error);
  assert.throws(() => parseNpy(makeNpy([2, 2], new Float32Array(4), { descr: "<f8" })), Error);
  assert.throws(() => parseNpy(new Uint8Array([1, 2, 3, 4]).buffer), Error);
});

const REAL = new URL("../../models/fer_fc_weight.npy", import.meta.url);
test("reads the real fc_weight the same as numpy", { skip: !existsSync(REAL) }, async () => {
  const buf = await readFile(REAL);
  const { shape, data } = parseNpy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  assert.deepEqual(shape, [7, 512]);
  for (const [i, v] of [[0, -0.0379767082631588], [1, -0.03637956827878952], [511, 0.04237420856952667], [512, 0.0328083299100399], [3583, -0.060277462005615234]]) {
    assert.equal(data[i], Math.fround(v));
  }
});
