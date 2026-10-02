import test from "node:test";
import assert from "node:assert/strict";
import { formatBytes, rankProbabilities, validateFile } from "../src/view.js";

const MB = 1024 * 1024;

test("validateFile rejeita arquivo acima de 10 MB", () => {
  assert.deepEqual(validateFile({ size: 11 * MB, type: "image/png" }), { ok: false, code: "size" });
});

test("validateFile rejeita formatos fora de JPEG, PNG e WebP", () => {
  assert.deepEqual(validateFile({ size: 1000, type: "image/gif" }), { ok: false, code: "format" });
  assert.deepEqual(validateFile({ size: 1000, type: "application/pdf" }), { ok: false, code: "format" });
  assert.deepEqual(validateFile({ size: 1000, type: "" }), { ok: false, code: "format" });
});

test("validateFile rejeita imagens acima de 25 megapixels", () => {
  assert.deepEqual(validateFile({ size: 2 * MB, type: "image/jpeg" }, { width: 6000, height: 5000 }), { ok: false, code: "pixels" });
});

test("validateFile aceita os formatos permitidos dentro dos limites", () => {
  for (const type of ["image/jpeg", "image/png", "image/webp"]) {
    assert.deepEqual(validateFile({ size: 2 * MB, type }), { ok: true });
  }
  assert.deepEqual(validateFile({ size: 10 * MB, type: "image/png" }, { width: 5000, height: 5000 }), { ok: true });
});

test("validateFile: o formato vem antes do tamanho", () => {
  assert.equal(validateFile({ size: 50 * MB, type: "image/gif" }).code, "format");
});

test("rankProbabilities ordena e preserva os 7 rótulos", () => {
  const ranked = rankProbabilities({ angry: 0.1, disgust: 0.02, fear: 0.03, happy: 0.6, sad: 0.05, surprise: 0.04, neutral: 0.16 });
  assert.equal(ranked.length, 7);
  assert.deepEqual(ranked.map((r) => r.label).slice(0, 3), ["happy", "neutral", "angry"]);
  assert.ok(ranked.every((r, i) => i === 0 || ranked[i - 1].p >= r.p));
});

test("formatBytes usa vírgula decimal como em pt-BR", () => {
  assert.equal(formatBytes(46_900_000), "46,9 MB");
  assert.equal(formatBytes(14_200_000), "14,2 MB");
  assert.equal(formatBytes(2_300), "2,3 kB");
  assert.equal(formatBytes(120), "120 B");
});
