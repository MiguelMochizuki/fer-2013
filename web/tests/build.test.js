import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assertSafeOutDir, build } from "../build.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const FAKE = {
  "fer_resnet18.onnx": Buffer.from("classifier-bytes".repeat(50)),
  "fer_fc_weight.npy": Buffer.from("fc-weight-bytes".repeat(10)),
  "face_detection_yunet_2026may.onnx": Buffer.from("detector-bytes".repeat(20)),
};

/** A temp models dir with fake files and a meta.json that pins their hashes. */
function setup({ pinWrong = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), "web-build-"));
  const models = join(root, "models");
  mkdirSync(models);
  for (const [name, bytes] of Object.entries(FAKE)) writeFileSync(join(models, name), bytes);
  const meta = {
    classifier_sha256: pinWrong ? "0".repeat(64) : sha(FAKE["fer_resnet18.onnx"]),
    fc_weight_sha256: sha(FAKE["fer_fc_weight.npy"]),
    yunet_2026_sha256: sha(FAKE["face_detection_yunet_2026may.onnx"]),
  };
  const metaPath = join(root, "meta.json");
  writeFileSync(metaPath, JSON.stringify(meta));
  return { root, models, metaPath, out: join(root, "dist") };
}

test("builds dist with the page, modules, fonts, icons, licenses, runtime and manifest", async () => {
  const { models, metaPath, out } = setup();
  await build({ modelsDir: models, outDir: out, metaPath });
  for (const f of [
    "index.html", "css/app.css", "src/app.js", "src/worker.js", "src/pipeline.js",
    "fonts/InstrumentSans-Variable.woff2", "fonts/LICENSE-InstrumentSans.txt", "icons/github-logo.svg", "icons/LICENSE-Phosphor.txt",
    "licenses/LICENSE-YuNet.txt", "licenses/LICENSE-onnxruntime.txt", "examples/astronaut.jpg", "ort/ort.wasm.min.mjs", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm",
    "manifest.json",
  ]) assert.ok(existsSync(join(out, f)), `falta ${f}`);
  assert.equal(existsSync(join(out, "tests")), false, "tests do not ship with the site");
  assert.equal(existsSync(join(out, "node_modules")), false);
});

test("models ship with sha8 in the name and the manifest matches", async () => {
  const { models, metaPath, out } = setup();
  await build({ modelsDir: models, outDir: out, metaPath });
  const manifest = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8"));
  assert.deepEqual(Object.keys(manifest.models).sort(), ["classifier", "detector", "fcWeight"]);
  for (const [key, entry] of Object.entries(manifest.models)) {
    assert.match(entry.path, new RegExp(`^models/${key}\\.[0-9a-f]{8}\\.(onnx|npy)$`));
    const bytes = readFileSync(join(out, entry.path));
    assert.equal(sha(bytes), entry.sha256, key);
    assert.equal(statSync(join(out, entry.path)).size, entry.size, key);
    assert.ok(entry.path.includes(entry.sha256.slice(0, 8)), "o nome carrega o hash");
  }
});

test("paths in the HTML and the manifest are relative", async () => {
  const { models, metaPath, out } = setup();
  await build({ modelsDir: models, outDir: out, metaPath });
  const html = readFileSync(join(out, "index.html"), "utf8");
  assert.doesNotMatch(html, /(?:src|href)="\//, "no absolute path in the HTML");
  const external = [...html.matchAll(/<(?:script|link)\b[^>]*(?:src|href)="(https?:)?\/\//g)];
  assert.deepEqual(external, [], "no external script or stylesheet");
  const manifest = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8"));
  for (const { path } of Object.values(manifest.models)) assert.doesNotMatch(path, /^(\/|https?:)/);
});

test("fails when the model hash differs from the golden", async () => {
  const { models, metaPath, out } = setup({ pinWrong: true });
  await assert.rejects(build({ modelsDir: models, outDir: out, metaPath }), /fer_resnet18\.onnx.*golden|golden.*fer_resnet18\.onnx/s);
  assert.equal(existsSync(join(out, "manifest.json")), false, "publishes nothing if something differs");
});

test("fails with a clear message when a model is missing", async () => {
  const { models, metaPath, out } = setup();
  const { rmSync } = await import("node:fs");
  rmSync(join(models, "fer_fc_weight.npy"));
  await assert.rejects(build({ modelsDir: models, outDir: out, metaPath }), /fer_fc_weight\.npy/);
});

test("assertSafeOutDir refuses the sources, the repo and their parents, and accepts a dist folder", () => {
  const web = join(import.meta.dirname, "..");
  for (const outDir of [web, join(web, ".."), join(web, "..", ".."), "/"]) {
    assert.throws(() => assertSafeOutDir(outDir), /refus/i, outDir);
  }
  assert.doesNotThrow(() => assertSafeOutDir(join(web, "dist")));
  assert.doesNotThrow(() => assertSafeOutDir(join(tmpdir(), "site")));
});
