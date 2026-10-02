#!/usr/bin/env node
/**
 * Assembles the static site for GitHub Pages.
 *
 *   node build.mjs --models <dir> --out <dir>
 *
 * Copies the page, modules, fonts, icons and example, the onnxruntime-web
 * runtime, and the three models under hash-stamped names plus a manifest.
 * Every model is checked against the hashes recorded in tests/golden/meta.json
 * first, so a new model release cannot ship before the golden files are
 * regenerated for it. Nothing is written if a check fails.
 */

import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { copyOrt } from "./scripts/copy-ort.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SITE_PARTS = ["index.html", "css", "src", "fonts", "icons", "licenses", "examples"];

// manifest key -> [file in the models dir, extension, meta.json key holding its pinned sha256]
const MODELS = {
  classifier: ["fer_resnet18.onnx", "onnx", "classifier_sha256"],
  detector: ["face_detection_yunet_2026may.onnx", "onnx", "yunet_2026_sha256"],
  fcWeight: ["fer_fc_weight.npy", "npy", "fc_weight_sha256"],
};

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

/** The build wipes `outDir`, so it must never be the sources, the repo, or a parent of either. */
export function assertSafeOutDir(outDir) {
  const target = resolve(outDir);
  for (const dir of [HERE, resolve(HERE, "..")]) {
    if (target === dir || dir.startsWith(target.endsWith(sep) ? target : target + sep)) {
      throw new Error(`refusing to build into ${target}: it would delete ${dir}`);
    }
  }
}

/**
 * @param {{ modelsDir: string, outDir: string, metaPath?: string }} options
 */
export async function build({ modelsDir, outDir, metaPath = join(HERE, "tests", "golden", "meta.json") }) {
  assertSafeOutDir(outDir);
  const meta = JSON.parse(readFileSync(metaPath, "utf8"));

  const loaded = {};
  for (const [key, [file, ext, metaKey]] of Object.entries(MODELS)) {
    const path = join(modelsDir, file);
    if (!existsSync(path)) throw new Error(`missing model ${file} in ${modelsDir}`);
    const bytes = readFileSync(path);
    const digest = sha256(bytes);
    if (digest !== meta[metaKey]) {
      throw new Error(`${file} does not match the golden files (sha256 ${digest.slice(0, 12)}..., golden ${String(meta[metaKey]).slice(0, 12)}...); regenerate them with scripts/make_web_golden.py`);
    }
    loaded[key] = { bytes, digest, ext };
  }

  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(join(outDir, "models"), { recursive: true });
  for (const part of SITE_PARTS) cpSync(join(HERE, part), join(outDir, part), { recursive: true });
  copyOrt(join(outDir, "ort"));

  const manifest = { models: {} };
  for (const [key, { bytes, digest, ext }] of Object.entries(loaded)) {
    const path = `models/${key}.${digest.slice(0, 8)}.${ext}`;
    writeFileSync(join(outDir, path), bytes);
    manifest.models[key] = { path, sha256: digest, size: bytes.length };
  }
  writeFileSync(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const modelsDir = resolve(arg("models", join(HERE, "..", "models")));
  const outDir = resolve(arg("out", join(HERE, "dist")));
  try {
    const manifest = await build({ modelsDir, outDir });
    console.log(`site written to ${outDir}`);
    for (const [key, { path, size }] of Object.entries(manifest.models)) console.log(`  ${key}: ${path} (${size} bytes)`);
  } catch (error) {
    console.error(`build failed: ${error.message}`);
    process.exit(1);
  }
}
