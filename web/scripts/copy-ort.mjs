#!/usr/bin/env node
/**
 * Copies the onnxruntime-web files the page needs (single-thread WASM, SIMD) next to the sources.
 *
 *   node scripts/copy-ort.mjs [outDir]   (default: web/ort)
 */
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const ORT_FILES = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

/**
 * Copy the runtime files from `node_modules/onnxruntime-web/dist` into `outDir`.
 * @param {string} outDir  destination folder; created if missing
 */
export function copyOrt(outDir) {
  const root = join(dirname(fileURLToPath(import.meta.url)), "..");
  const from = join(root, "node_modules", "onnxruntime-web", "dist");
  mkdirSync(outDir, { recursive: true });
  for (const f of ORT_FILES) copyFileSync(join(from, f), join(outDir, f));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const out = process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), "..", "ort");
  copyOrt(out);
  console.log(`copied ${ORT_FILES.length} runtime files to ${out}`);
}
