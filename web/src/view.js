/** Pure helpers for the page: no DOM, so they are unit-tested in Node. */

export const MAX_FILE_BYTES = 10 * 1024 * 1024;
export const MAX_PIXELS = 25_000_000;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

/**
 * Client-side limits, the same ones the API enforces (format, size, pixels).
 * The format is checked first so a 50 MB GIF is reported as a bad format.
 * @param {{ size: number, type: string }} file
 * @param {{ width: number, height: number }} [dims]  known once the image is decoded
 * @returns {{ ok: true } | { ok: false, code: "format" | "size" | "pixels" }}
 */
export function validateFile(file, dims) {
  if (!ALLOWED_TYPES.has(file.type)) return { ok: false, code: "format" };
  if (file.size > MAX_FILE_BYTES) return { ok: false, code: "size" };
  if (dims && dims.width * dims.height > MAX_PIXELS) return { ok: false, code: "pixels" };
  return { ok: true };
}

/**
 * @param {Record<string, number>} probabilities
 * @returns {{ label: string, p: number }[]} highest probability first
 */
export function rankProbabilities(probabilities) {
  return Object.entries(probabilities)
    .map(([label, p]) => ({ label, p }))
    .sort((a, b) => b.p - a.p);
}

/** Decimal units: 46.9 MB. */
export function formatBytes(n) {
  const decimal = (v) => v.toFixed(1);
  if (n >= 1e6) return `${decimal(n / 1e6)} MB`;
  if (n >= 1e3) return `${decimal(n / 1e3)} kB`;
  return `${n} B`;
}
