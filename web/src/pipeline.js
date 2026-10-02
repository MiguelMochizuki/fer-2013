/**
 * The whole analysis of one photo, mirroring `fer_2013.serving.api._predict`:
 * detect faces, truncate and clamp their boxes to integers inside the image,
 * crop and preprocess each face, classify them all in one call, and (optionally)
 * explain each prediction with Grad-CAM++. No DOM, no worker: runs in Node too.
 */

import { createClassifier, EMOTIONS } from "./classifier.js";
import { createYuNetDetector } from "./detector.js";
import { gradcamPP, renderOverlay } from "./gradcam.js";
import { cropSquareGray, toInput } from "./preprocess.js";

/** Faces per classifier call: bounds the input tensor (about 600 KB per face) in a crowd. */
const BATCH = 16;

/**
 * @typedef {object} Face
 * @property {{ x: number, y: number, w: number, h: number }} box  integer pixels, inside the image
 * @property {string} emotion
 * @property {number} confidence  calibrated probability of `emotion`
 * @property {Record<string, number>} probabilities  calibrated, keyed by emotion
 * @property {{ gray: Uint8Array, w: number, h: number }} crop
 * @property {Float32Array | null} cam  7x7 Grad-CAM++ map (49 values), null unless explained
 * @property {Uint8ClampedArray | null} heatmap  128x128 RGBA overlay, null unless explained
 */

/** Drop the alpha channel of RGBA pixels. */
function toRgb(pixels, w, h) {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
    rgb[j] = pixels[i];
    rgb[j + 1] = pixels[i + 1];
    rgb[j + 2] = pixels[i + 2];
  }
  return rgb;
}

/** Intersection with the image, as integers; null if nothing is left. */
function clampBox(box, width, height) {
  const x0 = Math.max(box.x, 0);
  const y0 = Math.max(box.y, 0);
  const x1 = Math.min(box.x + box.w, width);
  const y1 = Math.min(box.y + box.h, height);
  return x1 <= x0 || y1 <= y0 ? null : { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/**
 * @param {{ ort: object, detectorSession: object, classifierSession: object, fcWeight: Float32Array }} deps
 */
export function createPipeline({ ort, detectorSession, classifierSession, fcWeight }) {
  const detector = createYuNetDetector(ort, detectorSession);
  const classifier = createClassifier(ort, classifierSession);

  return {
    /**
     * @param {Uint8Array | Uint8ClampedArray} pixels  w*h*stride bytes
     * @param {number} w
     * @param {number} h
     * @param {{ explain?: boolean, stride?: 3 | 4 }} [options]
     * @returns {Promise<{ image: { width: number, height: number }, faces: Face[], timings: { detect: number, classify: number, gradcam: number, total: number } }>}
     */
    async analyze(pixels, w, h, { explain = true, stride = 3 } = {}) {
      const t0 = performance.now();
      const rgb = stride === 4 ? toRgb(pixels, w, h) : pixels;

      const found = await detector.detect(rgb, w, h);
      const t1 = performance.now();

      const crops = [];
      for (const raw of found) {
        const truncated = { x: Math.trunc(raw.x), y: Math.trunc(raw.y), w: Math.trunc(raw.w), h: Math.trunc(raw.h) };
        const box = clampBox(truncated, w, h);
        const crop = box && cropSquareGray(rgb, w, h, box);
        if (box && crop) crops.push({ box, crop });
      }

      const faces = [];
      let classifyMs = 0;
      let gradcamMs = 0;
      if (crops.length > 0) {
        const n = crops.length;
        const probs = new Float32Array(n * 7);
        const features = new Float32Array(n * 512 * 49);
        const c0 = performance.now();
        for (let start = 0; start < n; start += BATCH) {
          const m = Math.min(BATCH, n - start);
          const inputs = new Float32Array(m * 3 * 224 * 224);
          for (let j = 0; j < m; j++) {
            const { crop } = crops[start + j];
            inputs.set(toInput(crop.gray, crop.w, crop.h), j * 3 * 224 * 224);
          }
          const out = await classifier.classify(inputs, m);
          probs.set(out.probs, start * 7);
          features.set(out.features, start * 512 * 49);
        }
        classifyMs = performance.now() - c0;

        const g0 = performance.now();
        crops.forEach(({ box, crop }, i) => {
          const p = probs.subarray(i * 7, i * 7 + 7);
          let best = 0;
          for (let k = 1; k < 7; k++) if (p[k] > p[best]) best = k;
          let cam = null;
          let heatmap = null;
          if (explain) {
            cam = gradcamPP(features.subarray(i * 512 * 49, (i + 1) * 512 * 49), fcWeight, best);
            heatmap = renderOverlay(crop.gray, crop.w, crop.h, cam);
          }
          faces.push({
            box,
            emotion: EMOTIONS[best],
            confidence: p[best],
            probabilities: Object.fromEntries(EMOTIONS.map((e, k) => [e, p[k]])),
            crop,
            cam,
            heatmap,
          });
        });
        gradcamMs = explain ? performance.now() - g0 : 0;
      }

      const t2 = performance.now();
      return {
        image: { width: w, height: h },
        faces,
        timings: { detect: t1 - t0, classify: classifyMs, gradcam: gradcamMs, total: t2 - t0 },
      };
    },
  };
}
