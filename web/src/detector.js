/**
 * YuNet face detector for the browser.
 *
 * OpenCV decodes the network's 12 raw heads internally, so the decoding and the
 * NMS are ported here. The dynamic-shape model (2026may) runs at the image's own
 * size, zero-padded down and right to a multiple of 32, which is what OpenCV
 * does; results match `cv2.FaceDetectorYN` (IoU 1.000 in the spike).
 */

import { resizeU8 } from "./resample.js";

/** @typedef {{ x: number, y: number, w: number, h: number, score: number }} Box */

const STRIDES = [8, 16, 32];
const clip01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Python's `round`: halves go to the even neighbour. */
function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Turn the raw heads into candidate boxes (before NMS), in padded-input pixels.
 * @param {Record<string, Float32Array>} outputs  keys cls_8, obj_8, bbox_8, ... bbox_32
 * @param {number} paddedH  input height (multiple of 32)
 * @param {number} paddedW  input width (multiple of 32)
 * @param {number} scoreThreshold
 * @returns {Box[]}
 */
export function decodeYunet(outputs, paddedH, paddedW, scoreThreshold) {
  const boxes = [];
  for (const stride of STRIDES) {
    const cols = paddedW / stride;
    const rows = paddedH / stride;
    const cls = outputs[`cls_${stride}`];
    const obj = outputs[`obj_${stride}`];
    const bbox = outputs[`bbox_${stride}`];
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const i = row * cols + col;
        const score = Math.sqrt(clip01(cls[i]) * clip01(obj[i]));
        if (score < scoreThreshold) continue;
        const cx = (col + bbox[i * 4]) * stride;
        const cy = (row + bbox[i * 4 + 1]) * stride;
        const w = Math.exp(bbox[i * 4 + 2]) * stride;
        const h = Math.exp(bbox[i * 4 + 3]) * stride;
        boxes.push({ x: cx - w / 2, y: cy - h / 2, w, h, score });
      }
    }
  }
  return boxes;
}

function iou(a, b) {
  const w = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const h = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = w * h;
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-9);
}

/**
 * Greedy non-maximum suppression: highest score first, drop boxes with IoU above the threshold.
 * @param {Box[]} boxes
 * @param {number} iouThreshold
 * @returns {Box[]}
 */
export function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const box of sorted) {
    if (kept.every((k) => iou(k, box) <= iouThreshold)) kept.push(box);
  }
  return kept;
}

/**
 * @param {{ Tensor: new (type: string, data: Float32Array, dims: number[]) => object }} ort
 * @param {{ run(feeds: object): Promise<Record<string, { data: Float32Array }>> }} session
 * @param {{ scoreThreshold?: number, nmsThreshold?: number, maxSide?: number, maxFaces?: number }} [options]
 */
export function createYuNetDetector(ort, session, options = {}) {
  const { scoreThreshold = 0.6, nmsThreshold = 0.3, maxSide = 640, maxFaces = 10 } = options;
  return {
    /**
     * @param {Uint8Array} rgb  w*h*3 bytes, row-major RGB
     * @param {number} w
     * @param {number} h
     * @returns {Promise<Box[]>} boxes in pixels of the original image, best score first
     */
    async detect(rgb, w, h) {
      const scale = Math.min(1, maxSide / Math.max(w, h));
      let img = rgb;
      let iw = w;
      let ih = h;
      if (scale < 1) {
        iw = Math.max(1, pyRound(w * scale));
        ih = Math.max(1, pyRound(h * scale));
        img = resizeU8(rgb, w, h, 3, iw, ih, "bilinear");
      }
      const H = Math.ceil(ih / 32) * 32;
      const W = Math.ceil(iw / 32) * 32;
      const plane = H * W;
      const input = new Float32Array(3 * plane); // BGR, 0..255, zeros as padding
      for (let y = 0; y < ih; y++) {
        for (let x = 0; x < iw; x++) {
          const src = (y * iw + x) * 3;
          const dst = y * W + x;
          input[dst] = img[src + 2];
          input[plane + dst] = img[src + 1];
          input[2 * plane + dst] = img[src];
        }
      }
      const result = await session.run({ input: new ort.Tensor("float32", input, [1, 3, H, W]) });
      const heads = {};
      for (const stride of STRIDES) {
        for (const name of ["cls", "obj", "bbox"]) heads[`${name}_${stride}`] = result[`${name}_${stride}`].data;
      }
      return nms(decodeYunet(heads, H, W, scoreThreshold), nmsThreshold)
        .slice(0, maxFaces)
        .map((b) => ({ x: b.x / scale, y: b.y / scale, w: b.w / scale, h: b.h / scale, score: b.score }));
    },
  };
}
