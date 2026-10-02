/**
 * Face crop and classifier input, mirroring `fer_2013.serving.preprocess`.
 *
 * The model saw 48x48 grayscale faces upsampled to 224x224 (bicubic) and
 * normalized with ImageNet statistics; every step here follows the Python
 * implementation, including its integer truncation and edge handling.
 */

import { resizeF32, resizeU8, rgbToGray } from "./resample.js";

export const INPUT_SIZE = 224;
export const FACE_SIZE = 48;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];

/** Slide a window of `side` into [0, limit]; if it cannot fit, start at 0. */
const fit = (start, side, limit) => (side >= limit ? 0 : Math.min(Math.max(start, 0), limit - side));

/**
 * Square crop around `box` (grown by `margin` on every side) as 8-bit gray.
 * Near a border the square slides inside the image; when it is larger than the
 * image the crop is cut to the image and is not square. Empty boxes and boxes
 * outside the image give null.
 * @param {Uint8Array | Uint8ClampedArray} rgb  w*h*stride bytes (RGB or RGBA, alpha ignored)
 * @param {number} w
 * @param {number} h
 * @param {{ x: number, y: number, w: number, h: number }} box  integer pixels
 * @param {number} [margin]
 * @param {3 | 4} [stride]
 * @returns {{ gray: Uint8Array, w: number, h: number } | null}
 */
export function cropSquareGray(rgb, w, h, box, margin = 0.1, stride = 3) {
  if (box.w <= 0 || box.h <= 0) return null;
  if (box.x + box.w <= 0 || box.x >= w) return null;
  if (box.y + box.h <= 0 || box.y >= h) return null;
  const side = Math.trunc(Math.max(box.w, box.h) * (1 + 2 * margin));
  const x0 = fit(Math.trunc(box.x + box.w / 2 - side / 2), side, w);
  const y0 = fit(Math.trunc(box.y + box.h / 2 - side / 2), side, h);
  const cw = Math.min(x0 + side, w) - x0;
  const ch = Math.min(y0 + side, h) - y0;
  const pixels = new Uint8Array(cw * ch * stride);
  for (let row = 0; row < ch; row++) {
    const from = ((y0 + row) * w + x0) * stride;
    pixels.set(rgb.subarray(from, from + cw * stride), row * cw * stride);
  }
  return { gray: rgbToGray(pixels, cw, ch, stride), w: cw, h: ch };
}

/**
 * Classifier input for one face: 48x48 gray, bicubic to 224x224, replicated to
 * three channels and normalized (CHW, float32).
 * @param {Uint8Array} gray  w*h bytes
 * @param {number} w
 * @param {number} h
 * @returns {Float32Array} length 3 * 224 * 224
 */
export function toInput(gray, w, h) {
  const face = w === FACE_SIZE && h === FACE_SIZE ? gray : resizeU8(gray, w, h, 1, FACE_SIZE, FACE_SIZE, "lanczos");
  const unit = Float32Array.from(face, (v) => v / 255);
  const plane = resizeF32(unit, FACE_SIZE, FACE_SIZE, INPUT_SIZE, INPUT_SIZE, "bicubic");
  const out = new Float32Array(3 * plane.length);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < plane.length; i++) out[c * plane.length + i] = (plane[i] - MEAN[c]) / STD[c];
  }
  return out;
}
