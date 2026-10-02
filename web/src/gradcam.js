/**
 * Grad-CAM++ in closed form for the ResNet18 head (avgpool -> Dropout -> Linear).
 *
 * In eval mode the gradient of a class logit with respect to the layer4
 * activations is constant over space (g_k = w_k / (H*W)), so the Grad-CAM++
 * weights need only the activations and the Linear weights. This uses the raw
 * features and weights: the formula is not invariant to their scale, so the
 * calibration temperature must never be folded in.
 */

import { resizeF32, resizeU8 } from "./resample.js";

const EPS = 1e-6; // pytorch-grad-cam's GradCAMPlusPlus epsilon
const NORM_EPS = 1e-7; // pytorch-grad-cam's scale_cam_image epsilon
const CHANNELS = 512;
const HW = 49; // 7 x 7

/**
 * @param {Float32Array} features  one face, CHANNELS * 49, channel-major (K, 7, 7)
 * @param {Float32Array} fcWeight  7 * CHANNELS, class-major
 * @param {number} classIdx
 * @returns {Float32Array} 49 values in [0, 1]
 */
export function gradcamPP(features, fcWeight, classIdx) {
  const weights = new Float64Array(CHANNELS);
  for (let k = 0; k < CHANNELS; k++) {
    const g = fcWeight[classIdx * CHANNELS + k] / HW;
    let s = 0;
    for (let i = 0; i < HW; i++) s += features[k * HW + i];
    const g2 = g * g;
    const g3 = g2 * g;
    const aij = g !== 0 ? g2 / (2 * g2 + s * g3 + EPS) : 0;
    weights[k] = HW * Math.max(g, 0) * aij;
  }
  const cam = new Float64Array(HW);
  for (let k = 0; k < CHANNELS; k++) {
    const w = weights[k];
    if (w === 0) continue;
    for (let i = 0; i < HW; i++) cam[i] += w * features[k * HW + i];
  }
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < HW; i++) {
    cam[i] = Math.max(cam[i], 0);
    if (cam[i] < min) min = cam[i];
    if (cam[i] > max) max = cam[i];
  }
  const out = new Float32Array(HW);
  const scale = NORM_EPS + (max - min);
  for (let i = 0; i < HW; i++) out[i] = (cam[i] - min) / scale;
  return out;
}

const clip01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Jet-like colormap, x in [0,1] to [r, g, b] in [0,1]. */
function jet(x) {
  return [
    clip01(1.5 - Math.abs(4 * x - 3)),
    clip01(1.5 - Math.abs(4 * x - 2)),
    clip01(1.5 - Math.abs(4 * x - 1)),
  ];
}

/**
 * Blend the heatmap over the gray face crop, like the API's overlay.
 * @param {Uint8Array} gray  w*h bytes
 * @param {number} w
 * @param {number} h
 * @param {Float32Array} cam  49 values from `gradcamPP`
 * @param {number} [outSize]
 * @returns {Uint8ClampedArray} RGBA, outSize * outSize * 4, alpha 255
 */
export function renderOverlay(gray, w, h, cam, outSize = 128) {
  const heat = resizeF32(cam, 7, 7, outSize, outSize, "bilinear");
  const base = resizeU8(gray, w, h, 1, outSize, outSize, "bicubic");
  const out = new Uint8ClampedArray(outSize * outSize * 4);
  for (let i = 0; i < outSize * outSize; i++) {
    const [r, g, b] = jet(clip01(heat[i]));
    const g0 = 0.5 * Math.fround(base[i] / 255);
    out[i * 4] = Math.trunc((g0 + 0.5 * r) * 255);
    out[i * 4 + 1] = Math.trunc((g0 + 0.5 * g) * 255);
    out[i * 4 + 2] = Math.trunc((g0 + 0.5 * b) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}
