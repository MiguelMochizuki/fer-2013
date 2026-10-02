/**
 * Image resampling that reproduces Pillow's `Image.resize` numerically.
 *
 * The browser canvas does not scale like Pillow, and the classifier was fed
 * Pillow-resized images, so the algorithm is ported: separable convolution,
 * horizontal pass first, kernel support widened when shrinking, weights
 * normalized per output pixel. The 8-bit path uses Pillow's 22-bit fixed
 * point arithmetic with rounding after each pass, so results are identical
 * byte for byte; the float path accumulates in double and stores float32.
 */

const PRECISION_BITS = 32 - 8 - 2; // 22, as in Pillow's Resample.c

const sinc = (x) => (x === 0 ? 1 : Math.sin(x * Math.PI) / (x * Math.PI));

const FILTERS = {
  bilinear: { support: 1, fn: (x) => (Math.abs(x) < 1 ? 1 - Math.abs(x) : 0) },
  bicubic: {
    support: 2,
    fn: (x) => {
      const a = -0.5;
      x = Math.abs(x);
      if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
      if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
      return 0;
    },
  },
  lanczos: { support: 3, fn: (x) => (x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0) },
};

/**
 * Per output pixel: first input index, tap count and normalized weights.
 * @param {number} inSize
 * @param {number} outSize
 * @param {"bilinear" | "bicubic" | "lanczos"} filter
 * @returns {{ xmin: Int32Array, count: Int32Array, ksize: number, weights: Float64Array }}
 */
export function computeCoefficients(inSize, outSize, filter) {
  const { support: baseSupport, fn } = FILTERS[filter];
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = baseSupport * filterScale;
  const ksize = Math.ceil(support) * 2 + 1;
  const xmin = new Int32Array(outSize);
  const count = new Int32Array(outSize);
  const weights = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const hi = Math.min(Math.trunc(center + support + 0.5), inSize);
    const n = hi - lo;
    let total = 0;
    for (let x = 0; x < n; x++) {
      const w = fn((x + lo - center + 0.5) / filterScale);
      weights[xx * ksize + x] = w;
      total += w;
    }
    if (total !== 0) for (let x = 0; x < n; x++) weights[xx * ksize + x] /= total;
    xmin[xx] = lo;
    count[xx] = n;
  }
  return { xmin, count, ksize, weights };
}

/** Pillow's `normalize_coeffs_8bpc`: weights to signed 22-bit fixed point. */
function toFixedPoint(weights) {
  const fixed = new Int32Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    fixed[i] = Math.trunc(w < 0 ? -0.5 + w * (1 << PRECISION_BITS) : 0.5 + w * (1 << PRECISION_BITS));
  }
  return fixed;
}

const clip8 = (acc) => {
  const v = acc >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
};

/**
 * Resize interleaved 8-bit pixels exactly like Pillow.
 * @param {Uint8Array | Uint8ClampedArray} src  row-major, `channels` interleaved bytes per pixel
 * @param {number} w
 * @param {number} h
 * @param {number} channels
 * @param {number} outW
 * @param {number} outH
 * @param {"bilinear" | "bicubic" | "lanczos"} filter
 * @returns {Uint8Array}
 */
export function resizeU8(src, w, h, channels, outW, outH, filter) {
  let cur = Uint8Array.from(src);
  let curW = w;
  if (outW !== w) {
    const { xmin, count, ksize, weights } = computeCoefficients(w, outW, filter);
    const fixed = toFixedPoint(weights);
    const out = new Uint8Array(outW * h * channels);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < outW; x++) {
        for (let c = 0; c < channels; c++) {
          let acc = 1 << (PRECISION_BITS - 1);
          const base = (y * w + xmin[x]) * channels + c;
          for (let k = 0; k < count[x]; k++) acc += cur[base + k * channels] * fixed[x * ksize + k];
          out[(y * outW + x) * channels + c] = clip8(acc);
        }
      }
    }
    cur = out;
    curW = outW;
  }
  if (outH !== h) {
    const { xmin, count, ksize, weights } = computeCoefficients(h, outH, filter);
    const fixed = toFixedPoint(weights);
    const out = new Uint8Array(curW * outH * channels);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < curW; x++) {
        for (let c = 0; c < channels; c++) {
          let acc = 1 << (PRECISION_BITS - 1);
          for (let k = 0; k < count[y]; k++) {
            acc += cur[((xmin[y] + k) * curW + x) * channels + c] * fixed[y * ksize + k];
          }
          out[(y * curW + x) * channels + c] = clip8(acc);
        }
      }
    }
    cur = out;
  }
  return cur;
}

/**
 * Resize a single-channel float image like Pillow's mode "F".
 * @param {Float32Array} src
 * @param {number} w
 * @param {number} h
 * @param {number} outW
 * @param {number} outH
 * @param {"bilinear" | "bicubic" | "lanczos"} filter
 * @returns {Float32Array}
 */
export function resizeF32(src, w, h, outW, outH, filter) {
  let cur = Float32Array.from(src);
  let curW = w;
  if (outW !== w) {
    const { xmin, count, ksize, weights } = computeCoefficients(w, outW, filter);
    const out = new Float32Array(outW * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < outW; x++) {
        let acc = 0;
        for (let k = 0; k < count[x]; k++) acc += cur[y * w + xmin[x] + k] * weights[x * ksize + k];
        out[y * outW + x] = acc;
      }
    }
    cur = out;
    curW = outW;
  }
  if (outH !== h) {
    const { xmin, count, ksize, weights } = computeCoefficients(h, outH, filter);
    const out = new Float32Array(curW * outH);
    for (let y = 0; y < outH; y++) {
      for (let x = 0; x < curW; x++) {
        let acc = 0;
        for (let k = 0; k < count[y]; k++) acc += cur[(xmin[y] + k) * curW + x] * weights[y * ksize + k];
        out[y * curW + x] = acc;
      }
    }
    cur = out;
  }
  return cur;
}

/**
 * RGB(A) to 8-bit gray with Pillow's ITU-R 601-2 integer weights; alpha is ignored,
 * like `Image.convert("RGB").convert("L")`.
 * @param {Uint8Array | Uint8ClampedArray} pixels
 * @param {number} w
 * @param {number} h
 * @param {3 | 4} stride  bytes per pixel in `pixels`
 * @returns {Uint8Array}
 */
export function rgbToGray(pixels, w, h, stride) {
  const out = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += stride) {
    out[i] = (pixels[p] * 19595 + pixels[p + 1] * 38470 + pixels[p + 2] * 7471 + 0x8000) >> 16;
  }
  return out;
}
