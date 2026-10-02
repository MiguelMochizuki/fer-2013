/**
 * ONNX classifier wrapper. The exported model has three outputs: raw `logits`,
 * the `layer4` activations (`features`, used for Grad-CAM++) and `probs`, the
 * softmax of the logits divided by the calibration temperature.
 */

export const EMOTIONS = Object.freeze(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]);

const INPUT_LEN = 3 * 224 * 224;

/**
 * @param {{ Tensor: new (type: string, data: Float32Array, dims: number[]) => object }} ort
 * @param {{ run(feeds: object): Promise<Record<string, { data: Float32Array }>> }} session
 */
export function createClassifier(ort, session) {
  return {
    /**
     * Classify `n` faces in one call.
     * @param {Float32Array} inputs  n * 3 * 224 * 224, CHW, normalized (see `toInput`)
     * @param {number} n
     * @returns {Promise<{ probs: Float32Array, features: Float32Array }>}
     *   probs: n * 7 (calibrated); features: n * 512 * 49
     */
    async classify(inputs, n) {
      if (inputs.length !== n * INPUT_LEN) throw new Error(`expected ${n * INPUT_LEN} values, got ${inputs.length}`);
      const out = await session.run({ input: new ort.Tensor("float32", inputs, [n, 3, 224, 224]) });
      return { probs: Float32Array.from(out.probs.data), features: Float32Array.from(out.features.data) };
    },
  };
}
