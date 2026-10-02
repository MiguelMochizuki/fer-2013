import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";

const MODEL = new URL("../../models/fer_resnet18.onnx", import.meta.url);

test("the classifier runs on Node's WASM with 3 outputs", { skip: !existsSync(MODEL) }, async () => {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(await readFile(MODEL), {
    executionProviders: ["wasm"],
  });
  const input = new ort.Tensor("float32", new Float32Array(3 * 224 * 224), [1, 3, 224, 224]);
  const out = await session.run({ input });
  assert.deepEqual(out.logits.dims, [1, 7]);
  assert.deepEqual(out.features.dims, [1, 512, 7, 7]);
  assert.deepEqual(out.probs.dims, [1, 7]);
  const sum = out.probs.data.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-5, `probs sum ${sum}`);
});
