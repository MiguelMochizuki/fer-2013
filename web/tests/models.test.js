import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { loadModels, ModelDownloadError, ModelIntegrityError } from "../src/models.js";

const MANIFEST_URL = "https://site.test/fer-2013/manifest.json";
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const blobs = {
  classifier: Uint8Array.from({ length: 3000 }, (_, i) => i % 251),
  detector: Uint8Array.from({ length: 700 }, (_, i) => (i * 7) % 256),
  fcWeight: Uint8Array.from({ length: 120 }, (_, i) => (i * 3) % 256),
};
const paths = {
  classifier: "models/fer_resnet18.aaaaaaaa.onnx",
  detector: "models/yunet.bbbbbbbb.onnx",
  fcWeight: "models/fc.cccccccc.npy",
};
const manifest = {
  models: Object.fromEntries(
    Object.entries(blobs).map(([k, v]) => [k, { path: paths[k], sha256: sha(v), size: v.length }]),
  ),
};
const byUrl = Object.fromEntries(Object.entries(paths).map(([k, p]) => [new URL(p, MANIFEST_URL).href, blobs[k]]));

/** A body delivered in `parts` chunks; `cutAfter` makes the stream fail after that many chunks. */
function body(bytes, { parts = 3, cutAfter = Infinity } = {}) {
  const size = Math.ceil(bytes.length / parts);
  let sent = 0;
  return new ReadableStream({
    pull(controller) {
      if (sent >= cutAfter) return controller.error(new Error("network cut"));
      const start = sent * size;
      if (start >= bytes.length) return controller.close();
      controller.enqueue(bytes.slice(start, start + size));
      sent++;
    },
  });
}
function fakeFetch({ tamper = null, cutOn = null, calls = [] } = {}) {
  const fn = async (url) => {
    calls.push(url);
    if (url === MANIFEST_URL) return new Response(JSON.stringify(manifest));
    let bytes = byUrl[url];
    if (tamper && url === tamper) bytes = Uint8Array.from(bytes, (v, i) => (i === 5 ? v ^ 1 : v));
    return new Response(body(bytes, url === cutOn ? { cutAfter: 1 } : {}));
  };
  fn.calls = calls;
  return fn;
}
function memoryCache() {
  const store = new Map();
  return {
    store,
    async match(url) {
      return store.has(url) ? new Response(store.get(url).slice()) : undefined;
    },
    async put(url, response) {
      store.set(url, new Uint8Array(await response.arrayBuffer()));
    },
  };
}
const eq = (buf, bytes) => assert.deepEqual(new Uint8Array(buf), bytes);

test("baixa, confere o sha256, guarda no cache e reporta progresso crescente até o total", async () => {
  const cache = memoryCache();
  const seen = [];
  const out = await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch(), cache, onProgress: (l, t) => seen.push([l, t]) });
  eq(out.classifier, blobs.classifier);
  eq(out.detector, blobs.detector);
  eq(out.fcWeight, blobs.fcWeight);
  assert.equal(cache.store.size, 3);
  const total = 3000 + 700 + 120;
  assert.ok(seen.every(([, t]) => t === total));
  assert.ok(seen.every(([l], i) => i === 0 || l >= seen[i - 1][0]), "progresso monotônico");
  assert.equal(seen.at(-1)[0], total);
});

test("hash divergente lança ModelIntegrityError e não grava no cache", async () => {
  const cache = memoryCache();
  const bad = new URL(paths.detector, MANIFEST_URL).href;
  await assert.rejects(loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch({ tamper: bad }), cache }), ModelIntegrityError);
  assert.equal(cache.store.has(bad), false);
});

test("rede cortada no meio lança ModelDownloadError e permite nova tentativa", async () => {
  const cache = memoryCache();
  const cut = new URL(paths.classifier, MANIFEST_URL).href;
  await assert.rejects(loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch({ cutOn: cut }), cache }), ModelDownloadError);
  assert.equal(cache.store.has(cut), false);
  const out = await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch(), cache });
  eq(out.classifier, blobs.classifier);
});

test("resposta HTTP de erro lança ModelDownloadError", async () => {
  const fetchFn = async (url) => (url === MANIFEST_URL ? new Response(JSON.stringify(manifest)) : new Response("nope", { status: 404 }));
  await assert.rejects(loadModels({ manifestUrl: MANIFEST_URL, fetchFn }), ModelDownloadError);
});

test("acerto no cache não chama fetch para o modelo", async () => {
  const cache = memoryCache();
  await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch(), cache });
  const second = fakeFetch();
  const seen = [];
  const out = await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: second, cache, onProgress: (l, t) => seen.push([l, t]) });
  assert.deepEqual(second.calls, [MANIFEST_URL]);
  eq(out.detector, blobs.detector);
  assert.equal(seen.at(-1)[0], seen.at(-1)[1]);
});

test("entrada de cache corrompida é ignorada e baixada de novo", async () => {
  const cache = memoryCache();
  await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch(), cache });
  const key = new URL(paths.fcWeight, MANIFEST_URL).href;
  cache.store.set(key, new Uint8Array(120).fill(9));
  const f = fakeFetch();
  const out = await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: f, cache });
  eq(out.fcWeight, blobs.fcWeight);
  assert.ok(f.calls.includes(key));
});

test("sem cache disponível (null) ainda funciona", async () => {
  const out = await loadModels({ manifestUrl: MANIFEST_URL, fetchFn: fakeFetch(), cache: null });
  eq(out.classifier, blobs.classifier);
});
