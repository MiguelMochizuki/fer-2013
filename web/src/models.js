/**
 * Loads the three model files named by `manifest.json`: sequential streaming
 * downloads with byte progress, sha256 verification, and a Cache Storage layer.
 * A file is cached only after its hash matches; a cached copy that no longer
 * matches is ignored and downloaded again.
 */

const NAMES = ["classifier", "detector", "fcWeight"];

export class ModelIntegrityError extends Error {
  constructor(name, expected, actual) {
    super(`${name}: sha256 mismatch (expected ${expected.slice(0, 12)}..., got ${actual.slice(0, 12)}...)`);
    this.name = "ModelIntegrityError";
  }
}

export class ModelDownloadError extends Error {
  constructor(name, cause) {
    super(`${name}: download failed (${cause instanceof Error ? cause.message : cause})`);
    this.name = "ModelDownloadError";
    this.cause = cause;
  }
}

const hex = (buffer) => [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");

async function sha256(subtle, bytes) {
  return hex(await subtle.digest("SHA-256", bytes));
}

/** Read a response body fully, reporting each chunk's size. */
async function readBody(response, onChunk) {
  const chunks = [];
  let length = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    length += value.length;
    onChunk(value.length);
  }
  const out = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * @param {object} options
 * @param {string} options.manifestUrl  absolute URL of manifest.json
 * @param {typeof fetch} [options.fetchFn]
 * @param {{ match(url: string): Promise<Response | undefined>, put(url: string, response: Response): Promise<void> } | null} [options.cache]
 * @param {(loaded: number, total: number) => void} [options.onProgress]
 * @param {SubtleCrypto} [options.subtle]
 * @returns {Promise<{ classifier: ArrayBuffer, detector: ArrayBuffer, fcWeight: ArrayBuffer }>}
 */
export async function loadModels({
  manifestUrl,
  fetchFn = (...args) => fetch(...args),
  cache = null,
  onProgress = () => {},
  subtle = globalThis.crypto.subtle,
}) {
  let manifest;
  try {
    const res = await fetchFn(manifestUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    manifest = await res.json();
  } catch (error) {
    throw new ModelDownloadError("manifest", error);
  }

  const total = NAMES.reduce((sum, name) => sum + manifest.models[name].size, 0);
  let loaded = 0;
  const advance = (n) => {
    loaded += n;
    onProgress(loaded, total);
  };
  onProgress(0, total);

  const result = {};
  for (const name of NAMES) {
    const { path, sha256: want, size } = manifest.models[name];
    const url = new URL(path, manifestUrl).href;

    let bytes = null;
    const hit = cache ? await cache.match(url) : undefined;
    if (hit) {
      const cached = new Uint8Array(await hit.arrayBuffer());
      if ((await sha256(subtle, cached)) === want) {
        bytes = cached;
        advance(size);
      }
    }
    if (!bytes) {
      try {
        const res = await fetchFn(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        bytes = await readBody(res, advance);
      } catch (error) {
        throw new ModelDownloadError(name, error);
      }
      const got = await sha256(subtle, bytes);
      if (got !== want) throw new ModelIntegrityError(name, want, got);
      if (cache) await cache.put(url, new Response(bytes));
    }
    result[name] = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  }
  return result;
}
