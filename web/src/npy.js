/**
 * Minimal reader for NumPy .npy files holding little-endian float32 in C order.
 * Enough for the classifier head weights; anything else is rejected loudly.
 */

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

/**
 * @param {ArrayBuffer} buffer
 * @returns {{ shape: number[], data: Float32Array }}
 */
export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || MAGIC.some((b, i) => bytes[i] !== b)) {
    throw new Error("not an .npy file");
  }
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("ascii").decode(bytes.subarray(headerStart, headerStart + headerLen));

  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr === undefined || fortran === undefined || shapeText === undefined) {
    throw new Error("unreadable .npy header");
  }
  if (descr !== "<f4") throw new Error(`unsupported dtype ${descr}, expected <f4`);
  if (fortran === "True") throw new Error("fortran_order arrays are not supported");

  const shape = shapeText.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const start = headerStart + headerLen;
  if (bytes.length < start + count * 4) throw new Error("truncated .npy data");
  // slice() copies, which also guarantees the 4-byte alignment Float32Array needs
  return { shape, data: new Float32Array(buffer.slice(start, start + count * 4)) };
}
