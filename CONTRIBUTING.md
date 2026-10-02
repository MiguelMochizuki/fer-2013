# Contributing

Setup and the daily commands are in the [README](README.md#development). CI runs the same checks as the pre-commit hooks (ruff, mypy, pytest, `npm test --prefix web`), so a green local run means a green PR.

## Documentation Conventions

Two languages, each with the style the code already used.

### Python

Google-style docstrings. Type hints carry the types, so the docstring never repeats them.

- Every module has a docstring.
- Every public function, class and method has at least a one-line summary in the imperative ("Write the golden files", not "Writes" or "This function writes").
- Add `Args:`, `Returns:` and `Raises:` when the contract is not obvious from the signature: units, shapes, valid ranges, what is returned, what is raised. A trivial helper needs only the summary.
- Command line scripts: `main(argv)` documents `argv` and the exit code.
- FastAPI endpoints and Pydantic models use a plain description instead of `Args:`: it is rendered as is in `/docs` and in the OpenAPI schema.
- Nested helpers and `_private` names need a docstring only when the logic is not obvious.

```python
def quantize_classifier(
    fp32_path: Path,
    out_path: Path,
    calibration_images: np.ndarray,
    n_calibration: int = 1000,
) -> Path:
    """Write the int8 model to `out_path` and return it.

    Args:
        fp32_path: the exported fp32 classifier.
        out_path: where to write the quantized model.
        calibration_images: (N, 48, 48) uint8 training faces to calibrate on.
        n_calibration: how many random faces to calibrate with.

    Returns:
        ``out_path``.
    """
```

### JavaScript

JSDoc blocks. The code is plain JavaScript, so types go in braces; a parameter line is `@param {type} name  description` (no dash).

- Every file starts with a `/** ... */` block saying what the module is for and anything a reader must know before touching it.
- Every exported function has a summary line and `@param` / `@returns` for what is not obvious. Destructured options are documented as one `options` object.
- Comments explain why, not what.

```js
/**
 * Create a YuNet detector bound to an ONNX Runtime Web session of the dynamic-shape model.
 * @param {{ Tensor: new (type: string, data: Float32Array, dims: number[]) => object }} ort
 * @param {{ run(feeds: object): Promise<Record<string, { data: Float32Array }>> }} session
 * @param {{ scoreThreshold?: number, nmsThreshold?: number, maxSide?: number, maxFaces?: number }} [options]
 */
export function createYuNetDetector(ort, session, options = {}) {
```

### Prose

Docs are in English, written for a reader who has two minutes. Numbers come from a run you can point to, with the date or the file; do not round in favor of the result.

### Test names

Describe the behavior: "a hash mismatch throws ModelIntegrityError and writes nothing to the cache".
