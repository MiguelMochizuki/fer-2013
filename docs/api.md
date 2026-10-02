# API

A FastAPI service wraps the trained model: upload a photo, get one result per detected face (box, emotion, probabilities and, on request, a Grad-CAM++ heatmap). Inference runs on ONNX Runtime, so the image has no PyTorch in it.

```bash
curl -F "file=@photo.jpg" "http://localhost:7860/predict?explain=true"
```

```json
{
  "image": {"width": 640, "height": 480},
  "faces": [{
    "box": {"x": 120, "y": 80, "w": 96, "h": 96},
    "emotion": "happy",
    "confidence": 0.93,
    "probabilities": {"angry": 0.01, "disgust": 0.0, "fear": 0.01, "happy": 0.93, "sad": 0.01, "surprise": 0.02, "neutral": 0.02},
    "gradcam": "<base64 PNG, null unless explain=true>"
  }],
  "timings_ms": {"detect": 12.0, "classify": 35.0, "total": 52.0}
}
```

`probabilities` and `confidence` are calibrated (see [Calibration](results.md#calibration)). `GET /` is a small demo page, `GET /docs` the OpenAPI UI, `GET /health` reports the loaded model hash and its calibration temperature (`null` for models exported before calibration existed). Faces are found with [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT), the same model file the browser demo uses.

## Run it

```bash
 export the trained checkpoint (needs torch), quantize it, and fetch the face detector
uv run python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out-dir models/
mv models/fer_resnet18.onnx models/fer_resnet18_fp32.onnx
uv run python scripts/quantize_onnx.py --model models/fer_resnet18_fp32.onnx --out models/fer_resnet18.onnx
ONLY_YUNET=1 scripts/fetch_models.sh models serving/models.sha256
MODELS_DIR=models uv run uvicorn fer_2013.serving.api:create_app --factory --port 7860
```

Or with Docker, which downloads the models from a GitHub Release and verifies their sha256 at build time:

```bash
docker build -t fer-api .
docker run --rm -p 7860:7860 fer-api
```

To publish a new model: run `scripts/calibrate.py` and the export with `--calibration`, `gh release create models-vX.Y.Z models/fer_resnet18.onnx models/fer_fc_weight.npy` (model releases use `models-v*` tags so they don't trigger the app's `v*` deploy; bump major when the serving contract changes, such as architecture, input normalization or labels, minor for a retrain, patch for a re-export), append their `sha256sum` lines to `serving/models.sha256`, and point `RELEASE_URL` in the `Dockerfile` at the new tag. CI tests, builds the image and smoke tests it. The image runs anywhere Docker does; the public demo is the [browser build](browser-demo.md), which needs no server.

## Grad-CAM++ without PyTorch

The head is `avgpool -> Dropout -> Linear`, so in eval mode the gradient of a class logit with respect to the `layer4` activations is constant over space (`w_k / 49`). Plugging that into the Grad-CAM++ weights gives a closed form that needs only the activations and the `Linear` weights, both available from the ONNX model, so the API computes it in numpy. This holds only for this head, and it uses the raw logits and weights: dividing them by the calibration temperature would change the heatmap, so only the probabilities are calibrated. `tests/serving/test_gradcam.py` checks it against `pytorch-grad-cam` for every class (correlation above 0.999), and would fail if the architecture changed.

## Performance

The shipped int8 model scores 70.55% on the full test set through the ONNX Runtime pipeline with the serving preprocessing, against 70.80% for the fp32 ONNX model and 70.74% for PyTorch (see [Quantization](results.md#quantization)).

Latency measured with `scripts/benchmark.py` against the Docker image (a 260x260 image with one face; `p50` / `p95` of the full request). On a 12th Gen Intel i5-12450HX with no CPU limit, 60 requests:

| Request            | p50 (ms) | p95 (ms) |
|--------------------|----------|----------|
| `explain=false`    | 10.4     | 11.3     |
| `explain=true`     | 14.5     | 16.3     |

Restricted to the size of a typical free hosting tier (`docker run --cpus 0.1 --memory 512m`, 15 requests): about 1.3 s p50 without `explain` and 1.6 s with it, a boot of about 33 s, and 94 MiB of memory in use. It fits a 512 MB instance with room to spare. (With the fp32 model and OpenCV it was 1.8 to 2.0 s and 212 MiB.) The detector session is single-threaded on purpose: two ONNX Runtime threads busy-waiting on a tenth of a core made it three times slower.

Classifier alone, one 224x224 image, 2 threads: int8 5.8 ms p50, fp32 17.2 ms, PyTorch 24.1 ms. The Docker image is 365 MB on disk and 94 MB compressed (162 MB of Python packages, 11 MB of models, Python itself and a distroless base for the rest). The base is `gcr.io/distroless/cc-debian12:nonroot`: no shell, no package manager, running as a non-root user; the build copies in the Python runtime and only the shared libraries it needs. It has no OpenCV: YuNet runs on ONNX Runtime with the decoding and NMS in numpy (`serving/detector.py`), and a test checks its boxes against `cv2.FaceDetectorYN`. Together with the int8 model that took the image from 742 MB. An environment with PyTorch and its CUDA wheels is over 4 GB.
