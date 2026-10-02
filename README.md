# FER-2013

A ResNet18 that reads seven facial expressions, trained, calibrated, shrunk to 11 MB and running **entirely in your browser**, live from your camera, with a heatmap of where it looked.

**[Live demo](https://miguelmochizuki.github.io/fer-2013/)** · [API](#api) · [Results](#results) · [Limitations](#limitations-and-privacy)

![The demo: a photo with the detected face, the emotion typeset by probability, the crop and the Grad-CAM++ heatmap](docs/images/demo.png)

## Highlights

| | |
|---|---|
| **Nothing leaves the tab** | Face detection, classification and Grad-CAM++ run in WebAssembly. Photo or live camera, every face at once. First visit downloads 26 MB. |
| **Accurate and honest** | 70.6% accuracy, 0.706 macro-F1 on the FER-2013 test set. Confidences are calibrated (ECE 0.168 to 0.022): when it says 90% or more, it is right 94% of the time. |
| **Small and fast** | int8 quantization took the model from 44.7 MB to 11.3 MB and made it 3x faster for a quarter of a point of accuracy. |
| **Torch-free API** | FastAPI on ONNX Runtime in a 365 MB distroless image (it was 742 MB): 10 ms per request, 94 MiB of memory. |
| **Verified, not assumed** | The JavaScript port is tested against golden files from the Python pipeline: resizing is byte-identical to Pillow, detector boxes match OpenCV (IoU 0.999). |

## Try it

```bash
# in the browser: https://miguelmochizuki.github.io/fer-2013/

# the API
docker build -t fer-api . && docker run --rm -p 7860:7860 fer-api
curl -F "file=@photo.jpg" "http://localhost:7860/predict?explain=true"

# train it yourself (Python 3.12, uv)
uv sync
```

## How it works

```
photo or camera frame
  -> YuNet face detector (230 KB, ONNX)
  -> square crop with 10% margin, grayscale, 224x224
  -> ResNet18, int8 (ONNX): logits, layer4 features, calibrated probabilities
  -> Grad-CAM++ from the features and the final layer's weights (closed form, no autograd)
```

The same four steps run in Python (`src/fer_2013/serving`) and in JavaScript (`web/src`). Training:

```bash
uv run python scripts/download_data.py --csv-path data/raw/                 # needs Kaggle credentials in .env
uv run python scripts/preprocess_data.py --csv-path data/raw/fer2013.csv --out-dir data/processed/
uv run python scripts/train.py --config configs/default.yaml                # weighted sampler + loss, early stopping on macro-F1
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
uv run python scripts/gradcam_report.py --checkpoint checkpoints/best.pt --split test
uv run python scripts/calibrate.py --checkpoint checkpoints/best.pt         # temperature scaling on the validation split
uv run python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out-dir models/ --calibration reports/calibration.json
mv models/fer_resnet18.onnx models/fer_resnet18_fp32.onnx
uv run python scripts/quantize_onnx.py --model models/fer_resnet18_fp32.onnx --out models/fer_resnet18.onnx
```

Override any config value with `--set section.field=value` (see `configs/default.yaml`).

## Results

| | Test accuracy | Macro-F1 | ECE | Size | One image |
|---|---|---|---|---|---|
| fp32 | 0.708 | 0.710 | 0.023 | 44.7 MB | 17.2 ms |
| **int8 (shipped)** | **0.706** | **0.706** | **0.021** | **11.3 MB** | **5.8 ms** |

`happy` (F1 0.89) and `surprise` (0.81) are read well; `sad` (0.55) and `fear` (0.59) are the weak spots, mostly confused with `neutral` and `angry`. Temperature scaling (T = 2.40, fitted on validation only) cut the expected calibration error from 0.168 to 0.022 without changing a single prediction.

<p>
<img src="docs/images/confusion_matrix.png" width="49%" alt="Confusion matrix on the test set">
<img src="docs/images/reliability_diagram.png" width="49%" alt="Reliability diagram after calibration">
</p>

Per-class numbers, Grad-CAM++ examples, the overfitting analysis, calibration and quantization details: [docs/results.md](docs/results.md).

## Engineering notes

- **The int8 model that was wrong only on some CPUs.** The first quantization looked perfect on my machine and was off by 0.26 in probability on a GitHub runner: AVX2 CPUs without VNNI saturate an int16 intermediate. A golden-file test caught it; a 7-bit range makes the results bit-identical on both. [Details](docs/results.md#quantization).
- **OpenCV out of the image.** YuNet's decoding and NMS are a few dozen lines of numpy, tested against `cv2.FaceDetectorYN`. The same algorithm powers the browser detector.
- **Closed-form Grad-CAM++.** The head is `avgpool -> dropout -> linear`, so the gradients are constant over space and the heatmap needs only the activations and the weights. [Why it works](docs/api.md#grad-cam-without-pytorch).
- **Live camera.** A tracker keeps face numbers between frames and smooths probabilities with an exponential average, because the classifier moves a few points with a few pixels of crop. [Browser demo](docs/browser-demo.md).
- **Free, permanent hosting.** The demo is a static site on GitHub Pages; there is no server to pay for or to keep alive.

## API

```bash
curl -F "file=@photo.jpg" "http://localhost:7860/predict?explain=true"
```

One result per detected face: box, emotion, calibrated probabilities and, with `explain=true`, a base64 Grad-CAM++ PNG. `GET /` is a demo page, `GET /docs` the OpenAPI UI, `GET /health` the loaded model hash. Run it without Docker, response format, latency tables: [docs/api.md](docs/api.md).

## Limitations and privacy

- About 71% accuracy on an acted, noisy dataset. Emotion labels read off a face are not a reliable read of how someone feels: do not use this for decisions about people.
- The model was trained on 48x48 grayscale faces. A webcam in poor light or at an angle is read worse than the test set suggests, and `neutral` tends to win.
- Confidences move a few points with a few pixels of crop. Small faces in a large crowd photo can be missed (detection runs at up to 640 px).
- Uploads are processed in memory and never written to disk or logged; the browser demo sends nothing anywhere. The page is under a strict CSP; the worker that touches the pixels is same-origin code with no network calls, but GitHub Pages cannot send CSP headers for workers, so the browser does not enforce that part.

More: [docs/browser-demo.md](docs/browser-demo.md#limitations-and-privacy).

## Project organization

```
src/fer_2013/
  data/ training/ evaluation/   download, train (Pydantic config, early stopping), metrics, Grad-CAM++, calibration
  models/                       ResNet18, ONNX export, int8 quantization (needs torch; local only)
  serving/                      torch-free runtime: preprocess, YuNet, classifier, Grad-CAM++, FastAPI
web/                            the static site: ES modules, worker, tests against Python golden files
scripts/                        thin CLIs over the package; fetch_models.sh, smoke_test.sh
tests/  web/tests/              pytest and node:test (the site is tested against Python golden files)
docs/                           results, API and browser-demo deep dives
serving/models.sha256           pinned hashes of every model the image and the site download
```

**Stack:** Python 3.12 and [uv](https://docs.astral.sh/uv/), PyTorch (training only), ONNX Runtime, FastAPI, OpenCV (tests only), Docker (distroless), onnxruntime-web, GitHub Actions and Pages.

## Releases

`v*` tags are the application; `models-v*` tags are the trained weights, published as **pre-releases on purpose**: the Dockerfile, CI and the site download from them, so they must stay. Model versions: major for a contract change, minor for a retrain or re-quantization, patch for a re-export.

## Development

```bash
uv run ruff check --fix . && uv run ruff format .   # lint and format
uv run mypy                                         # strict type check
uv run pytest -q                                    # python tests
npm test --prefix web                               # site tests
uv run pre-commit install                           # ruff + mypy on commit, tests on push
```

## License

[MIT](LICENSE) from v0.2.0 (v0.1.0 and v0.1.1 were published under the AGPL-3.0 and stay under it). The data is the [FER-2013 mirror on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) (CC0); third-party attributions are in [NOTICE](NOTICE).
