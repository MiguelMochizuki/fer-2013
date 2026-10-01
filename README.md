# FER-2013

Facial expression recognition on the [FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset: a ResNet18 fine-tuned on 48x48 grayscale faces to classify one of seven emotions, with a full pipeline from raw CSV to trained model to evaluation report.

## Emotions

`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

## Pipeline

1. **Download**: fetch `fer2013.csv` from Kaggle.
2. **Preprocess**: parse the CSV into `.npy` arrays, split into train/val/test.
3. **Train**: fine-tune an ImageNet-pretrained ResNet18 with a weighted sampler and weighted loss to counter class imbalance, early stopping on validation macro-F1.
4. **Evaluate**: run the best checkpoint on a split, produce metrics, a confusion matrix, and precision-recall curves.
5. **Explain**: run Grad-CAM++ on the same checkpoint to see which regions of the face drive correct and incorrect predictions.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

To download the dataset yourself, add Kaggle credentials to a `.env` file (see `.env.example`):

```
KAGGLE_USERNAME=your-username
KAGGLE_API_TOKEN=your-token
```

## Usage

```bash
# 1. Download the raw CSV
uv run python scripts/download_data.py --csv-path data/raw/

# 2. Preprocess into train/val/test arrays
uv run python scripts/preprocess_data.py --csv-path data/raw/fer2013.csv --out-dir data/processed/

# 3. Train
uv run python scripts/train.py --config configs/default.yaml

# Override any config value from the command line
uv run python scripts/train.py --config configs/default.yaml --set training.epochs=5 --set data.batch_size=128

# 4. Evaluate the best checkpoint on the test set
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
```

Training writes checkpoints to `checkpoints/`, TensorBoard logs to `runs/`, and a per-epoch history JSON to `reports/`. View training progress with:

```bash
uv run tensorboard --logdir runs
```

Evaluation writes metrics, prediction arrays, and plots (confusion matrix, PR curves, training curves) to `reports/`.

## Configuration

Training is driven by a YAML config (`configs/default.yaml`):

```yaml
data:
  processed_dir: data/processed
  batch_size: 64
  num_workers: 4

model:
  num_classes: 7
  pretrained: true

training:
  epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-4
  early_stopping_patience: 5
  early_stopping_metric: macro_f1
  early_stopping_mode: max
  seed: 42

checkpoint:
  dir: checkpoints
  save_best: true
  save_last: true

tensorboard:
  enabled: true
  log_dir: runs
  run_name: fer2013_resnet18
```

Any field can be overridden per run with `--set section.field=value` (see Usage above).

## Results

Latest run: commit `a672782`, trained 2026-09-27.

Training stopped early after 30 epochs (no early-stopping trigger hit); best validation macro-F1 of **0.6952** was reached at epoch 29.

### Test set

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.7122 |
| Macro-F1 | 0.7153 |

### Per-class (test set)

| Emotion  | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| angry    | 0.643     | 0.635  | 0.639 | 491     |
| disgust  | 0.843     | 0.782  | 0.811 | 55      |
| fear     | 0.639     | 0.547  | 0.590 | 528     |
| happy    | 0.900     | 0.870  | 0.885 | 879     |
| sad      | 0.549     | 0.572  | 0.561 | 594     |
| surprise | 0.835     | 0.841  | 0.838 | 416     |
| neutral  | 0.641     | 0.730  | 0.683 | 626     |

`fear` and `sad` are the weakest classes and are also the most frequently confused with each other; `happy` and `disgust` are the strongest, though `disgust` has by far the smallest support (55 test samples) so its F1 is noisier than the others.

![Confusion matrix](docs/images/confusion_matrix.png)

![Precision-recall curves](docs/images/pr_curves.png)

![Training curves](docs/images/training_curves.png)

Regenerate this table and these plots for a new run with:

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
```

which writes `reports/test_metrics.json` and fresh plots to `reports/`. `reports/` itself is not tracked; to update the images above, copy the new PNGs into `docs/images/` and commit them.

### Grad-CAM

Grad-CAM++ heatmaps over `model.layer4`, on the logit of the predicted class, for the 6 most confident correct predictions and the 6 most confident misclassifications on the test set.

![Grad-CAM, correctly classified](docs/images/gradcam_correct.png)

Correct predictions concentrate on the mouth and eyes, the regions that carry most of the expression.

![Grad-CAM, confident misclassifications](docs/images/gradcam_misclassified.png)

Several confident misclassifications latch onto glasses, hair, or image artifacts instead of the face, which is consistent with `fear`/`sad`/`angry` being the weakest and most-confused classes above.

Regenerate with:

```bash
uv run python scripts/gradcam_report.py --checkpoint checkpoints/best.pt --split test
```

## API

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

`GET /` is a small demo page, `GET /docs` the OpenAPI UI, `GET /health` reports the loaded model hash. Faces are found with [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT).

### Run it

```bash
# export the trained checkpoint (needs torch) and fetch the face detector
uv run python scripts/export_onnx.py --checkpoint checkpoints/best.pt --out-dir models/
ONLY_YUNET=1 scripts/fetch_models.sh models serving/models.sha256
MODELS_DIR=models uv run uvicorn fer_2013.serving.api:create_app --factory --port 7860
```

Or with Docker, which downloads the models from a GitHub Release and verifies their sha256 at build time:

```bash
docker build -t fer-api .
docker run --rm -p 7860:7860 fer-api
```

To publish a new model: run the export, `gh release create models-vN models/fer_resnet18.onnx models/fer_fc_weight.npy` (model releases use `models-v*` tags, app releases use `v*`), append their `sha256sum` lines to `serving/models.sha256`, and point `RELEASE_URL` in the `Dockerfile` at the new tag. CI tests, builds the image and smoke tests it; pushing a `v*` tag then deploys to a Hugging Face Space (Docker SDK, free CPU tier, which sleeps after 48 hours without traffic and wakes on the next visit) and creates a GitHub Release.

### Grad-CAM++ without PyTorch

The head is `avgpool -> Dropout -> Linear`, so in eval mode the gradient of a class logit with respect to the `layer4` activations is constant over space (`w_k / 49`). Plugging that into the Grad-CAM++ weights gives a closed form that needs only the activations and the `Linear` weights, both available from the ONNX model, so the API computes it in numpy. This holds only for this head. `tests/serving/test_gradcam.py` checks it against `pytorch-grad-cam` for every class (correlation above 0.999), and would fail if the architecture changed.

### Performance

Measured with `scripts/benchmark.py` on a 12th Gen Intel i5-12450HX against the Docker image (an untrained classifier of the same architecture, so timings are representative; a 260x260 image with one face, 50 requests):

| Request            | p50 (ms) | p95 (ms) |
|--------------------|----------|----------|
| `explain=false`    | 22.8     | 24.3     |
| `explain=true`     | 26.9     | 30.5     |

Classifier alone, one 224x224 image, 2 threads: ONNX Runtime 15.6 ms p50 versus PyTorch 24.1 ms. The Docker image is 742 MB on disk (330 MB of Python packages, 43 MB of models); an environment with PyTorch and its CUDA wheels is over 4 GB. Free-tier hardware is slower than this machine.

### Limitations and privacy

- The classifier reaches about 71% accuracy on FER-2013 (see Results) and inherits the dataset's biases: acted or web-scraped expressions, uneven demographics, noisy labels. Emotion labels from a face are not a reliable read of how someone feels. Do not use this for decisions about people.
- Uploaded images are processed in memory and never written to disk or logged. Uploads are limited to 5 MB and JPEG, PNG or WebP.
- The service is public and unauthenticated; it caps concurrent work and answers `503` when busy.

## Development

```bash
uv run ruff check --fix .   # lint
uv run ruff format .        # format
uv run mypy                 # type check
uv run pytest -q            # tests
```

Pre-commit hooks run lint and format on commit, and type checking plus the full test suite on push:

```bash
uv run pre-commit install
```

## License

Released under the [GNU AGPL-3.0](LICENSE). The training data is the [FER-2013 mirror on Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013) (CC0); see [NOTICE](NOTICE) for third-party attributions.
