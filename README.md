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
