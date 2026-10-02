# Results

Latest run: trained 2026-10-01 with the training code at `a672782` and the default seed (42), the same code and seed as the previous run, which this one replaces. It ran all 30 epochs without triggering early stopping; the best validation macro-F1 of **0.6842** was reached at epoch 28.

## Test set

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.7074 |
| Macro-F1 | 0.7090 |

The previous run with the same code and seed scored 0.7122 and 0.7153. Training only calls `torch.manual_seed` and does not enable deterministic CUDA kernels, so runs are not bit-reproducible, and with 3,589 test images the standard error of the accuracy is about 0.8 points. A gap of half a point between two single runs is within that noise; averaging several seeds would be needed to tell real changes from it.

## Per-class (test set)

| Emotion  | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| angry    | 0.626     | 0.646  | 0.636 | 491     |
| disgust  | 0.840     | 0.764  | 0.800 | 55      |
| fear     | 0.590     | 0.589  | 0.590 | 528     |
| happy    | 0.910     | 0.866  | 0.887 | 879     |
| sad      | 0.565     | 0.544  | 0.554 | 594     |
| surprise | 0.808     | 0.820  | 0.814 | 416     |
| neutral  | 0.657     | 0.709  | 0.682 | 626     |

`sad` (F1 0.554) and `fear` (0.590) are the weakest classes. `sad` is mostly mistaken for `neutral` (18% of its samples), `angry` (12%) and `fear` (11%); `fear` for `sad` (15%) and `angry` (11%). `happy` (0.887) and `surprise` (0.814) are the strongest. `disgust` scores 0.800 but has by far the smallest support (55 test samples), so its F1 is noisier than the others.

![Confusion matrix](images/confusion_matrix.png)

![Precision-recall curves](images/pr_curves.png)

![Training curves](images/training_curves.png)

The model overfits: validation loss is lowest at epoch 4 (1.03) and climbs to 1.48 while training loss falls to 0.05, and the final training accuracy is 0.959 against 0.690 on validation. Validation macro-F1 keeps creeping up and plateaus around 0.68 from epoch 22 on, which is why early stopping on that metric never fires. Stronger regularization or earlier stopping on validation loss are the obvious next experiments.

Regenerate this table and these plots for a new run with:

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test
```

which writes `reports/test_metrics.json` and fresh plots to `reports/`. `reports/` itself is not tracked; to update the images above, copy the new PNGs into `docs/images/` and commit them.

## Grad-CAM

Grad-CAM++ heatmaps over `model.layer4`, on the logit of the predicted class, for the 8 most confident correct predictions and the 8 most confident misclassifications on the test set.

![Grad-CAM, correctly classified](images/gradcam_correct.png)

Correct predictions concentrate on the mouth and nose, with the eyebrows and eyes also lit up for `angry` and `disgust`: the central face region that carries most of the expression.

![Grad-CAM, confident misclassifications](images/gradcam_misclassified.png)

The confident mistakes are mostly `fear` and `sad`/`angry` samples. Several involve an open mouth (`fear` read as `surprise`), glasses, a tilted or partly out-of-frame face, or a wink, where the heat lands on the glasses rim or on one eye instead of the whole expression. That is consistent with `fear`, `sad` and `angry` being the weakest and most-confused classes above.

Regenerate with:

```bash
uv run python scripts/gradcam_report.py --checkpoint checkpoints/best.pt --split test
```

## Calibration

The model is overconfident: on the test set 38.3% of its predictions come with a confidence above 0.99, and its mean confidence is 0.876 against a 0.707 accuracy. Temperature scaling divides the logits by one scalar before the softmax, which changes the confidences but never the predicted class, so accuracy is untouched. The temperature is fitted on the validation split only (T = 2.40, 3,589 images) and evaluated on the test set.

| Split | Probabilities | Accuracy | ECE | NLL | Brier | Mean confidence |
|-------|---------------|----------|-----|-----|-------|-----------------|
| val | softmax | 0.688 | 0.185 | 1.319 | 0.491 | 0.873 |
| val | T = 2.40 | 0.688 | 0.016 | 0.917 | 0.434 | 0.684 |
| test | softmax | 0.707 | 0.168 | 1.209 | 0.461 | 0.876 |
| test | T = 2.40 | 0.707 | 0.022 | 0.867 | 0.415 | 0.687 |

ECE is the expected calibration error of the top-label confidence over 15 bins; NLL and Brier are proper scoring rules, lower is better. After calibration the confidence tracks accuracy: in the exported model's test predictions, accuracy is 0.939 when confidence is at least 0.9 (725 images), 0.835 between 0.7 and 0.9 (1,095), 0.630 between 0.5 and 0.7 (939) and 0.427 below 0.5 (830).

![Reliability diagram](images/reliability_diagram.png)

Limits: a single temperature cannot fix everything. The middle of the curve ends up slightly under-confident, and training used balanced sampling, so per-class biases from that prior shift are not corrected. The calibration holds for data like FER-2013's validation split, not necessarily for photos from another population.

Regenerate with:

```bash
uv run python scripts/calibrate.py --checkpoint checkpoints/best.pt
```

## Quantization

The shipped classifier is quantized to int8 after export: static QDQ quantization with per-channel weights, MinMax calibration on 1,000 random training faces, 7-bit range, and the softmax kept in float so `probs` stay exact. The graph, its three outputs and the temperature metadata are unchanged, so nothing downstream knows the difference except the size and the speed. `scripts/quantize_onnx.py` refuses to write a model that agrees with the fp32 one on fewer than 95% of 1,000 validation faces or loses more than a point of accuracy.

| Model | Test accuracy | Macro-F1 | ECE (T refit on val) | Size | One image (2 threads) |
|-------|---------------|----------|----------------------|------|-----------------------|
| fp32 | 0.7080 | 0.7095 | 0.0234 (T = 2.395) | 44.7 MB | 17.2 ms |
| int8 | 0.7055 | 0.7062 | 0.0210 (T = 2.403) | 11.3 MB | 5.8 ms |

Accuracy, macro-F1 and calibration stay within noise (the standard error of the accuracy is about 0.8 points); 96.3% of predictions equal the fp32 ones, and the Grad-CAM++ maps correlate 0.9986 with the fp32 maps (worst face 0.992). The temperature refit on the int8 logits is within 0.01 of the stored one, so the T stored in the model (2.395) was kept. The fp32 model is still published in the model release for tests and comparison.

**Why the 7-bit range matters.** With the default 8-bit range the same model looked just as good on the development machine (one with AVX-VNNI) and was badly wrong on a GitHub runner: AVX2 CPUs without VNNI saturate an int16 intermediate in ONNX Runtime's uint8 by int8 kernels, and the runner's probabilities were off by 0.26 and its feature maps by 55% (relative L2). Quantizing with `reduce_range` costs about a quarter of a point of accuracy here and gives results that are bit-identical on both CPUs. int8 kernels still round a few activations differently in ONNX Runtime Web than in native ONNX Runtime, so the browser's probabilities differ from Python's by up to 7e-3.
