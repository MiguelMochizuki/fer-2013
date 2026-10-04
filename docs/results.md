# Results

Latest model: `models-v1.3.0`, trained on the FER+ labels with the code at `b1f911d`, seed 42 and the recipe in `configs/default.yaml`. It ran all 30 epochs; validation loss was lowest at epoch 28 (0.4416), which is the checkpoint that was kept.

## Why FER+

FER+ is the same 35,887 images as FER-2013 with new labels: about 10 crowd workers voted on every image, instead of the original automated process. The two label sets disagree a lot. Among the images FER+ keeps, only 64.9% of the training labels and 64.6% of the test labels are the same as in FER-2013, and the gap is largest on the rare classes: of the 515 test images FER-2013 calls `fear`, FER+ agrees on 17%; of the 50 it calls `disgust`, 36%. A model graded against labels that two annotation processes disagree on a third of the time is graded mostly on the noise, so the headline numbers use FER+.

What is dropped and how it is used (`preprocess_ferplus`):

- Rows whose majority vote is `contempt`, `unknown` or not-a-face are removed, which leaves 28,224 training, 3,529 validation and 3,519 test images (of 28,709, 3,589 and 3,589).
- Training uses the vote fractions of the seven classes as soft targets with plain cross-entropy. Validation and test use the majority vote, as in the [FER+ paper](https://arxiv.org/abs/1608.01041), which reports 83.9% to 85.0% with a VGG13 on 64x64 images over eight classes (a different setup, so the numbers are only a sanity check).
- Not done: the paper's outlier-vote rejection (resetting emotions with fewer than two votes before normalizing) and an eighth `contempt` class.

## Test set

FER+ test labels (3,519 images):

| Metric   | Value  |
|----------|--------|
| Accuracy | 0.8588 |
| Macro-F1 | 0.7702 |

On the original FER-2013 test labels (3,589 images) the same model gets 0.6121 accuracy and 0.5351 macro-F1, against 0.7074 and 0.7090 for the previous model (`models-v1.2.0`), which was trained on those labels. These two comparisons are different tasks and neither says one model is better than the other. On the FER+ test labels the previous model scores 0.6300 and 0.5644.

Runs are reproducible on the same hardware: two 1-epoch runs with this seed gave identical losses and accuracies to every digit (this is what the seeding in `training/train.py` is for, and the check that found a late seed in an earlier version). A different GPU or driver can still change low-order digits. With 3,519 test images the standard error of the accuracy is about 0.6 points, and the final number comes from one seed.

## Per-class (FER+ test set)

| Emotion  | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| angry    | 0.890     | 0.791  | 0.838 | 359     |
| disgust  | 0.609     | 0.500  | 0.549 | 28      |
| fear     | 0.775     | 0.474  | 0.588 | 116     |
| happy    | 0.938     | 0.945  | 0.942 | 933     |
| sad      | 0.762     | 0.713  | 0.736 | 480     |
| surprise | 0.833     | 0.902  | 0.866 | 430     |
| neutral  | 0.845     | 0.901  | 0.872 | 1173    |

`happy` (F1 0.942), `neutral` (0.872) and `surprise` (0.866) are the strongest. `fear` (0.588) and `disgust` (0.549) are the weakest, and for the same reason: they are rare in training (807 and 236 images of 28,224) and the model under-predicts them (recall 0.47 and 0.50, precision 0.78 and 0.61). `fear` goes to `surprise` in 31% of cases and to `sad` in 11%; `disgust` goes to `angry` in 21% and to `neutral` in 14%; `sad` goes to `neutral` in 22%. `disgust` has only 28 test images, so its numbers move by several points with a handful of images.

![Confusion matrix](images/confusion_matrix.png)

![Precision-recall curves](images/pr_curves.png)

![Training curves](images/training_curves.png)

The model still overfits, less than before: at the kept epoch the validation loss is 0.4416 and it stays flat to epoch 30 (0.4437), while training loss falls to 0.613 and the final training accuracy is 0.921 against 0.851 on validation. Validation macro-F1 (0.765 at the end, 0.772 at best) is noisy because of `disgust`.

Regenerate this table and these plots for a new run with:

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/ferplus/best.pt --processed-dir data/processed_ferplus --split test
```

which writes `reports/test_metrics.json` and fresh plots to `reports/` (`--reports-dir` to choose another folder). `reports/` itself is not tracked; to update the images above, copy the new PNGs into `docs/images/` and commit them.

## Recipe: why no class weights

The first FER+ runs used what the FER-2013 model used: a weighted sampler, class weights in the loss and early stopping on macro-F1. They reached 72.3% accuracy (and 70.1% with the sampler off and the weights on), 12 to 15 points below the plain recipe, because the inverse-frequency weight of `disgust` is about 17 against about 0.5 for `happy`. With 236 noisy training images the model could not learn the class, only that predicting it is cheap: on the test set `disgust` had recall 0.86 and precision 0.09. Turning off both the sampler and the weights, and stopping on validation loss, gave 84.7% and then 85.9% with the seeding fix. The paper does not use class weights either, and reports poor `disgust` and `contempt` for the same reason. The recipe was chosen after looking at per-class test results of the two weighted runs, not only validation results, so the test numbers are not a clean hold-out for that one decision.

## Grad-CAM

Grad-CAM++ heatmaps over `model.layer4`, on the logit of the predicted class, for the 8 most confident correct predictions and the 8 most confident misclassifications on the FER+ test set.

![Grad-CAM, correctly classified](images/gradcam_correct.png)

The 8 most confident correct predictions are all `happy` at 1.00. The heat sits on the mouth, nose and cheeks, with little on the eyes: the model reads `happy` from the lower half of the face.

![Grad-CAM, confident misclassifications](images/gradcam_misclassified.png)

Five of the 8 confident mistakes are read as `neutral` (true `angry` or `sad`), and the others as `happy` and `sad`. Several involve a hand on the face or a strongly tilted head, and the heat covers the whole central face instead of a feature. That fits the confusion matrix: the model's errors are mostly toward `neutral`.

Regenerate with:

```bash
uv run python scripts/gradcam_report.py --checkpoint checkpoints/ferplus/best.pt --processed-dir data/processed_ferplus --split test
```

## Calibration

This model is under-confident, the opposite of the previous one: on the test set its mean confidence is 0.768 against a 0.859 accuracy (soft-label training is a likely cause). Temperature scaling divides the logits by one scalar before the softmax, which changes the confidences but never the predicted class, so accuracy is untouched. The temperature is fitted on the validation split only (T = 0.674, 3,529 images, so it sharpens) and evaluated on the test set.

| Split | Probabilities | Accuracy | ECE | NLL | Brier | Mean confidence |
|-------|---------------|----------|-----|-----|-------|-----------------|
| val | softmax | 0.852 | 0.086 | 0.442 | 0.228 | 0.767 |
| val | T = 0.674 | 0.852 | 0.020 | 0.407 | 0.214 | 0.841 |
| test | softmax | 0.859 | 0.092 | 0.436 | 0.225 | 0.768 |
| test | T = 0.674 | 0.859 | 0.021 | 0.398 | 0.210 | 0.844 |

ECE is the expected calibration error of the top-label confidence over 15 bins; NLL and Brier are proper scoring rules, lower is better. After calibration the confidence tracks accuracy: in the shipped int8 model's test predictions, accuracy is 0.981 when confidence is at least 0.9 (1,863 images), 0.813 between 0.7 and 0.9 (880), 0.647 between 0.5 and 0.7 (614) and 0.438 below 0.5 (162).

![Reliability diagram](images/reliability_diagram.png)

Limits: a single temperature cannot fix everything, and the calibration holds for data like FER+'s validation split, not necessarily for photos from another population. The temperature is fitted on the fp32 model's validation logits and used as is for the int8 model.

Regenerate with:

```bash
uv run python scripts/calibrate.py --checkpoint checkpoints/ferplus/best.pt --processed-dir data/processed_ferplus
```

## Quantization

The shipped classifier is quantized to int8 after export: static QDQ quantization with per-channel weights, MinMax calibration on 1,000 random training faces, 7-bit range, and the softmax kept in float so `probs` stay exact. The graph, its three outputs and the temperature metadata are unchanged, so nothing downstream knows the difference except the size and the speed. `scripts/quantize_onnx.py` refuses to write a model that agrees with the fp32 one on fewer than 95% of 1,000 validation faces or loses more than a point of accuracy.

| Model | Test accuracy | Macro-F1 | ECE (stored T) | Size | One image (2 threads) |
|-------|---------------|----------|----------------|------|-----------------------|
| PyTorch checkpoint | 0.8588 | 0.7702 | 0.0209 | n/a | 23.4 ms |
| fp32 ONNX | 0.8588 | 0.7702 | 0.0209 | 44.7 MB | 15.6 ms |
| int8 ONNX | 0.8556 | 0.7690 | 0.0146 | 11.3 MB | 5.3 ms |

int8 loses 0.3 points of accuracy (about half the standard error of the test set), and 98.0% of its predictions equal the fp32 ones. Latency is the p50 of 200 runs of one 224x224 image on 2 threads, on the same machine as the API benchmark. The fp32 model is still published in the model release for tests and comparison. The Grad-CAM++ maps of the int8 model correlate 0.9991 on average with the fp32 maps over the 3,519 test faces (worst face 0.9875), each on the class fp32 predicts.

**Why the 7-bit range matters.** With the default 8-bit range the same model looked just as good on the development machine (one with AVX-VNNI) and was badly wrong on a GitHub runner: AVX2 CPUs without VNNI saturate an int16 intermediate in ONNX Runtime's uint8 by int8 kernels, and the runner's probabilities were off by 0.26 and its feature maps by 55% (relative L2). That was measured on the first model (`models-v1.2.0`), and the quantization code is the same. Quantizing with `reduce_range` gives results that are bit-identical on both CPUs. int8 kernels still round a few activations differently in ONNX Runtime Web than in native ONNX Runtime, so the browser's probabilities differ from Python's slightly (the golden tests allow up to 1e-2).

## The previous model

`models-v1.2.0` was trained on the original FER-2013 labels with a weighted sampler and loss and early stopping on macro-F1 (`configs/fer2013.yaml`), at `a672782`, before seeding was fixed, so it cannot be reproduced bit for bit. On the FER-2013 test labels (3,589 images) it scores 0.7074 accuracy and 0.7090 macro-F1, int8 0.7055 and 0.7062, with a temperature of 2.40 (it was over-confident: ECE 0.168 before, 0.022 after). The tables of that version are in the git history of this file.
