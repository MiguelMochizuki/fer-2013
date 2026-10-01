#!/bin/sh
# Download the three serving models and verify them against pinned sha256 sums.
# Usage: RELEASE_URL=<release download url> fetch_models.sh OUT_DIR SHA_FILE
set -eu

OUT=${1:?usage: fetch_models.sh OUT_DIR SHA_FILE}
SHA_FILE=${2:?usage: fetch_models.sh OUT_DIR SHA_FILE}
RELEASE_URL=${RELEASE_URL:?RELEASE_URL is required}
YUNET_URL=https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Fail fast if a model is missing from the checksum file.
for f in fer_resnet18.onnx fer_fc_weight.npy face_detection_yunet_2023mar.onnx; do
  grep -q "  $f\$" "$SHA_FILE" || { echo "no checksum for $f in $SHA_FILE" >&2; exit 1; }
done

mkdir -p "$OUT"
curl -fsSL "$RELEASE_URL/fer_resnet18.onnx" -o "$OUT/fer_resnet18.onnx"
curl -fsSL "$RELEASE_URL/fer_fc_weight.npy" -o "$OUT/fer_fc_weight.npy"
curl -fsSL "$YUNET_URL" -o "$OUT/face_detection_yunet_2023mar.onnx"
(cd "$OUT" && sha256sum -c "$(realpath "$SHA_FILE")")
