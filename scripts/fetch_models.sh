#!/bin/sh
# Download the serving models and verify them against pinned sha256 sums.
# Usage: RELEASE_URL=<release download url> fetch_models.sh OUT_DIR SHA_FILE
#        ONLY_YUNET=1 fetch_models.sh OUT_DIR SHA_FILE   (detector only, for CI)
set -eu

OUT=${1:?usage: fetch_models.sh OUT_DIR SHA_FILE}
SHA_FILE=${2:?usage: fetch_models.sh OUT_DIR SHA_FILE}
YUNET=face_detection_yunet_2023mar.onnx
YUNET_URL=https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/$YUNET

if [ -n "${ONLY_YUNET:-}" ]; then
  FILES=$YUNET
else
  RELEASE_URL=${RELEASE_URL:?RELEASE_URL is required}
  FILES="fer_resnet18.onnx fer_fc_weight.npy $YUNET"
fi

# Fail fast if a model is missing from the checksum file.
for f in $FILES; do
  grep -q "  $f\$" "$SHA_FILE" || { echo "no checksum for $f in $SHA_FILE" >&2; exit 1; }
done

mkdir -p "$OUT"
curl -fsSL "$YUNET_URL" -o "$OUT/$YUNET"
if [ -z "${ONLY_YUNET:-}" ]; then
  curl -fsSL "$RELEASE_URL/fer_resnet18.onnx" -o "$OUT/fer_resnet18.onnx"
  curl -fsSL "$RELEASE_URL/fer_fc_weight.npy" -o "$OUT/fer_fc_weight.npy"
fi
SUMS=$(mktemp)
for f in $FILES; do grep "  $f\$" "$SHA_FILE" >> "$SUMS"; done
(cd "$OUT" && sha256sum -c "$SUMS")
rm -f "$SUMS"
