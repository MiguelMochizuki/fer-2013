#!/bin/sh
# Download the serving models and verify them against pinned sha256 sums.
# Usage: RELEASE_URL=<release download url> fetch_models.sh OUT_DIR SHA_FILE
#        ONLY_YUNET=1 fetch_models.sh OUT_DIR SHA_FILE   (both YuNet files, for tests)
# The API and the browser demo use the same dynamic-shape YuNet (2026may). The
# 2023mar file is only OpenCV's reference in tests/serving/test_detector.py.
set -eu

OUT=${1:?usage: fetch_models.sh OUT_DIR SHA_FILE}
SHA_FILE=${2:?usage: fetch_models.sh OUT_DIR SHA_FILE}
YUNET_ZOO=https://media.githubusercontent.com/media/opencv/opencv_zoo
YUNET=face_detection_yunet_2026may.onnx
YUNET_URL=$YUNET_ZOO/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/$YUNET
YUNET_REF=face_detection_yunet_2023mar.onnx
YUNET_REF_URL=$YUNET_ZOO/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/$YUNET_REF

if [ -n "${ONLY_YUNET:-}" ]; then
  FILES="$YUNET $YUNET_REF"
else
  RELEASE_URL=${RELEASE_URL:?RELEASE_URL is required}
  FILES="fer_resnet18.onnx fer_fc_weight.npy $YUNET"
fi

# Fail fast if a model is missing from the checksum file.
for f in $FILES; do
  grep -q "  $f\$" "$SHA_FILE" || { echo "no checksum for $f in $SHA_FILE" >&2; exit 1; }
done

mkdir -p "$OUT"
for f in $FILES; do
  case $f in
    "$YUNET") curl -fsSL "$YUNET_URL" -o "$OUT/$f" ;;
    "$YUNET_REF") curl -fsSL "$YUNET_REF_URL" -o "$OUT/$f" ;;
    *) curl -fsSL "$RELEASE_URL/$f" -o "$OUT/$f" ;;
  esac
done
SUMS=$(mktemp)
for f in $FILES; do grep "  $f\$" "$SHA_FILE" >> "$SUMS"; done
(cd "$OUT" && sha256sum -c "$SUMS")
rm -f "$SUMS"
