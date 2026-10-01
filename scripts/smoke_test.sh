#!/bin/sh
# Smoke test a built image: start it, hit /health and /predict, scan the logs.
# Usage: smoke_test.sh IMAGE [FIXTURE_IMAGE]
set -eu

IMAGE=${1:?usage: smoke_test.sh IMAGE [FIXTURE_IMAGE]}
FIXTURE=${2:-tests/serving/fixtures/face.jpg}
NAME=fer-smoke
PORT=${PORT:-7870}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:7860" "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 30); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  sleep 1
done
curl -sf "localhost:$PORT/health" | jq -e '.status == "ok"' >/dev/null
curl -sf -F "file=@$FIXTURE" "localhost:$PORT/predict?explain=true" \
  | jq -e '(.faces | length) >= 1 and (.faces[0].gradcam | length) > 0' >/dev/null

if docker logs "$NAME" 2>&1 | grep -iE 'traceback|error'; then
  echo "Server logged errors" >&2
  exit 1
fi
echo "smoke test passed"
