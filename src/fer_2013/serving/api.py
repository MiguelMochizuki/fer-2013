"""FastAPI app: POST /predict, GET /health.

Uploads are bounded (size, pixels, formats), processed only in memory, and
errors never expose internals. Concurrency is capped; extra requests get 503.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from starlette.formparsers import MultiPartParser
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fer_2013.serving.classifier import Classifier
from fer_2013.serving.detector import FaceDetector, YuNetDetector
from fer_2013.serving.gradcam import gradcam_pp, overlay_png_b64
from fer_2013.serving.images import (
    MAX_UPLOAD_BYTES,
    ImageRejected,
    decode_image,
    read_limited,
)
from fer_2013.serving.labels import EMOTIONS
from fer_2013.serving.preprocess import crop_gray, to_input

logger = logging.getLogger("fer_2013.serving")

MULTIPART_OVERHEAD = 64 * 1024
CHUNK = 64 * 1024
STATIC_DIR = Path(__file__).parent / "static"
CLASSIFIER_FILE = "fer_resnet18.onnx"
FC_WEIGHT_FILE = "fer_fc_weight.npy"
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"

# Keep uploads (<= limit) in memory instead of spooling them to a temp file.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD


class BoxOut(BaseModel):
    x: int
    y: int
    w: int
    h: int


class FaceOut(BaseModel):
    box: BoxOut
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    gradcam: str | None


class ImageInfo(BaseModel):
    width: int
    height: int


class Timings(BaseModel):
    detect: float
    classify: float
    total: float


class PredictResponse(BaseModel):
    image: ImageInfo
    faces: list[FaceOut]
    timings_ms: Timings


class _BodyLimit:
    """Reject request bodies over `limit` bytes while they stream in."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length", b"")
        if length.isdigit() and int(length) > self.limit:
            resp = JSONResponse({"detail": "file too large"}, status_code=413)
            await resp(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.limit:
                    raise HTTPException(413, "file too large")
            return msg

        await self.app(scope, limited_receive, send)


def create_app(
    classifier: Classifier | None = None,
    detector: FaceDetector | None = None,
    *,
    max_concurrent: int = 2,
) -> FastAPI:
    """Build the app. Missing models are loaded from $MODELS_DIR at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        models = Path(os.environ.get("MODELS_DIR", "/app/models"))
        if app.state.classifier is None:
            app.state.classifier = Classifier(
                models / CLASSIFIER_FILE, models / FC_WEIGHT_FILE
            )
        if app.state.detector is None:
            app.state.detector = YuNetDetector(models / DETECTOR_FILE)
        yield

    app = FastAPI(title="FER-2013 emotion API", lifespan=lifespan)
    app.state.classifier = classifier
    app.state.detector = detector
    app.add_middleware(_BodyLimit, limit=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD)
    slots = threading.BoundedSemaphore(max_concurrent)

    @app.exception_handler(ImageRejected)
    async def _rejected(_: Request, exc: ImageRejected) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "models": {
                "classifier": app.state.classifier.sha,
                "detector": type(app.state.detector).__name__,
            },
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        file: UploadFile, explain: bool = False
    ) -> PredictResponse | JSONResponse:
        if not slots.acquire(blocking=False):
            return JSONResponse(
                {"detail": "server busy"}, status_code=503, headers={"Retry-After": "1"}
            )
        try:
            return _predict(file, explain)
        finally:
            slots.release()

    def _predict(file: UploadFile, explain: bool) -> PredictResponse:
        clf: Classifier = app.state.classifier
        det: FaceDetector = app.state.detector
        t0 = time.perf_counter()
        image = decode_image(read_limited(iter(lambda: file.file.read(CHUNK), b"")))

        boxes = det.detect(image)
        t1 = time.perf_counter()

        crops = [
            (box, gray) for box in boxes if (gray := crop_gray(image, box)) is not None
        ]
        faces: list[FaceOut] = []
        if crops:
            batch = np.concatenate([to_input(gray) for _, gray in crops])
            probs, features = clf.predict(batch)
            for i, (box, gray) in enumerate(crops):
                cls = int(probs[i].argmax())
                heat = (
                    overlay_png_b64(gray, gradcam_pp(features[i], clf.fc_weight, cls))
                    if explain
                    else None
                )
                faces.append(
                    FaceOut(
                        box=BoxOut(x=box.x, y=box.y, w=box.w, h=box.h),
                        emotion=EMOTIONS[cls],
                        confidence=float(probs[i][cls]),
                        probabilities={
                            e: float(p) for e, p in zip(EMOTIONS, probs[i], strict=True)
                        },
                        gradcam=heat,
                    )
                )
        t2 = time.perf_counter()
        logger.info("predict faces=%d total_ms=%.1f", len(faces), (t2 - t0) * 1e3)
        return PredictResponse(
            image=ImageInfo(width=image.width, height=image.height),
            faces=faces,
            timings_ms=Timings(
                detect=(t1 - t0) * 1e3, classify=(t2 - t1) * 1e3, total=(t2 - t0) * 1e3
            ),
        )

    return app
