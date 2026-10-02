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
from fer_2013.serving.detector import Box, FaceDetector, YuNetDetector
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
DETECTOR_FILE = "face_detection_yunet_2026may.onnx"

# Keep uploads (<= limit) in memory instead of spooling them to a temp file.
MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD


class BoxOut(BaseModel):
    """A face box in pixels of the uploaded image."""

    x: int
    y: int
    w: int
    h: int


class FaceOut(BaseModel):
    """One detected face: box, calibrated emotion probabilities and an optional heatmap."""

    box: BoxOut
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    gradcam: str | None


class ImageInfo(BaseModel):
    """Size of the decoded upload in pixels."""

    width: int
    height: int


class Timings(BaseModel):
    """Where the request time went, in milliseconds."""

    detect: float
    classify: float
    total: float


class PredictResponse(BaseModel):
    """Result of POST /predict: one entry per detected face."""

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


class _ConcurrencyLimit:
    """Answer 503 to POST /predict when all slots are taken.

    Runs before the body is read, so a rejected upload is never buffered.
    """

    def __init__(self, app: ASGIApp, slots: threading.BoundedSemaphore) -> None:
        self.app = app
        self.slots = slots

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["path"] != "/predict"
            or scope["method"] != "POST"
        ):
            await self.app(scope, receive, send)
            return
        if not self.slots.acquire(blocking=False):
            resp = JSONResponse(
                {"detail": "server busy"}, status_code=503, headers={"Retry-After": "1"}
            )
            await resp(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.slots.release()


def _clamp(box: Box, width: int, height: int) -> Box | None:
    """Intersect a detector box with the image; None if nothing is left."""
    x0, y0 = max(box.x, 0), max(box.y, 0)
    x1, y1 = min(box.x + box.w, width), min(box.y + box.h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0, y0, x1 - x0, y1 - y0, box.score)


def create_app(
    classifier: Classifier | None = None,
    detector: FaceDetector | None = None,
    *,
    max_concurrent: int = 2,
) -> FastAPI:
    """Build the app. Missing models are loaded from $MODELS_DIR at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the models at startup unless they were injected (tests do)."""
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
    # Added last, so it is the outermost middleware and runs first.
    app.add_middleware(
        _ConcurrencyLimit, slots=threading.BoundedSemaphore(max_concurrent)
    )

    @app.exception_handler(ImageRejected)
    async def _rejected(_: Request, exc: ImageRejected) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the demo page."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, object]:
        """Report the loaded classifier hash, its calibration temperature and the detector class."""
        return {
            "status": "ok",
            "models": {
                "classifier": app.state.classifier.sha,
                "temperature": app.state.classifier.temperature,
                "detector": type(app.state.detector).__name__,
            },
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(file: UploadFile, explain: bool = False) -> PredictResponse:
        """Detect the faces in an uploaded JPEG, PNG or WebP and classify each one.

        Set ``explain=true`` to include a base64 Grad-CAM++ PNG per face.
        """
        return _predict(file, explain)

    def _predict(file: UploadFile, explain: bool) -> PredictResponse:
        clf: Classifier = app.state.classifier
        det: FaceDetector = app.state.detector
        t0 = time.perf_counter()
        image = decode_image(read_limited(iter(lambda: file.file.read(CHUNK), b"")))
        t_decoded = time.perf_counter()

        boxes = [
            b
            for raw in det.detect(image)
            if (b := _clamp(raw, image.width, image.height)) is not None
        ]
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
                detect=(t1 - t_decoded) * 1e3,
                classify=(t2 - t1) * 1e3,
                total=(t2 - t0) * 1e3,
            ),
        )

    return app
