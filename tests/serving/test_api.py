import base64
import io
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from PIL import Image
from starlette.formparsers import MultiPartParser

from fer_2013.serving.api import create_app
from fer_2013.serving.classifier import Classifier
from fer_2013.serving.detector import Box
from fer_2013.serving.images import MAX_UPLOAD_BYTES
from fer_2013.serving.labels import EMOTIONS


class FakeDetector:
    def __init__(self, boxes: list[Box] | None = None) -> None:
        self.boxes = boxes if boxes is not None else [Box(20, 20, 100, 100, 0.9)]
        self.seen_sizes: list[tuple[int, int]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail = False

    def detect(self, image: Image.Image) -> list[Box]:
        self.seen_sizes.append(image.size)
        self.entered.set()
        assert self.release.wait(5)
        if self.fail:
            raise RuntimeError("secret internal detail")
        return self.boxes


def _png(size: tuple[int, int] = (200, 200), mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, 100 if mode in ("L", "P", "I;16") else None).save(
        buf, format="PNG"
    )
    return buf.getvalue()


def _post(client: TestClient, data: bytes, **params: object) -> Response:
    resp: Response = client.post(
        "/predict", params=params, files={"file": ("x.png", data, "image/png")}
    )
    return resp


@pytest.fixture()
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture()
def classifier(onnx_models: tuple[Path, Path]) -> Classifier:
    return Classifier(*onnx_models)


@pytest.fixture()
def client(classifier: Classifier, detector: FakeDetector) -> Iterator[TestClient]:
    with TestClient(
        create_app(classifier, detector), raise_server_exceptions=False
    ) as c:
        yield c


def test_predict_returns_faces_with_all_probabilities(client: TestClient) -> None:
    r = _post(client, _png())
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 200, "height": 200}
    (face,) = body["faces"]
    assert face["box"] == {"x": 20, "y": 20, "w": 100, "h": 100}
    assert face["emotion"] in EMOTIONS
    assert set(face["probabilities"]) == set(EMOTIONS)
    assert sum(face["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert face["confidence"] == pytest.approx(max(face["probabilities"].values()))
    assert face["gradcam"] is None


def test_explain_true_returns_png_base64(client: TestClient) -> None:
    (face,) = _post(client, _png(), explain="true").json()["faces"]
    img = Image.open(io.BytesIO(base64.b64decode(face["gradcam"])))
    assert img.format == "PNG" and img.size == (128, 128)


def test_no_face_returns_empty_list_200(
    client: TestClient, detector: FakeDetector
) -> None:
    detector.boxes = []
    r = _post(client, _png())
    assert r.status_code == 200 and r.json()["faces"] == []


def test_empty_area_box_is_skipped(client: TestClient, detector: FakeDetector) -> None:
    detector.boxes = [Box(5, 5, 0, 10, 0.9), Box(20, 20, 50, 50, 0.8)]
    assert len(_post(client, _png()).json()["faces"]) == 1


def test_multiple_faces_each_have_own_result(
    client: TestClient, detector: FakeDetector
) -> None:
    detector.boxes = [Box(0, 0, 60, 60, 0.9), Box(100, 100, 60, 60, 0.8)]
    faces = _post(client, _png(), explain="true").json()["faces"]
    assert [f["box"]["x"] for f in faces] == [0, 100]
    assert all(f["gradcam"] for f in faces)


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "I;16"])
def test_other_image_modes_are_accepted(client: TestClient, mode: str) -> None:
    assert _post(client, _png(mode=mode)).status_code == 200


def test_non_image_422(client: TestClient) -> None:
    r = _post(client, b"definitely not an image")
    assert r.status_code == 422 and r.json() == {
        "detail": "invalid or unsupported image"
    }


def test_upload_over_5mb_413(client: TestClient) -> None:
    assert _post(client, b"x" * (MAX_UPLOAD_BYTES + 1)).status_code == 413


def test_chunked_upload_over_limit_413(client: TestClient) -> None:
    boundary = "bnd"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="x.png"\r\nContent-Type: image/png\r\n\r\n'
    ).encode()

    def body() -> Iterator[bytes]:
        yield head
        for _ in range(8):  # 8 MB, no Content-Length (chunked)
            yield b"x" * (1024 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    r = client.post(
        "/predict",
        content=body(),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert r.status_code == 413


def test_upload_is_kept_in_memory() -> None:
    assert MultiPartParser.spool_max_size >= MAX_UPLOAD_BYTES


def test_busy_returns_503_with_retry_after(
    classifier: Classifier, detector: FakeDetector
) -> None:
    detector.release.clear()
    app = create_app(classifier, detector, max_concurrent=1)
    with (
        TestClient(app, raise_server_exceptions=False) as c,
        ThreadPoolExecutor(1) as pool,
    ):
        first = pool.submit(_post, c, _png())
        assert detector.entered.wait(5)
        r = _post(c, _png())
        detector.release.set()
        assert first.result().status_code == 200
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"


def test_internal_error_is_generic_500(
    client: TestClient, detector: FakeDetector
) -> None:
    detector.fail = True
    r = _post(client, _png())
    assert r.status_code == 500
    assert r.json() == {"detail": "internal error"}
    assert "secret" not in r.text


def test_health_reports_model_sha(client: TestClient, classifier: Classifier) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models"]["classifier"] == classifier.sha


def test_exif_rotated_photo_is_upright_before_detect(
    client: TestClient, detector: FakeDetector
) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (80, 40)).save(buf, format="JPEG", exif=exif.tobytes())
    detector.seen_sizes.clear()
    r = client.post("/predict", files={"file": ("x.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200
    assert detector.seen_sizes == [(40, 80)]
    assert r.json()["image"] == {"width": 40, "height": 80}


def test_timings_present(client: TestClient) -> None:
    t = _post(client, _png()).json()["timings_ms"]
    assert set(t) == {"detect", "classify", "total"}
    assert t["total"] >= t["detect"] >= 0


def test_index_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert '<input type="file"' in r.text


def test_response_boxes_are_clamped_to_the_image(
    client: TestClient, detector: FakeDetector
) -> None:
    detector.boxes = [Box(-3, 10, 50, 50, 0.9), Box(180, 180, 50, 50, 0.8)]
    faces = _post(client, _png((200, 200))).json()["faces"]
    boxes = [f["box"] for f in faces]
    assert boxes == [
        {"x": 0, "y": 10, "w": 47, "h": 50},
        {"x": 180, "y": 180, "w": 20, "h": 20},
    ]


def test_timings_detect_excludes_upload_decoding(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from fer_2013.serving import api
    from fer_2013.serving.images import decode_image as real

    def slow_decode(data: bytes) -> Image.Image:
        time.sleep(0.2)
        return real(data)

    monkeypatch.setattr(api, "decode_image", slow_decode)
    t = _post(client, _png()).json()["timings_ms"]
    assert t["detect"] < 100
    assert t["total"] >= 200
