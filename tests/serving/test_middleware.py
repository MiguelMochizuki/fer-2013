import asyncio
import threading

import pytest
from starlette.exceptions import HTTPException
from starlette.types import Message, Receive, Scope, Send

from fer_2013.serving.api import _BodyLimit, _ConcurrencyLimit


def _scope(path: str = "/predict", method: str = "POST") -> dict[str, object]:
    return {"type": "http", "path": path, "method": method, "headers": []}


def test_body_limit_stops_reading_early() -> None:
    chunks_read = 0

    async def receive() -> Message:
        nonlocal chunks_read
        chunks_read += 1
        return {
            "type": "http.request",
            "body": b"x" * 1024 * 1024,
            "more_body": chunks_read < 8,  # an 8 MB body
        }

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        while True:  # a handler that wants the whole body
            msg = await receive()
            if not msg.get("more_body"):
                return

    async def send(msg: Message) -> None: ...

    app = _BodyLimit(inner, limit=3 * 1024 * 1024)
    with pytest.raises(HTTPException) as e:
        asyncio.run(app(_scope(), receive, send))
    assert e.value.status_code == 413
    assert chunks_read == 4  # cut at the first chunk past the limit, not at 8


def test_busy_request_is_rejected_before_its_body_is_read() -> None:
    gate = threading.Semaphore(0)
    receive_calls = 0
    sent: list[Message] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.get_running_loop().run_in_executor(None, gate.acquire)

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: Message) -> None:
        sent.append(msg)

    slots = threading.BoundedSemaphore(1)
    app = _ConcurrencyLimit(inner, slots=slots)

    async def scenario() -> None:
        first = asyncio.create_task(app(_scope(), receive, send))
        await asyncio.sleep(0.05)  # first holds the only slot
        await app(_scope(), receive, send)
        gate.release()
        await first

    asyncio.run(scenario())
    assert sent[0]["status"] == 503
    assert receive_calls == 0
    assert slots.acquire(blocking=False)  # the slot was released afterwards
