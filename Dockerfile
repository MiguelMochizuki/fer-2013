# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so this layer is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --only-group serve --no-install-project \
    && find .venv -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +

ARG RELEASE_URL=https://github.com/MiguelMochizuki/fer-2013/releases/download/models-v1.2.0
ARG SHA_FILE=serving/models.sha256
COPY scripts/fetch_models.sh /usr/local/bin/fetch_models.sh
COPY ${SHA_FILE} /tmp/models.sha256
RUN RELEASE_URL=${RELEASE_URL} fetch_models.sh /app/models /tmp/models.sha256

# Collect the shared libraries Python and its extension modules need, minus the ones
# the distroless base already has, so the final image can drop Debian and its shell.
RUN mkdir -p /deps && { ldd /usr/local/bin/python3.12 /usr/local/lib/python3.12/lib-dynload/*.so /app/.venv/lib/python3.12/site-packages/*/*.so* /app/.venv/lib/python3.12/site-packages/*.libs/*.so* 2>/dev/null \
    | grep -o '/[^ ]*\.so[^ ]*' | sort -u | grep -v -E '/(libc|libm|libdl|libpthread|librt|libutil|ld-linux[^/]*|libgcc_s|libstdc\+\+)\.so' ; } | while read -r f; do [ -e "$f" ] && cp --parents -L "$f" /deps/; done; true

# Python's own tests, IDLE, pip and friends are dead weight at runtime.
RUN cd /usr/local/lib/python3.12 && rm -rf test idlelib ensurepip lib2to3 turtledemo tkinter site-packages/*


FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /usr/local/bin/python3.12 /usr/local/bin/python3 /usr/local/bin/
COPY --from=builder /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/libpython3.12.so.1.0
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /deps /
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
COPY src /app/src
WORKDIR /app
ENV LD_LIBRARY_PATH=/usr/local/lib \
    PATH=/app/.venv/bin:/usr/local/bin \
    PYTHONPATH=/app/src \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "fer_2013.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
