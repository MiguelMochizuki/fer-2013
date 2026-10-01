# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so this layer is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --only-group serve --no-install-project

ARG RELEASE_URL=https://github.com/MiguelMochizuki/fer-2013/releases/download/models-v1
ARG SHA_FILE=serving/models.sha256
COPY scripts/fetch_models.sh /usr/local/bin/fetch_models.sh
COPY ${SHA_FILE} /tmp/models.sha256
RUN RELEASE_URL=${RELEASE_URL} fetch_models.sh /app/models /tmp/models.sha256


FROM python:3.12-slim-bookworm
# Hugging Face Spaces runs Docker Spaces as UID 1000.
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models
COPY src /app/src
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    MODELS_DIR=/app/models \
    PYTHONUNBUFFERED=1
USER user
EXPOSE 7860
CMD ["uvicorn", "fer_2013.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
