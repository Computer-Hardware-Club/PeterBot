FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system peterbot \
    && useradd --system --gid peterbot --home-dir /app --shell /usr/sbin/nologin peterbot

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY bot.py README.md config.json .env.example club-knowledge.md ./
COPY docker ./docker
COPY peterbot ./peterbot
COPY deploy/housekeeping.py deploy/state_backup.py ./deploy/

RUN chmod +x docker/entrypoint.sh \
    && mkdir -p /app/peterbot-data /app/logs \
    && chown -R peterbot:peterbot /app

USER peterbot

ENTRYPOINT ["/usr/bin/tini", "--", "./docker/entrypoint.sh"]

FROM base AS bot
ARG PETERBOT_REVISION=unknown
ENV PETERBOT_REVISION=$PETERBOT_REVISION
LABEL org.opencontainers.image.revision=$PETERBOT_REVISION

FROM ghcr.io/ggml-org/llama.cpp:server AS llama_cpp_server

FROM base AS bundled

ENV LD_LIBRARY_PATH=/app

COPY --from=llama_cpp_server /app/ /app/

USER peterbot

ENTRYPOINT ["/usr/bin/tini", "--", "./docker/entrypoint.sh"]
