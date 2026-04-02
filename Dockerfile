FROM python:3.12-slim AS base

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

COPY bot.py README.md config.json .env.example ./
COPY docker ./docker
COPY peterbot ./peterbot

RUN chmod +x docker/entrypoint.sh \
    && mkdir -p /app/peterbot-data /app/logs \
    && chown -R peterbot:peterbot /app

USER peterbot

ENTRYPOINT ["/usr/bin/tini", "--", "./docker/entrypoint.sh"]

FROM base AS bot

FROM ghcr.io/ggml-org/llama.cpp:server AS llama_cpp_server

FROM base AS bundled

ENV LD_LIBRARY_PATH=/app

COPY --from=llama_cpp_server /app/ /app/

USER peterbot

ENTRYPOINT ["/usr/bin/tini", "--", "./docker/entrypoint.sh"]
