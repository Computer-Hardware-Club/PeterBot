FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

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

FROM ghcr.io/ggml-org/llama.cpp:server AS bundled

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

USER root

RUN if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends python3 python3-pip tini && rm -rf /var/lib/apt/lists/* \
        && groupadd --system peterbot \
        && useradd --system --gid peterbot --home-dir /app --shell /usr/sbin/nologin peterbot; \
    elif command -v apk >/dev/null 2>&1; then \
        apk add --no-cache python3 py3-pip tini \
        && addgroup -S peterbot \
        && adduser -S -D -H -h /app -G peterbot peterbot; \
    else \
        echo "Unsupported package manager in llama.cpp server image" >&2; \
        exit 1; \
    fi

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
