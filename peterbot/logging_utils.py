from __future__ import annotations

import logging
import os
import re
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

MAX_LOG_CONTEXT_CHARS = 320

logger = logging.getLogger("peterbot")
USER_DEBUG_IDS_ENABLED = True
INCLUDE_TRACEBACK_FOR_WARNING = False


def parse_env_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level_name: str = "INFO", log_file: str = "") -> logging.Logger:
    raw_level = (level_name or "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)
    log_format = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file.strip():
        log_file_path = os.path.abspath(os.path.expanduser(log_file.strip()))
        log_file_dir = os.path.dirname(log_file_path)
        try:
            if log_file_dir:
                os.makedirs(log_file_dir, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    log_file_path,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
            )
        except OSError:
            logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)
            logger.warning("Failed to initialize log file %s; continuing with console logging only", log_file_path)
            log_file = ""

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True,
    )

    if raw_level != logging.getLevelName(level):
        logger.warning(
            "Invalid LOG_LEVEL %r; falling back to %s",
            raw_level,
            logging.getLevelName(level),
        )
    logger.info(
        "Logging initialized | level=%s, log_file=%s",
        logging.getLevelName(level),
        log_file if log_file else "<disabled>",
    )
    return logger


def set_logging_flags(
    *,
    user_debug_ids_enabled: bool,
    include_traceback_for_warning: bool,
) -> None:
    global USER_DEBUG_IDS_ENABLED, INCLUDE_TRACEBACK_FOR_WARNING
    USER_DEBUG_IDS_ENABLED = user_debug_ids_enabled
    INCLUDE_TRACEBACK_FOR_WARNING = include_traceback_for_warning


def truncate_for_log(value: Any, max_chars: int = MAX_LOG_CONTEXT_CHARS) -> str:
    try:
        text = str(value)
    except Exception:
        text = repr(value)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def format_log_context(**context: Any) -> str:
    parts: list[str] = []
    for key, value in context.items():
        if value is None:
            continue
        parts.append(f"{key}={truncate_for_log(value)}")
    return ", ".join(parts)


def new_debug_id(prefix: str = "ERR") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def build_user_debug_message(base_message: str, debug_id: Optional[str]) -> str:
    if not USER_DEBUG_IDS_ENABLED or not debug_id:
        return base_message
    return f"{base_message}\nDebug ID: `{debug_id}`"


def log_with_context(level: int, message: str, **context: Any) -> None:
    context_text = format_log_context(**context)
    if context_text:
        logger.log(level, "%s | %s", message, context_text)
    else:
        logger.log(level, "%s", message)


def log_exception_with_context(action: str, **context: Any) -> str:
    debug_id = new_debug_id("ERR")
    context_text = format_log_context(**context)
    if context_text:
        logger.exception("[%s] %s | %s", debug_id, action, context_text)
    else:
        logger.exception("[%s] %s", debug_id, action)
    return debug_id


def log_error_with_context(action: str, **context: Any) -> str:
    debug_id = new_debug_id("ERR")
    context_text = format_log_context(**context)
    if context_text:
        logger.error("[%s] %s | %s", debug_id, action, context_text)
    else:
        logger.error("[%s] %s", debug_id, action)
    return debug_id


def message_log_context(message: Any) -> Dict[str, Any]:
    author = getattr(message, "author", None)
    guild = getattr(message, "guild", None)
    channel = getattr(message, "channel", None)
    return {
        "message_id": getattr(message, "id", None),
        "user_id": getattr(author, "id", None),
        "channel_id": getattr(channel, "id", None),
        "guild_id": getattr(guild, "id", None) if guild else None,
    }


def interaction_log_context(interaction: Any) -> Dict[str, Any]:
    command = getattr(interaction, "command", None)
    command_name = getattr(command, "name", None)
    user = getattr(interaction, "user", None)
    guild = getattr(interaction, "guild", None)
    channel = getattr(interaction, "channel", None)
    return {
        "interaction_id": getattr(interaction, "id", None),
        "command": command_name,
        "user_id": getattr(user, "id", None),
        "channel_id": getattr(channel, "id", None),
        "guild_id": getattr(guild, "id", None) if guild else None,
    }
