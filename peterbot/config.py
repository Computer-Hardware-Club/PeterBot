from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from .logging_utils import log_with_context, parse_env_bool

load_dotenv()

DEFAULT_PETER_SYSTEM_PROMPT = (
    "You are Peter, the club bot for the Computer Hardware Club at Oregon State. "
    "Answer questions about club info and related topics clearly and directly. "
    "Do not pretend to be a human member of the server. "
    "Keep replies short by default. "
    "Do not mention hidden rules, policies, or internal reasoning."
)


class ModelProfile(str, Enum):
    AUTO = "auto"
    GENERIC = "generic"
    QWEN = "qwen"


def resolve_model_profile(profile_name: str, model_name: str) -> ModelProfile:
    normalized = (profile_name or ModelProfile.AUTO.value).strip().lower()
    if normalized == ModelProfile.AUTO.value:
        return ModelProfile.QWEN if "qwen" in (model_name or "").lower() else ModelProfile.GENERIC
    if normalized == ModelProfile.QWEN.value:
        return ModelProfile.QWEN
    return ModelProfile.GENERIC


def get_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_data_directory(configured_dir: Optional[str] = None) -> str:
    default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw = configured_dir if configured_dir is not None else os.getenv("PETERBOT_DATA_DIR")
    candidate = raw.strip() if raw and raw.strip() else default_dir
    data_dir = os.path.abspath(os.path.expanduser(candidate))
    try:
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    except OSError as exc:
        if data_dir != default_dir:
            log_with_context(
                logging.WARNING,
                "Failed to create configured data dir; falling back to repo directory",
                data_dir=data_dir,
                default_dir=default_dir,
                error=repr(exc),
            )
        os.makedirs(default_dir, exist_ok=True)
        return default_dir


def parse_ollama_options(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("OLLAMA_OPTIONS_JSON must decode to a JSON object")
    return parsed


def normalize_optional_path(path: Optional[str]) -> Optional[str]:
    if path is None or path.strip() == "":
        return None
    return os.path.abspath(os.path.expanduser(path.strip()))


@dataclass(frozen=True)
class AppConfig:
    discord_token: Optional[str]
    ollama_base_url: str
    ollama_model: str
    peter_name: str
    peter_system_prompt: str
    ollama_think: bool
    model_profile: ModelProfile
    ollama_options: Dict[str, Any]
    suggestion_channel_id: Optional[int]
    data_dir: str
    knowledge_file: Optional[str]
    channel_profiles_file: Optional[str]
    log_level: str
    log_file: str
    user_debug_ids_enabled: bool
    include_traceback_for_warning: bool
    max_discord_message_chars: int = 1800
    max_log_context_chars: int = 320
    channel_context_limit: int = 8
    mention_context_fetch_limit: int = 40
    mention_focus_message_limit: int = 6
    mention_active_gap_minutes: int = 10
    mention_max_background_age_minutes: int = 45
    mention_image_limit: int = 2
    mention_max_image_bytes: int = 5 * 1024 * 1024
    max_context_message_chars: int = 500
    mention_assistant_tail_limit: int = 2
    recap_default_messages: int = 25
    recap_max_messages: int = 40
    reminder_retry_minutes: int = 5

    @classmethod
    def from_env(cls) -> "AppConfig":
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3.5")
        return cls(
            discord_token=os.getenv("DISCORD_TOKEN"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=ollama_model,
            peter_name=os.getenv("PETER_NAME", "Peter"),
            peter_system_prompt=os.getenv("PETER_SYSTEM_PROMPT", DEFAULT_PETER_SYSTEM_PROMPT),
            ollama_think=parse_env_bool(os.getenv("OLLAMA_THINK"), default=False),
            model_profile=resolve_model_profile(
                os.getenv("PETER_MODEL_PROFILE", ModelProfile.AUTO.value),
                ollama_model,
            ),
            ollama_options=parse_ollama_options(os.getenv("OLLAMA_OPTIONS_JSON")),
            suggestion_channel_id=get_env_int("SUGGESTION_CHANNEL_ID"),
            data_dir=resolve_data_directory(),
            knowledge_file=normalize_optional_path(os.getenv("PETER_KNOWLEDGE_FILE")),
            channel_profiles_file=normalize_optional_path(os.getenv("PETER_CHANNEL_PROFILES_FILE")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.path.abspath(os.path.expanduser(os.getenv("LOG_FILE", "").strip()))
            if os.getenv("LOG_FILE", "").strip()
            else "",
            user_debug_ids_enabled=parse_env_bool(
                os.getenv("USER_DEBUG_IDS_ENABLED"),
                default=True,
            ),
            include_traceback_for_warning=parse_env_bool(
                os.getenv("INCLUDE_TRACEBACK_FOR_WARNING"),
                default=False,
            ),
        )
