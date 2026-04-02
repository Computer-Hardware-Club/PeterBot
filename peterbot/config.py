from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

from .logging_utils import log_with_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.json"

DEFAULT_PETER_SYSTEM_PROMPT = """
# Peter the Bot - System Prompt

You are Peter, the official AI assistant for the Computer Hardware Club at Oregon State University (OSU). Your purpose is to be a helpful, friendly, and informative resource for all club members within the Discord server.

### **Your Identity:**
- **Your name is Peter.**
- You are an AI powered by the Gemma 4 E4B language model.
- You share your name with the club's Dell PowerEdge R620 server. You can mention this as a fun fact when relevant, but dont share when completely irrelevant.

### **Core Knowledge Base:**
You must use the following information when answering questions about the club.

**1. About the Club:**
- **Mission:** The Computer Hardware Club at OSU is a student-run organization dedicated to helping students build their resumes and get real, valuable experience. We provide a collaborative environment where members can learn about anything that piques their interest. We encourage our members to pick up ambitious projects and support them along the way, regardless of subject matter. If it involves electricity, we're game.
- **Community:** Our club welcomes students of all skill levels and backgrounds. Whether you're just starting your hardware journey or you're an experienced system administrator, our community offers opportunities to learn, share knowledge, and grow your technical skills. Through hands-on workshops, project collaborations, and regular meetings, we help bridge the gap between classroom learning and real-world implementation.

**2. Leadership & Meetings:**
- **President:** Oliver
- **Vice-President:** Scott
- **Technical Coordinator:** Lexi
- **Faculty Advisor:** Professor Ulbrich
- **Meetings:** We meet every **Friday at 6:00 PM** in **Kelly Engineering Center (KEC), room 1005**.
- **Meeting Types:** On scheduled workshop dates, we hold official workshops. On other Fridays, we have casual project/social nights. **Members should always check the Discord for announcements about these casual meetings**, as they are planned on a week-to-week basis.

**3. Membership:**
- **How to Join:** To officially join, go to the Ideal Logic website (https://apps.ideal-logic.com/osusee), log in with your OSU ONID, search for "Computer Hardware Club," and click join.
- **Cost & Eligibility:** There are **no member dues**. Everyone affiliated with OSU is eligible for membership, and people from all backgrounds and experience levels are encouraged to join.

**4. Club Resources & Involvement:**
- **Club Website:** https://computerhardwareclub.org/index.html
- **Getting Involved:** Members can be active in the Discord, help other members with questions and projects, and reach out to club officers to see if they can assist with club management or projects.
- **Club Server "Peter":** The club owns a Dell PowerEdge R620 server with the following specs: 2x Intel Xeon E5-2670 (totaling 16 cores), 32GB DDR3 ECC RAM, and dual 750W PSUs.
- **Server Access:** **Important:** At the moment, members do not have access to the server. The club does not have dedicated campus rack space yet, so the server cannot be run full-time.

### **Discord Interaction Rules:**
- **Code Formatting:** When providing code snippets, always use Discord's code block markdown. Start the block with three backticks and the language name (for example, `python`, `javascript`, `bash`), and end it with three backticks.
  - Example:
    ```python
    def hello_world():
        print("Hello, Computer Hardware Club!")
    ```
- **General Formatting:** Use Discord markdown to make your answers clear and easy to read.
  - `**Bold Text**` for emphasis.
  - `*Italicized Text*` for nuance.
  - `> Blockquote` for quoting users.
  - `||Spoiler Text||` to hide sensitive information.
- **Tone:** Be conversational and helpful without sounding robotic or corporate. Remember you are speaking to university students in a club setting.
""".strip()


def load_app_environment(
    env_file: Path = ENV_FILE,
    *,
    override: bool = True,
) -> bool:
    return load_dotenv(dotenv_path=env_file, override=override)


load_app_environment()


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


def resolve_data_directory(configured_dir: Optional[str] = None) -> str:
    default_dir = str((PROJECT_ROOT / "peterbot-data").resolve())
    candidate = configured_dir.strip() if configured_dir and configured_dir.strip() else default_dir
    data_dir = os.path.abspath(os.path.expanduser(candidate))
    try:
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    except OSError as exc:
        if data_dir != default_dir:
            log_with_context(
                logging.WARNING,
                "Failed to create configured data dir; falling back to default data dir",
                data_dir=data_dir,
                default_dir=default_dir,
                error=repr(exc),
            )
        os.makedirs(default_dir, exist_ok=True)
        return default_dir


def _resolve_config_path(config_file: Optional[str] = None) -> Path:
    raw = config_file or os.getenv("PETERBOT_CONFIG_FILE") or str(DEFAULT_CONFIG_FILE)
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _load_config_json(config_path: Path) -> Mapping[str, Any]:
    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Configuration file not found: {config_path}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Configuration file is not valid JSON: {config_path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Configuration file root must be a JSON object: {config_path}")
    return parsed


def _expect_section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a JSON object")
    return value


def _require_str(section: Mapping[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{key}' must be a non-empty string")
    return value.strip()


def _optional_str(section: Mapping[str, Any], key: str, default: Optional[str] = None) -> Optional[str]:
    value = section.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config value '{key}' must be a string")
    stripped = value.strip()
    return stripped if stripped else None


def _str_or_default(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Config value '{key}' must be a string")
    stripped = value.strip()
    return stripped if stripped else default


def _bool_or_default(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a boolean")
    return value


def _int_or_default(section: Mapping[str, Any], key: str, default: int, *, minimum: Optional[int] = None) -> int:
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"Config value '{key}' must be >= {minimum}")
    return value


def _optional_int(section: Mapping[str, Any], key: str) -> Optional[int]:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be an integer")
    return value


def _optional_float(section: Mapping[str, Any], key: str) -> Optional[float]:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a number")
    return float(value)


def _dict_or_default(section: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = section.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config value '{key}' must be a JSON object")
    return dict(value)


def _list_of_strings(section: Mapping[str, Any], key: str) -> list[str]:
    value = section.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Config value '{key}' must be a list of non-empty strings")
    return [item.strip() for item in value]


def _resolve_optional_path(
    base_dir: Path,
    raw_path: Optional[str],
    *,
    must_exist: bool = False,
) -> Optional[str]:
    if raw_path is None or not raw_path.strip():
        return None
    path = Path(os.path.expanduser(raw_path.strip()))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    resolved = str(path)
    if must_exist and not path.exists():
        raise ValueError(f"Configured path does not exist: {resolved}")
    return resolved


def _normalize_base_url(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("inference.base_url must start with http:// or https:// and include a host")
    return raw.rstrip("/")


@dataclass(frozen=True)
class PersonaConfig:
    name: str
    system_prompt: str
    model_profile: ModelProfile


@dataclass(frozen=True)
class DiscordConfig:
    suggestion_channel_id: Optional[int]


@dataclass(frozen=True)
class InferenceConfig:
    base_url: str
    model: str
    timeout_seconds: int
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    extra_request_body: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlamaServerConfig:
    enabled: bool
    model_path: Optional[str]
    host: str
    port: int
    ctx_size: int
    threads: int
    batch_size: int
    parallel: int
    continuous_batching: bool
    n_gpu_layers: int
    metrics: bool
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str
    knowledge_file: Optional[str]
    channel_profiles_file: Optional[str]
    log_file: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    user_debug_ids_enabled: bool
    include_traceback_for_warning: bool


@dataclass(frozen=True)
class BehaviorConfig:
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


@dataclass(frozen=True)
class AppConfig:
    discord_token: Optional[str]
    llama_cpp_api_key: Optional[str]
    persona: PersonaConfig
    discord: DiscordConfig
    inference: InferenceConfig
    llama_server: LlamaServerConfig
    paths: PathsConfig
    logging: LoggingConfig
    behavior: BehaviorConfig
    config_path: str

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> "AppConfig":
        load_app_environment()
        config_path = _resolve_config_path(config_file)
        raw = _load_config_json(config_path)
        base_dir = config_path.parent

        persona_section = _expect_section(raw, "persona")
        discord_section = _expect_section(raw, "discord")
        inference_section = _expect_section(raw, "inference")
        llama_server_section = _expect_section(raw, "llama_server")
        paths_section = _expect_section(raw, "paths")
        logging_section = _expect_section(raw, "logging")
        behavior_section = _expect_section(raw, "behavior")

        model_name = _require_str(inference_section, "model")
        persona = PersonaConfig(
            name=_str_or_default(persona_section, "name", "Peter"),
            system_prompt=_str_or_default(persona_section, "system_prompt", DEFAULT_PETER_SYSTEM_PROMPT),
            model_profile=resolve_model_profile(
                _str_or_default(persona_section, "model_profile", ModelProfile.AUTO.value),
                model_name,
            ),
        )
        config = cls(
            discord_token=os.getenv("DISCORD_TOKEN"),
            llama_cpp_api_key=_optional_str({"llama_cpp_api_key": os.getenv("LLAMA_CPP_API_KEY")}, "llama_cpp_api_key"),
            persona=persona,
            discord=DiscordConfig(
                suggestion_channel_id=_optional_int(discord_section, "suggestion_channel_id"),
            ),
            inference=InferenceConfig(
                base_url=_normalize_base_url(_require_str(inference_section, "base_url")),
                model=model_name,
                timeout_seconds=_int_or_default(inference_section, "timeout_seconds", 300, minimum=1),
                max_tokens=_optional_int(inference_section, "max_tokens"),
                temperature=_optional_float(inference_section, "temperature"),
                top_p=_optional_float(inference_section, "top_p"),
                extra_request_body=_dict_or_default(inference_section, "extra_request_body"),
            ),
            llama_server=LlamaServerConfig(
                enabled=_bool_or_default(llama_server_section, "enabled", False),
                model_path=_resolve_optional_path(
                    base_dir,
                    _optional_str(llama_server_section, "model_path"),
                    must_exist=False,
                ),
                host=_str_or_default(llama_server_section, "host", "127.0.0.1"),
                port=_int_or_default(llama_server_section, "port", 8080, minimum=1),
                ctx_size=_int_or_default(llama_server_section, "ctx_size", 4096, minimum=1),
                threads=_int_or_default(llama_server_section, "threads", 0, minimum=0),
                batch_size=_int_or_default(llama_server_section, "batch_size", 512, minimum=1),
                parallel=_int_or_default(llama_server_section, "parallel", 1, minimum=1),
                continuous_batching=_bool_or_default(llama_server_section, "continuous_batching", True),
                n_gpu_layers=_int_or_default(llama_server_section, "n_gpu_layers", 0, minimum=0),
                metrics=_bool_or_default(llama_server_section, "metrics", False),
                extra_args=_list_of_strings(llama_server_section, "extra_args"),
            ),
            paths=PathsConfig(
                data_dir=resolve_data_directory(
                    _resolve_optional_path(base_dir, _optional_str(paths_section, "data_dir"), must_exist=False)
                ),
                knowledge_file=_resolve_optional_path(
                    base_dir,
                    _optional_str(paths_section, "knowledge_file"),
                    must_exist=True,
                ),
                channel_profiles_file=_resolve_optional_path(
                    base_dir,
                    _optional_str(paths_section, "channel_profiles_file"),
                    must_exist=True,
                ),
                log_file=_resolve_optional_path(
                    base_dir,
                    _optional_str(paths_section, "log_file"),
                    must_exist=False,
                )
                or "",
            ),
            logging=LoggingConfig(
                level=_str_or_default(logging_section, "level", "INFO"),
                user_debug_ids_enabled=_bool_or_default(logging_section, "user_debug_ids_enabled", True),
                include_traceback_for_warning=_bool_or_default(
                    logging_section,
                    "include_traceback_for_warning",
                    False,
                ),
            ),
            behavior=BehaviorConfig(
                max_discord_message_chars=_int_or_default(
                    behavior_section,
                    "max_discord_message_chars",
                    1800,
                    minimum=1,
                ),
                max_log_context_chars=_int_or_default(behavior_section, "max_log_context_chars", 320, minimum=1),
                channel_context_limit=_int_or_default(behavior_section, "channel_context_limit", 8, minimum=1),
                mention_context_fetch_limit=_int_or_default(
                    behavior_section,
                    "mention_context_fetch_limit",
                    40,
                    minimum=1,
                ),
                mention_focus_message_limit=_int_or_default(
                    behavior_section,
                    "mention_focus_message_limit",
                    6,
                    minimum=1,
                ),
                mention_active_gap_minutes=_int_or_default(
                    behavior_section,
                    "mention_active_gap_minutes",
                    10,
                    minimum=1,
                ),
                mention_max_background_age_minutes=_int_or_default(
                    behavior_section,
                    "mention_max_background_age_minutes",
                    45,
                    minimum=1,
                ),
                mention_image_limit=_int_or_default(behavior_section, "mention_image_limit", 2, minimum=0),
                mention_max_image_bytes=_int_or_default(
                    behavior_section,
                    "mention_max_image_bytes",
                    5 * 1024 * 1024,
                    minimum=1,
                ),
                max_context_message_chars=_int_or_default(
                    behavior_section,
                    "max_context_message_chars",
                    500,
                    minimum=1,
                ),
                mention_assistant_tail_limit=_int_or_default(
                    behavior_section,
                    "mention_assistant_tail_limit",
                    2,
                    minimum=0,
                ),
                recap_default_messages=_int_or_default(
                    behavior_section,
                    "recap_default_messages",
                    25,
                    minimum=1,
                ),
                recap_max_messages=_int_or_default(behavior_section, "recap_max_messages", 40, minimum=1),
                reminder_retry_minutes=_int_or_default(
                    behavior_section,
                    "reminder_retry_minutes",
                    5,
                    minimum=1,
                ),
            ),
            config_path=str(config_path),
        )
        config.validate()
        return config

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls.load()

    def validate(self) -> None:
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN is not set. Add it to .env.")

        if not self.inference.model.strip():
            raise ValueError("inference.model must not be empty")

        if self.inference.max_tokens is not None and self.inference.max_tokens <= 0:
            raise ValueError("inference.max_tokens must be > 0 when set")

        if self.inference.temperature is not None and self.inference.temperature < 0:
            raise ValueError("inference.temperature must be >= 0 when set")

        if self.inference.top_p is not None and not 0 < self.inference.top_p <= 1:
            raise ValueError("inference.top_p must be > 0 and <= 1 when set")

        if not self.paths.data_dir:
            raise ValueError("Resolved data_dir is empty")

        if self.llama_server.enabled:
            if not self.llama_server.model_path:
                raise ValueError("llama_server.model_path is required when llama_server.enabled is true")
            if not Path(self.llama_server.model_path).exists():
                raise ValueError(f"llama_server.model_path does not exist: {self.llama_server.model_path}")

            base = urlparse(self.inference.base_url)
            if base.hostname not in {self.llama_server.host, "127.0.0.1", "localhost"}:
                raise ValueError("Bundled llama_server requires inference.base_url to point at the local server")
            if (base.port or (443 if base.scheme == "https" else 80)) != self.llama_server.port:
                raise ValueError("Bundled llama_server requires inference.base_url to use llama_server.port")

    @property
    def peter_name(self) -> str:
        return self.persona.name

    @property
    def peter_system_prompt(self) -> str:
        return self.persona.system_prompt

    @property
    def model_profile(self) -> ModelProfile:
        return self.persona.model_profile

    @property
    def suggestion_channel_id(self) -> Optional[int]:
        return self.discord.suggestion_channel_id

    @property
    def data_dir(self) -> str:
        return self.paths.data_dir

    @property
    def knowledge_file(self) -> Optional[str]:
        return self.paths.knowledge_file

    @property
    def channel_profiles_file(self) -> Optional[str]:
        return self.paths.channel_profiles_file

    @property
    def log_level(self) -> str:
        return self.logging.level

    @property
    def log_file(self) -> str:
        return self.paths.log_file

    @property
    def user_debug_ids_enabled(self) -> bool:
        return self.logging.user_debug_ids_enabled

    @property
    def include_traceback_for_warning(self) -> bool:
        return self.logging.include_traceback_for_warning

    @property
    def max_discord_message_chars(self) -> int:
        return self.behavior.max_discord_message_chars

    @property
    def max_log_context_chars(self) -> int:
        return self.behavior.max_log_context_chars

    @property
    def channel_context_limit(self) -> int:
        return self.behavior.channel_context_limit

    @property
    def mention_context_fetch_limit(self) -> int:
        return self.behavior.mention_context_fetch_limit

    @property
    def mention_focus_message_limit(self) -> int:
        return self.behavior.mention_focus_message_limit

    @property
    def mention_active_gap_minutes(self) -> int:
        return self.behavior.mention_active_gap_minutes

    @property
    def mention_max_background_age_minutes(self) -> int:
        return self.behavior.mention_max_background_age_minutes

    @property
    def mention_image_limit(self) -> int:
        return self.behavior.mention_image_limit

    @property
    def mention_max_image_bytes(self) -> int:
        return self.behavior.mention_max_image_bytes

    @property
    def max_context_message_chars(self) -> int:
        return self.behavior.max_context_message_chars

    @property
    def mention_assistant_tail_limit(self) -> int:
        return self.behavior.mention_assistant_tail_limit

    @property
    def recap_default_messages(self) -> int:
        return self.behavior.recap_default_messages

    @property
    def recap_max_messages(self) -> int:
        return self.behavior.recap_max_messages

    @property
    def reminder_retry_minutes(self) -> int:
        return self.behavior.reminder_retry_minutes
