import json
from pathlib import Path

import pytest

from peterbot.config import (
    AppConfig,
    BehaviorConfig,
    DiscordConfig,
    InferenceConfig,
    LlamaServerConfig,
    LoggingConfig,
    ModelProfile,
    PathsConfig,
    PersonaConfig,
    load_app_environment,
    resolve_data_directory,
    resolve_model_profile,
)
from peterbot.knowledge import build_knowledge_excerpt, load_channel_profiles, load_knowledge_chunks, rank_knowledge_chunks
from peterbot.prompts import (
    MENTION_MODE,
    add_no_think_suffix,
    build_context_line,
    build_system_prompt,
    cleanup_response_text,
)


FIXTURES = Path(__file__).parent / "fixtures"
PERSONALITY_FIXTURES = json.loads((FIXTURES / "personality_cleanup.json").read_text(encoding="utf-8"))


def build_config(tmp_path) -> AppConfig:
    return AppConfig(
        discord_token="token",
        llama_cpp_api_key=None,
        persona=PersonaConfig(
            name="Peter",
            system_prompt="You are Peter.",
            model_profile=ModelProfile.QWEN,
        ),
        discord=DiscordConfig(suggestion_channel_id=123),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:8080",
            model="qwen3.5",
            timeout_seconds=300,
            max_tokens=512,
            temperature=0.3,
            top_p=None,
            extra_request_body={},
        ),
        llama_server=LlamaServerConfig(
            enabled=False,
            model_path=None,
            host="127.0.0.1",
            port=8080,
            ctx_size=4096,
            threads=0,
            batch_size=512,
            parallel=1,
            continuous_batching=True,
            n_gpu_layers=0,
            metrics=False,
            extra_args=[],
        ),
        paths=PathsConfig(
            data_dir=str(tmp_path),
            knowledge_file=str(FIXTURES / "club_knowledge.md"),
            channel_profiles_file=str(FIXTURES / "channel_profiles.json"),
            log_file="",
        ),
        logging=LoggingConfig(
            level="INFO",
            user_debug_ids_enabled=True,
            include_traceback_for_warning=False,
        ),
        behavior=BehaviorConfig(),
        config_path=str(tmp_path / "config.json"),
    )


def build_config_payload(tmp_path: Path) -> dict:
    return {
        "persona": {
            "name": "Peter",
            "system_prompt": "You are Peter.",
            "model_profile": "auto",
        },
        "discord": {
            "suggestion_channel_id": 123,
        },
        "inference": {
            "base_url": "http://127.0.0.1:8080",
            "model": "qwen3.5",
            "timeout_seconds": 420,
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": None,
            "extra_request_body": {},
        },
        "llama_server": {
            "enabled": False,
            "model_path": None,
            "host": "127.0.0.1",
            "port": 8080,
            "ctx_size": 4096,
            "threads": 0,
            "batch_size": 512,
            "parallel": 1,
            "continuous_batching": True,
            "n_gpu_layers": 0,
            "metrics": False,
            "extra_args": [],
        },
        "paths": {
            "data_dir": str(tmp_path / "state"),
            "knowledge_file": str(FIXTURES / "club_knowledge.md"),
            "channel_profiles_file": str(FIXTURES / "channel_profiles.json"),
            "log_file": str(tmp_path / "logs" / "peterbot.log"),
        },
        "logging": {
            "level": "INFO",
            "user_debug_ids_enabled": True,
            "include_traceback_for_warning": False,
        },
        "behavior": {
            "max_discord_message_chars": 1800,
            "max_log_context_chars": 320,
            "channel_context_limit": 8,
            "mention_context_fetch_limit": 40,
            "mention_focus_message_limit": 6,
            "mention_active_gap_minutes": 10,
            "mention_max_background_age_minutes": 45,
            "mention_image_limit": 2,
            "mention_max_image_bytes": 5242880,
            "max_context_message_chars": 500,
            "mention_assistant_tail_limit": 2,
            "recap_default_messages": 25,
            "recap_max_messages": 40,
            "reminder_retry_minutes": 5,
        },
    }


def write_config(tmp_path: Path, payload: dict) -> Path:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(payload), encoding="utf-8")
    return config_file


def test_resolve_model_profile_auto_prefers_qwen() -> None:
    assert resolve_model_profile("auto", "qwen3.5:14b") == ModelProfile.QWEN
    assert resolve_model_profile("auto", "ministral-3:8b") == ModelProfile.GENERIC


def test_load_app_environment_uses_repo_env_file_values_over_stale_env(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DISCORD_TOKEN=env-token\n", encoding="utf-8")
    monkeypatch.setenv("DISCORD_TOKEN", "stale-token")
    config_file = write_config(tmp_path, build_config_payload(tmp_path))

    load_app_environment(env_file, override=True)
    config = AppConfig.load(str(config_file))

    assert config.discord_token == "env-token"


def test_app_config_reads_json_config_and_secret_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "token-from-env")
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "secret-key")
    config_file = write_config(tmp_path, build_config_payload(tmp_path))

    config = AppConfig.load(str(config_file))

    assert config.discord_token == "token-from-env"
    assert config.llama_cpp_api_key == "secret-key"
    assert config.inference.timeout_seconds == 420
    assert config.inference.temperature == 0.3


def test_app_config_rejects_invalid_base_url(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "token")
    payload = build_config_payload(tmp_path)
    payload["inference"]["base_url"] = "localhost:8080"
    config_file = write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="inference.base_url"):
        AppConfig.load(str(config_file))


def test_app_config_requires_model_path_for_bundled_llama_server(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "token")
    payload = build_config_payload(tmp_path)
    payload["llama_server"]["enabled"] = True
    config_file = write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="llama_server.model_path"):
        AppConfig.load(str(config_file))


def test_resolve_data_directory_uses_configured_path(tmp_path) -> None:
    configured = tmp_path / "peter-state"
    resolved = Path(resolve_data_directory(str(configured)))
    assert resolved == configured.resolve()
    assert resolved.is_dir()


def test_resolve_data_directory_falls_back_when_configured_path_fails(tmp_path, monkeypatch) -> None:
    default_dir = (tmp_path / "default").resolve()
    configured = tmp_path / "broken"
    original_makedirs = Path.mkdir

    def fake_makedirs(path, exist_ok=False):  # type: ignore[no-untyped-def]
        target = Path(path).resolve()
        if target == configured.resolve():
            raise OSError("permission denied")
        original_makedirs(target, parents=True, exist_ok=exist_ok)

    monkeypatch.setattr("peterbot.config.os.makedirs", fake_makedirs)
    monkeypatch.setattr("peterbot.config.PROJECT_ROOT", default_dir.parent)

    resolved = Path(resolve_data_directory(str(configured)))
    assert resolved == (default_dir.parent / "peterbot-data").resolve()
    assert resolved.is_dir()


def test_build_system_prompt_layers_qwen_rules_channel_profile_and_knowledge(tmp_path) -> None:
    config = build_config(tmp_path)
    channel_profiles = load_channel_profiles(config.channel_profiles_file)
    knowledge_chunks = load_knowledge_chunks(config.knowledge_file)
    ranked_chunks = rank_knowledge_chunks(
        "when is the next meeting for the club",
        knowledge_chunks,
        channel_profile=channel_profiles["1234"],
    )

    prompt = build_system_prompt(
        config,
        build_context_line(author_name="Taylor", guild_name="CHC", channel_name="general"),
        mode=MENTION_MODE,
        focus_note="This is the immediate reply target.",
        channel_profile=channel_profiles["1234"],
        knowledge_chunks=ranked_chunks,
    )

    assert "Identity: Your name is Peter." in prompt
    assert "You are the club bot or assistant, not a human member of the server." in prompt
    assert "Use one short paragraph by default." in prompt
    assert "Do not ask a follow up question unless clarification is actually required." in prompt
    assert "Do not use hyphen, en dash, or em dash punctuation in normal reply prose." in prompt
    assert "Focused context: This is the immediate reply target." in prompt
    assert "Channel profile:" in prompt
    assert "Relevant club knowledge:" in prompt
    assert "We meet every Thursday at 6:30 PM" in prompt
    assert "real person chatting" not in prompt


def test_cleanup_response_text_strips_qwen_canned_phrasing() -> None:
    raw = "Absolutely, here's a quick summary...\n\nThat build is still solid!!!\n\nHope that helps."
    cleaned = cleanup_response_text(raw, profile=ModelProfile.QWEN)
    assert cleaned == "That build is still solid!"


def test_cleanup_response_text_keeps_short_affirmation_when_cleanup_would_empty_it() -> None:
    assert cleanup_response_text("Sure.", profile=ModelProfile.QWEN) == "Sure."


def test_build_knowledge_excerpt_caps_large_sections() -> None:
    chunks = load_knowledge_chunks(str(FIXTURES / "club_knowledge.md"))
    excerpt = build_knowledge_excerpt(chunks, max_chars=60)
    assert excerpt is not None
    assert len(excerpt) <= 60


def test_cleanup_response_text_removes_fake_human_identity_tail() -> None:
    fixture = PERSONALITY_FIXTURES["identity_response"]
    cleaned = cleanup_response_text(fixture["raw"], profile=ModelProfile.QWEN)
    assert cleaned == fixture["expected"]
    assert "-" not in cleaned


def test_cleanup_response_text_normalizes_simple_greeting() -> None:
    fixture = PERSONALITY_FIXTURES["hello_response"]
    assert cleanup_response_text(fixture["raw"], profile=ModelProfile.QWEN) == fixture["expected"]


def test_cleanup_response_text_removes_social_second_paragraph() -> None:
    fixture = PERSONALITY_FIXTURES["mention_response"]
    assert cleanup_response_text(fixture["raw"], profile=ModelProfile.QWEN) == fixture["expected"]


def test_cleanup_response_text_replaces_dashes_but_preserves_literals() -> None:
    raw = "Read https://example.com/foo-bar and `my-file.py` for the follow-up - it explains the club-bot setup."
    cleaned = cleanup_response_text(raw, profile=ModelProfile.QWEN)
    assert "https://example.com/foo-bar" in cleaned
    assert "`my-file.py`" in cleaned
    assert "follow up" in cleaned
    assert "club bot" in cleaned
    assert " - " not in cleaned
    assert "—" not in cleaned
    assert "–" not in cleaned


def test_add_no_think_suffix_is_noop_for_llama_cpp_requests() -> None:
    assert add_no_think_suffix("What time is the meeting?") == "What time is the meeting?"


def test_custom_persona_seed_cannot_override_hard_style_rules(tmp_path) -> None:
    config = build_config(tmp_path)
    config = AppConfig(
        discord_token=config.discord_token,
        llama_cpp_api_key=config.llama_cpp_api_key,
        persona=PersonaConfig(
            name="Peter",
            system_prompt="You are Peter, a super warm best friend who acts human.",
            model_profile=ModelProfile.QWEN,
        ),
        discord=config.discord,
        inference=config.inference,
        llama_server=config.llama_server,
        paths=config.paths,
        logging=config.logging,
        behavior=config.behavior,
        config_path=config.config_path,
    )
    prompt = build_system_prompt(
        config,
        build_context_line(author_name="Oliver", guild_name="CHC", channel_name="general"),
        mode=MENTION_MODE,
    )
    assert "super warm best friend who acts human" in prompt
    assert "You are the club bot or assistant, not a human member of the server." in prompt
    assert "Do not ask a follow up question unless clarification is actually required." in prompt
