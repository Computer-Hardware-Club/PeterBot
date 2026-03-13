import json
from pathlib import Path

from peterbot.config import AppConfig, ModelProfile, resolve_data_directory, resolve_model_profile
from peterbot.knowledge import build_knowledge_excerpt, load_channel_profiles, load_knowledge_chunks, rank_knowledge_chunks
from peterbot.prompts import MENTION_MODE, build_context_line, build_system_prompt, cleanup_response_text


FIXTURES = Path(__file__).parent / "fixtures"
PERSONALITY_FIXTURES = json.loads((FIXTURES / "personality_cleanup.json").read_text(encoding="utf-8"))


def build_config(tmp_path) -> AppConfig:
    return AppConfig(
        discord_token="token",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3.5",
        peter_name="Peter",
        peter_system_prompt="You are Peter.",
        ollama_think=False,
        model_profile=ModelProfile.QWEN,
        ollama_options={"temperature": 0.3},
        suggestion_channel_id=123,
        data_dir=str(tmp_path),
        knowledge_file=str(FIXTURES / "club_knowledge.md"),
        channel_profiles_file=str(FIXTURES / "channel_profiles.json"),
        log_level="INFO",
        log_file="",
        user_debug_ids_enabled=True,
        include_traceback_for_warning=False,
    )


def test_resolve_model_profile_auto_prefers_qwen() -> None:
    assert resolve_model_profile("auto", "qwen3.5:14b") == ModelProfile.QWEN
    assert resolve_model_profile("auto", "ministral-3:8b") == ModelProfile.GENERIC


def test_resolve_data_directory_uses_env_var(tmp_path, monkeypatch) -> None:
    configured = tmp_path / "peter-state"
    monkeypatch.setenv("PETERBOT_DATA_DIR", str(configured))
    resolved = Path(resolve_data_directory())
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
    monkeypatch.setattr("peterbot.config.os.path.dirname", lambda _: str(default_dir / "pkg"))

    resolved = Path(resolve_data_directory(str(configured)))
    assert resolved == default_dir
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


def test_custom_persona_seed_cannot_override_hard_style_rules(tmp_path) -> None:
    config = AppConfig(
        discord_token="token",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3.5",
        peter_name="Peter",
        peter_system_prompt="You are Peter, a super warm best friend who acts human.",
        ollama_think=False,
        model_profile=ModelProfile.QWEN,
        ollama_options={},
        suggestion_channel_id=None,
        data_dir=str(tmp_path),
        knowledge_file=None,
        channel_profiles_file=None,
        log_level="INFO",
        log_file="",
        user_debug_ids_enabled=True,
        include_traceback_for_warning=False,
    )
    prompt = build_system_prompt(
        config,
        build_context_line(author_name="Oliver", guild_name="CHC", channel_name="general"),
        mode=MENTION_MODE,
    )
    assert "super warm best friend who acts human" in prompt
    assert "You are the club bot or assistant, not a human member of the server." in prompt
    assert "Do not ask a follow up question unless clarification is actually required." in prompt
