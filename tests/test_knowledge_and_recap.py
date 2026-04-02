from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from peterbot.commands import build_prompt_artifacts, clamp_recap_count
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
)
from peterbot.context import build_recap_history
from peterbot.knowledge import (
    load_knowledge_index,
    load_channel_profiles,
    load_knowledge_chunks,
    rank_knowledge_chunks,
    resolve_channel_profile,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_knowledge_ranking_finds_meeting_section() -> None:
    chunks = load_knowledge_chunks(str(FIXTURES / "club_knowledge.md"))
    ranked = rank_knowledge_chunks("What time is the club meeting on Thursday?", chunks)
    assert ranked
    assert ranked[0].heading == "Meetings"


def test_resolve_channel_profile_by_name_and_id() -> None:
    profiles = load_channel_profiles(str(FIXTURES / "channel_profiles.json"))
    assert resolve_channel_profile(SimpleNamespace(id=999, name="hardware-help"), profiles) == profiles["hardware-help"]
    assert resolve_channel_profile(SimpleNamespace(id=1234, name="general"), profiles) == profiles["1234"]


def test_build_recap_history_formats_authors_and_relative_age() -> None:
    current_time = datetime(2026, 3, 10, 12, 0, 0)
    entries = [
        {
            "author_name": "Maya",
            "role": "user",
            "content": "we should move the meeting to lab 2",
            "created_at": datetime(2026, 3, 10, 11, 55, 0),
        },
        {
            "author_name": "Peter",
            "role": "assistant",
            "content": "lab 2 is open after 6",
            "created_at": datetime(2026, 3, 10, 11, 57, 0),
        },
    ]
    history = build_recap_history(entries, current_time)
    assert history[0]["content"].startswith("[5 minutes ago] Maya:")
    assert history[1]["role"] == "assistant"


def test_clamp_recap_count_enforces_bounds() -> None:
    assert clamp_recap_count(2, 40) == 5
    assert clamp_recap_count(60, 40) == 40
    assert clamp_recap_count(25, 40) == 25


def test_vague_prompt_does_not_pull_knowledge_from_channel_topics_alone() -> None:
    chunks = load_knowledge_chunks(str(FIXTURES / "club_knowledge.md"))
    profiles = load_channel_profiles(str(FIXTURES / "channel_profiles.json"))
    ranked = rank_knowledge_chunks("thanks", chunks, channel_profile=profiles["hardware-help"])
    assert ranked == []


def test_recap_prompt_artifacts_skip_channel_profile_and_knowledge(tmp_path) -> None:
    config = AppConfig(
        discord_token="token",
        llama_cpp_api_key=None,
        persona=PersonaConfig(
            name="Peter",
            system_prompt="You are Peter.",
            model_profile=ModelProfile.QWEN,
        ),
        discord=DiscordConfig(suggestion_channel_id=None),
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
    knowledge_index = load_knowledge_index(
        knowledge_file=config.knowledge_file,
        channel_profiles_file=config.channel_profiles_file,
    )
    system_prompt, knowledge_chunks = build_prompt_artifacts(
        config=config,
        knowledge_index=knowledge_index,
        prompt_text="Summarize the recent discussion.",
        author_name="Maya",
        guild_name="CHC",
        channel=SimpleNamespace(id=1234, name="hardware-help"),
        mode="recap",
        include_channel_profile=False,
        include_knowledge=False,
    )
    assert knowledge_chunks == []
    assert "Relevant club knowledge:" not in system_prompt
    assert "Channel profile:" not in system_prompt
