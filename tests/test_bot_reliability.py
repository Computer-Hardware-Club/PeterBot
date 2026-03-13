import ast
import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


BOT_PATH = Path(__file__).resolve().parents[1] / "bot.py"
BOT_SOURCE = BOT_PATH.read_text(encoding="utf-8")
BOT_AST = ast.parse(BOT_SOURCE)


def load_definitions(*names: str, extra_globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {
        "__file__": str(BOT_PATH),
        "logging": logging,
        "json": json,
        "os": os,
        "re": re,
        "tempfile": tempfile,
        "datetime": datetime,
        "timedelta": timedelta,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "REMINDER_RETRY_DELAY": timedelta(minutes=5),
        "MAX_LOG_CONTEXT_CHARS": 320,
        "logger": logging.getLogger("test-peterbot"),
        "INCLUDE_TRACEBACK_FOR_WARNING": False,
        "USER_DEBUG_IDS_ENABLED": True,
        "truncate_for_log": lambda value, max_chars=320: str(value),
        "log_with_context": lambda level, message, **context: None,
        "log_exception_with_context": lambda action, **context: "ERR-test",
        "build_user_debug_message": lambda base_message, debug_id: base_message,
    }
    if extra_globals:
        namespace.update(extra_globals)

    remaining = set(names)
    for node in BOT_AST.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in remaining:
            continue

        source = ast.get_source_segment(BOT_SOURCE, node)
        if source is None:
            raise AssertionError(f"Could not load definition source for {node.name}")
        exec(source, namespace)  # noqa: S102 - controlled test fixture execution
        remaining.remove(node.name)

    if remaining:
        raise AssertionError(f"Missing definitions in bot.py: {sorted(remaining)}")

    return namespace


def test_parse_reminder_time_core_formats() -> None:
    ns = load_definitions("add_one_year", "normalize_2_digit_year", "parse_reminder_time")
    parse_reminder_time = ns["parse_reminder_time"]

    now = datetime(2026, 3, 10, 12, 0, 0)

    assert parse_reminder_time("in 45 minutes", now=now) == datetime(2026, 3, 10, 12, 45, 0)
    assert parse_reminder_time("tomorrow at 9:15 PM", now=now) == datetime(
        2026, 3, 11, 21, 15, 0
    )
    assert parse_reminder_time("03/10/26 2:30 PM", now=now) == datetime(2026, 3, 10, 14, 30, 0)
    assert parse_reminder_time("in 0 minutes", now=now) is None


def test_resolve_data_directory_uses_env_var(tmp_path, monkeypatch) -> None:
    ns = load_definitions("resolve_data_directory")
    resolve_data_directory = ns["resolve_data_directory"]

    configured = tmp_path / "peter-state"
    monkeypatch.setenv("PETERBOT_DATA_DIR", str(configured))
    resolved = Path(resolve_data_directory())

    assert resolved == configured.resolve()
    assert resolved.is_dir()


def test_reminder_manager_uses_stable_data_directory(tmp_path, monkeypatch) -> None:
    ns = load_definitions("resolve_data_directory", "write_json_atomic", "ReminderManager")
    reminder_manager_cls = ns["ReminderManager"]

    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    data_dir = tmp_path / "data"
    manager = reminder_manager_cls(data_dir=str(data_dir))
    manager.add_reminder(
        user_id=42,
        message="check persistence path",
        remind_time=datetime(2026, 3, 12, 9, 30, 0),
    )

    assert Path(manager.reminders_file) == (data_dir / "reminders.json")
    assert Path(manager.reminders_file).exists()
    assert not (cwd_dir / "reminders.json").exists()


def test_atomic_save_and_load_round_trip(tmp_path) -> None:
    ns = load_definitions("resolve_data_directory", "write_json_atomic", "ReminderManager")
    reminder_manager_cls = ns["ReminderManager"]
    manager = reminder_manager_cls(data_dir=str(tmp_path))

    manager.add_reminder(
        user_id=7,
        message="round trip reminder",
        remind_time=datetime(2026, 3, 15, 10, 0, 0),
    )

    created_files = sorted(path.name for path in tmp_path.iterdir())
    assert created_files == ["reminders.json"]

    reloaded = reminder_manager_cls(data_dir=str(tmp_path))
    reloaded.load_reminders()

    assert len(reloaded.reminders) == 1
    assert reloaded.reminders[0]["user_id"] == 7
    assert reloaded.reminders[0]["message"] == "round trip reminder"
    assert reloaded.reminders[0]["remind_time"] == datetime(2026, 3, 15, 10, 0, 0)


def test_legacy_fallback_reads_old_files_once(tmp_path, monkeypatch) -> None:
    ns = load_definitions("resolve_data_directory", "write_json_atomic", "ReminderManager")
    reminder_manager_cls = ns["ReminderManager"]

    legacy_dir = tmp_path / "legacy"
    data_dir = tmp_path / "new-data"
    legacy_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.chdir(legacy_dir)

    legacy_reminders = [
        {
            "user_id": 123,
            "message": "legacy reminder",
            "remind_time": "2026-03-20T09:00:00",
            "created_at": "2026-03-10T08:00:00",
        }
    ]
    (legacy_dir / "reminders.json").write_text(json.dumps(legacy_reminders), encoding="utf-8")

    shutdown_time = (datetime.now() - timedelta(minutes=2)).isoformat()
    (legacy_dir / "bot_shutdown.json").write_text(
        json.dumps({"shutdown_time": shutdown_time}),
        encoding="utf-8",
    )

    manager = reminder_manager_cls(data_dir=str(data_dir))
    manager.load_reminders()
    downtime = manager.get_downtime()

    assert len(manager.reminders) == 1
    assert manager.reminders[0]["message"] == "legacy reminder"
    assert downtime is not None
    assert not (legacy_dir / "bot_shutdown.json").exists()


def test_safe_send_interaction_message_uses_response_then_followup() -> None:
    class DiscordStub:
        class Interaction:  # pragma: no cover - type annotation placeholder
            pass

    ns = load_definitions(
        "safe_send_interaction_message",
        extra_globals={"discord": DiscordStub},
    )
    safe_send_interaction_message = ns["safe_send_interaction_message"]

    class FakeResponse:
        def __init__(self, done: bool = False) -> None:
            self._done = done
            self.messages: List[Tuple[str, bool]] = []

        def is_done(self) -> bool:
            return self._done

        async def send_message(self, text: str, ephemeral: bool = True) -> None:
            self.messages.append((text, ephemeral))
            self._done = True

    class FakeFollowup:
        def __init__(self) -> None:
            self.messages: List[Tuple[str, bool]] = []

        async def send(self, text: str, ephemeral: bool = True) -> None:
            self.messages.append((text, ephemeral))

    async def run_scenario(initial_done: bool) -> SimpleNamespace:
        interaction = SimpleNamespace(
            response=FakeResponse(done=initial_done),
            followup=FakeFollowup(),
        )
        await safe_send_interaction_message(interaction, "hello", ephemeral=True)
        return interaction

    first = asyncio.run(run_scenario(initial_done=False))
    assert first.response.messages == [("hello", True)]
    assert first.followup.messages == []

    second = asyncio.run(run_scenario(initial_done=True))
    assert second.response.messages == []
    assert second.followup.messages == [("hello", True)]


def load_mention_context_defs(*extra_names: str) -> Dict[str, Any]:
    common_names = (
        "build_context_line",
        "build_mention_system_prompt",
        "format_relative_age",
        "prompt_requires_strong_target",
        "find_message_entry_index",
        "build_recent_tail_entries",
        "collect_message_cluster",
        "count_distinct_recent_user_authors",
        "extract_relevance_tokens",
        "score_focus_candidate",
        "entries_are_linked",
        "collect_focus_thread",
        "trim_focus_entries",
        "build_mention_conversation_history",
        "build_mention_user_content",
        "select_mention_focus_target",
        "build_mention_focus_note",
        "build_mention_context_bundle",
        "add_no_think_suffix",
        "build_chat_messages",
    )
    return load_definitions(
        *(common_names + extra_names),
        extra_globals={
            "MENTION_ACTIVE_GAP_MINUTES": 10,
            "MENTION_MAX_BACKGROUND_AGE_MINUTES": 45,
            "MENTION_FOCUS_MESSAGE_LIMIT": 6,
            "PETER_NAME": "Peter",
            "PETER_SYSTEM_PROMPT": "You are Peter.",
            "bot": SimpleNamespace(user=SimpleNamespace(id=999, display_name="Peter")),
        },
    )


def load_image_defs(*extra_names: str) -> Dict[str, Any]:
    class DiscordStub:
        class Message:  # pragma: no cover - type annotation placeholder
            pass

        class HTTPException(Exception):
            pass

    return load_definitions(
        "strip_bot_mentions",
        "is_image_attachment",
        "build_current_mention_prompt_text",
        "load_mention_image_payloads",
        "add_no_think_suffix",
        "build_chat_messages",
        *extra_names,
        extra_globals={
            "base64": base64,
            "MENTION_IMAGE_LIMIT": 2,
            "MENTION_MAX_IMAGE_BYTES": 5 * 1024 * 1024,
            "bot": SimpleNamespace(user=SimpleNamespace(id=999, display_name="Peter")),
            "discord": DiscordStub,
        },
    )


def make_context_entry(
    *,
    message_id: int,
    author_name: str,
    role: str,
    content: str,
    created_at: datetime,
    reply_to_message_id: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "message_id": message_id,
        "author_id": message_id + 1000,
        "author_name": author_name,
        "role": role,
        "content": content,
        "created_at": created_at,
        "reply_to_message_id": reply_to_message_id,
    }


def make_current_message(author_name: str, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        author=SimpleNamespace(display_name=author_name, name=author_name),
        created_at=created_at,
    )


def test_vague_mention_targets_recent_active_turn_instead_of_old_history() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)
    recent_entries = [
        make_context_entry(
            message_id=1,
            author_name="Wil",
            role="user",
            content="I have a 5800x3d and the 1000W is actually somewhat necessary.",
            created_at=datetime(2026, 3, 12, 17, 41, 0),
        ),
        make_context_entry(
            message_id=2,
            author_name="Peter",
            role="assistant",
            content="Older unrelated reply about meetings and PSU pricing.",
            created_at=datetime(2026, 3, 13, 9, 0, 0),
        ),
        make_context_entry(
            message_id=3,
            author_name="Levi",
            role="user",
            content="look how many periods he uses now and how short his sentences are",
            created_at=datetime(2026, 3, 13, 9, 59, 0),
        ),
    ]

    bundle = build_mention_context_bundle(
        current_message,
        "what do you think about that",
        recent_entries,
    )

    history_text = " ".join(item["content"] for item in bundle["conversation_history"])
    assert bundle["selection_reason"] == "immediate_previous_turn"
    assert bundle["target_message_id"] == 3
    assert bundle["clarification_text"] is None
    assert "look how many periods" in history_text
    assert "5800x3d" not in history_text
    assert "meetings and PSU pricing" not in history_text


def test_explicit_reply_target_overrides_newer_chatter() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)
    explicit_target = make_context_entry(
        message_id=10,
        author_name="Wil",
        role="user",
        content="The PSU upgrade is worth it because the old unit is ten years old.",
        created_at=datetime(2026, 3, 13, 9, 40, 0),
    )
    recent_entries = [
        explicit_target,
        make_context_entry(
            message_id=11,
            author_name="Peter",
            role="assistant",
            content="I would replace it too.",
            created_at=datetime(2026, 3, 13, 9, 41, 0),
        ),
        make_context_entry(
            message_id=12,
            author_name="Levi",
            role="user",
            content="Unrelated side chatter about music taste.",
            created_at=datetime(2026, 3, 13, 9, 59, 0),
        ),
    ]

    bundle = build_mention_context_bundle(
        current_message,
        "what do you think about that",
        recent_entries,
        explicit_reply_entry=explicit_target,
    )

    history_text = " ".join(item["content"] for item in bundle["conversation_history"])
    assert bundle["selection_reason"] == "explicit_reply"
    assert bundle["target_message_id"] == 10
    assert "PSU upgrade is worth it" in history_text
    assert "music taste" not in history_text


def test_stale_backlog_is_dropped_from_focus_context() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)
    recent_entries = [
        make_context_entry(
            message_id=20,
            author_name="Wil",
            role="user",
            content="Old backlog about PSUs from earlier this morning.",
            created_at=datetime(2026, 3, 13, 7, 30, 0),
        )
    ]

    bundle = build_mention_context_bundle(
        current_message,
        "I think a 1000W PSU is probably enough here",
        recent_entries,
    )

    assert bundle["selection_reason"] == "no_target"
    assert bundle["target_message_id"] is None
    assert bundle["selected_count"] == 0
    assert bundle["clarification_text"] is None
    assert bundle["conversation_history"] == []


def test_vague_mention_without_anchor_asks_for_clarification() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)

    bundle = build_mention_context_bundle(current_message, "thoughts?", [])

    assert bundle["selection_reason"] == "no_target"
    assert bundle["clarification_text"] == "Which message are you asking me about?"
    assert bundle["selected_count"] == 0


def test_vague_mention_with_interleaved_recent_users_asks_for_clarification() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)
    recent_entries = [
        make_context_entry(
            message_id=25,
            author_name="Wil",
            role="user",
            content="The PSU thing is mostly about transient spikes.",
            created_at=datetime(2026, 3, 13, 9, 58, 0),
        ),
        make_context_entry(
            message_id=26,
            author_name="Levi",
            role="user",
            content="Scott still has good taste in music though.",
            created_at=datetime(2026, 3, 13, 9, 59, 0),
        ),
    ]

    bundle = build_mention_context_bundle(
        current_message,
        "what do you think about that",
        recent_entries,
    )

    assert bundle["selection_reason"] == "ambiguous_recent_turns"
    assert bundle["target_message_id"] is None
    assert bundle["clarification_text"] == "Which message are you asking me about?"
    assert bundle["conversation_history"] == []


def test_mention_prompt_assembly_marks_current_turn_and_freshness() -> None:
    ns = load_mention_context_defs()
    build_mention_context_bundle = ns["build_mention_context_bundle"]
    build_context_line = ns["build_context_line"]
    build_mention_system_prompt = ns["build_mention_system_prompt"]
    build_chat_messages = ns["build_chat_messages"]

    now = datetime(2026, 3, 13, 10, 0, 0)
    current_message = make_current_message("Oliver", now)
    recent_entries = [
        make_context_entry(
            message_id=30,
            author_name="Levi",
            role="user",
            content="look how many periods he uses now",
            created_at=datetime(2026, 3, 13, 9, 58, 0),
        ),
        make_context_entry(
            message_id=31,
            author_name="Peter",
            role="assistant",
            content="Oliver. But Scott's got good taste in music.",
            created_at=datetime(2026, 3, 13, 9, 59, 0),
            reply_to_message_id=30,
        ),
    ]

    bundle = build_mention_context_bundle(
        current_message,
        "what do you think about that",
        recent_entries,
    )
    context_line = build_context_line(
        author_name="Oliver",
        guild_name="Hardware Club",
        channel_name="general",
    )
    system_prompt = build_mention_system_prompt(context_line, focus_note=bundle["focus_note"])
    messages = build_chat_messages(
        "what do you think about that",
        author_name="Oliver",
        conversation_history=bundle["conversation_history"],
        system_prompt=system_prompt,
        user_content=bundle["user_content"],
    )

    assert "Focused context:" in messages[0]["content"]
    assert "Reply to the newest addressed turn first." in messages[0]["content"]
    assert any("[2 minutes ago] Levi:" in item["content"] for item in messages[1:-1])
    assert any("[1 minute ago] Peter" in item["content"] for item in messages[1:-1])
    assert messages[-1]["content"].startswith(
        "[Current message | now] Oliver: what do you think about that"
    )
    assert messages[-1]["content"].endswith("/no_think")


def test_image_only_mention_builds_prompt_and_attaches_images_to_user_message() -> None:
    ns = load_image_defs()
    build_current_mention_prompt_text = ns["build_current_mention_prompt_text"]
    load_mention_image_payloads = ns["load_mention_image_payloads"]
    build_chat_messages = ns["build_chat_messages"]

    class FakeAttachment:
        def __init__(self, filename: str, content_type: str, data: bytes) -> None:
            self.filename = filename
            self.content_type = content_type
            self.size = len(data)
            self._data = data

        async def read(self) -> bytes:
            return self._data

    attachment = FakeAttachment("case-photo.png", "image/png", b"abc123")
    message = SimpleNamespace(
        content="<@999>",
        attachments=[attachment],
        author=SimpleNamespace(display_name="Oliver", name="Oliver"),
        id=123,
        channel=SimpleNamespace(id=456),
        guild=SimpleNamespace(id=789),
    )

    prompt_text = build_current_mention_prompt_text(message)
    encoded_images = asyncio.run(load_mention_image_payloads(message))
    messages = build_chat_messages(
        prompt_text,
        author_name="Oliver",
        conversation_history=[],
        system_prompt="You are Peter.",
        user_content=f"[Current message | now] Oliver: {prompt_text}",
        user_images=encoded_images,
    )

    assert prompt_text.startswith("What do you think about this?")
    assert "[attachments: case-photo.png]" in prompt_text
    assert messages[-1]["images"] == [base64.b64encode(b"abc123").decode("ascii")]
