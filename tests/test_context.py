import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from peterbot.context import (
    build_current_mention_prompt_text,
    build_mention_context_bundle,
    prompt_requires_strong_target,
)


FIXTURES = Path(__file__).parent / "fixtures" / "mention_scenarios.json"


def load_scenarios():
    raw = json.loads(FIXTURES.read_text(encoding="utf-8"))
    for scenario in raw.values():
        for entry in scenario["entries"]:
            entry["created_at"] = datetime.fromisoformat(entry["created_at"])
        scenario["entries"].sort(key=lambda entry: entry["created_at"])
        scenario["current_time"] = datetime.fromisoformat(scenario["current_time"])
    return raw


def build_message(prompt_text: str, current_time: datetime, *, attachments=None, display_name="Avery"):
    return SimpleNamespace(
        content=prompt_text,
        attachments=attachments or [],
        author=SimpleNamespace(display_name=display_name),
        created_at=current_time,
    )


def test_prompt_requires_strong_target_for_vague_language() -> None:
    assert prompt_requires_strong_target("thoughts?")
    assert prompt_requires_strong_target("What do you think about that?")
    assert not prompt_requires_strong_target("Should we keep the 5800x3d build as-is?")


def test_vague_mention_targets_recent_turn_instead_of_stale_history() -> None:
    scenarios = load_scenarios()
    scenario = scenarios["recent_vague_turn"]
    bundle = build_mention_context_bundle(
        build_message(scenario["prompt"], scenario["current_time"]),
        scenario["prompt"],
        scenario["entries"],
        focus_message_limit=6,
        active_gap_minutes=10,
        max_background_age_minutes=45,
        assistant_tail_limit=2,
    )

    history_text = " ".join(item["content"] for item in bundle["conversation_history"])
    assert bundle["selection_reason"] == "recent_peter_exchange"
    assert "look how many periods" in history_text
    assert "meeting parts budget" not in history_text


def test_ambiguous_recent_turns_require_clarification() -> None:
    scenarios = load_scenarios()
    scenario = scenarios["ambiguous_recent_turns"]
    bundle = build_mention_context_bundle(
        build_message(scenario["prompt"], scenario["current_time"]),
        scenario["prompt"],
        scenario["entries"],
        focus_message_limit=6,
        active_gap_minutes=10,
        max_background_age_minutes=45,
        assistant_tail_limit=2,
    )

    assert bundle["clarification_text"] == "Which message are you asking me about?"
    assert bundle["conversation_history"] == []


def test_direct_reply_stays_anchored_to_explicit_target() -> None:
    scenarios = load_scenarios()
    scenario = scenarios["direct_reply"]
    explicit_reply_entry = next(
        entry for entry in scenario["entries"] if entry["message_id"] == scenario["reply_target_message_id"]
    )
    bundle = build_mention_context_bundle(
        build_message(scenario["prompt"], scenario["current_time"]),
        scenario["prompt"],
        scenario["entries"],
        focus_message_limit=6,
        active_gap_minutes=10,
        max_background_age_minutes=45,
        assistant_tail_limit=2,
        explicit_reply_entry=explicit_reply_entry,
    )

    assert bundle["selection_reason"] == "explicit_reply"
    assert bundle["target_message_id"] == 30
    assert "5800x3d" in " ".join(item["content"] for item in bundle["conversation_history"])


def test_unrelated_recent_peter_reply_is_not_appended_to_focus_history() -> None:
    current_time = datetime(2026, 3, 10, 13, 0, 0)
    entries = [
        {
            "message_id": 40,
            "author_id": 501,
            "author_name": "Lena",
            "role": "user",
            "content": "the PSU swap seems worth it",
            "created_at": datetime(2026, 3, 10, 12, 54, 0),
            "reply_to_message_id": None,
        },
        {
            "message_id": 41,
            "author_id": 999,
            "author_name": "Peter",
            "role": "assistant",
            "content": "yeah, that upgrade makes sense",
            "created_at": datetime(2026, 3, 10, 12, 55, 0),
            "reply_to_message_id": 40,
        },
        {
            "message_id": 42,
            "author_id": 502,
            "author_name": "Owen",
            "role": "user",
            "content": "also the meetup snacks were good",
            "created_at": datetime(2026, 3, 10, 12, 58, 0),
            "reply_to_message_id": None,
        },
        {
            "message_id": 43,
            "author_id": 999,
            "author_name": "Peter",
            "role": "assistant",
            "content": "the chips disappeared instantly",
            "created_at": datetime(2026, 3, 10, 12, 59, 0),
            "reply_to_message_id": 42,
        },
    ]
    bundle = build_mention_context_bundle(
        build_message("thoughts on the PSU swap?", current_time),
        "thoughts on the PSU swap?",
        entries,
        focus_message_limit=6,
        active_gap_minutes=10,
        max_background_age_minutes=45,
        assistant_tail_limit=2,
    )
    history_text = " ".join(item["content"] for item in bundle["conversation_history"])
    assert "upgrade makes sense" in history_text
    assert "chips disappeared instantly" not in history_text


def test_image_only_mention_builds_natural_prompt_and_attachment_note() -> None:
    attachment = SimpleNamespace(filename="case-photo.png", content_type="image/png")
    message = SimpleNamespace(content="", attachments=[attachment])
    prompt = build_current_mention_prompt_text(message, bot_user_id=999)
    assert prompt.startswith("What do you think about this?")
    assert "[attachments: case-photo.png]" in prompt
