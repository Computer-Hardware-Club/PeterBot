from datetime import datetime, timedelta, timezone

import pytest

from peterbot.conversation_store import ConversationStore


def add(store, source, *, guild=1, user=2, channel=3, audience="public", prompt="hello", answer="hi"):
    return store.append_turn(guild_id=guild, user_id=user, channel_id=channel,
                             source_message_id=source, audience=audience,
                             prompt=prompt, answer=answer)


def context(store, *, guild=1, user=2, channel=3, audience="public", max_chars=8000):
    return store.context(guild_id=guild, user_id=user, channel_id=channel,
                         audience=audience, max_chars=max_chars)


def test_restart_scope_and_idempotent_ingress(tmp_path):
    path = tmp_path / "conversation.sqlite3"
    store = ConversationStore(str(path))
    assert add(store, 4, prompt="Please keep the board private", answer="Understood")
    assert not add(store, 4, prompt="duplicate", answer="duplicate")
    assert add(store, 5, audience="private", prompt="secret design", answer="private answer")
    assert add(store, 6, user=9, prompt="other user's work", answer="other answer")
    assert add(store, 7, guild=8, prompt="other guild", answer="other guild answer")
    assert add(store, 8, channel=10, prompt="other channel", answer="other channel answer")
    store.db.close()
    reopened = ConversationStore(str(path))
    visible = str(context(reopened))
    assert "keep the board private" in visible
    for forbidden in ("secret design", "other user's work", "other guild", "other channel", "duplicate"):
        assert forbidden not in visible
    assert "secret design" in str(context(reopened, audience="private"))


def test_compaction_keeps_literal_constraints_and_bounded_context(tmp_path):
    store = ConversationStore(str(tmp_path / "conversation.sqlite3"))
    add(store, 1, prompt="Build a controller. Do not publish it yet.", answer="Starting.")
    for index in range(2, 22):
        add(store, index, prompt=f"step {index}: " + "x" * 1500,
            answer=f"completed step {index}: " + "y" * 1500)
    result = context(store, max_chars=3500)
    serialized = str(result)
    assert "Do not publish it yet" in serialized
    assert "step 21" in serialized
    assert sum(len(item["content"]) for item in result) <= 3500
    assert "step 2:" not in serialized


def test_no_unfinished_or_reasoning_entries_and_retention(tmp_path):
    store = ConversationStore(str(tmp_path / "conversation.sqlite3"))
    with pytest.raises(ValueError):
        add(store, 1, answer="")
    with pytest.raises(ValueError):
        add(store, 2, audience="everyone")
    add(store, 3, prompt="Do this", answer="Done")
    assert "Done" in str(context(store))
    with pytest.raises(ValueError):
        store.delete_before(datetime.now())
    assert store.delete_before(datetime.now(timezone.utc) + timedelta(days=1)) == 1
    assert context(store) == []


def test_unknown_future_schema_is_rejected(tmp_path):
    path = tmp_path / "conversation.sqlite3"
    store = ConversationStore(str(path))
    store.db.execute("PRAGMA user_version=200")
    store.db.close()
    with pytest.raises(ValueError, match="newer"):
        ConversationStore(str(path))
