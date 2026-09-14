import json

import pytest

from peterbot.agent_jobs import JobStore


@pytest.fixture
def store(tmp_path):
    value = JobStore(str(tmp_path / "jobs.sqlite"))
    yield value
    value.close()


def create(store, user_id=1, guild_id=10, **kwargs):
    return store.create(guild_id=guild_id, user_id=user_id, channel_id=20,
                        source_message_id=30, prompt="Research hardware options", **kwargs)


def test_restart_interrupts_running_but_preserves_queued_and_finished(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    queued = create(first)
    running = create(first)
    completed = create(first, user_id=2)
    first.update(running["id"], status="running")
    first.update(completed["id"], status="completed", answer="saved answer",
                 artifacts=[{"path": "report.md"}], delivered=True)
    first.close()

    restarted = JobStore(path)
    try:
        assert restarted.get(queued["id"])["status"] == "queued"
        interrupted = restarted.get(running["id"])
        assert interrupted["status"] == "interrupted"
        assert "restarted" in interrupted["answer"]
        assert interrupted["prompt"] == running["prompt"]
        saved = restarted.get(completed["id"])
        assert saved["status"] == "completed"
        assert saved["answer"] == "saved answer"
        assert json.loads(saved["artifacts"]) == [{"path": "report.md"}]
        assert saved["delivered"] == 1
        assert [j["id"] for j in restarted.pending()] == [queued["id"]]
        assert [j["id"] for j in restarted.undelivered()] == [running["id"]]
    finally:
        restarted.close()


def test_owned_and_list_owned_are_actor_and_guild_scoped(store):
    own = create(store)
    create(store, user_id=2)
    create(store, guild_id=11)
    assert store.owned(own["id"], 10, 1)["id"] == own["id"]
    assert [j["id"] for j in store.list_owned(10, 1)] == [own["id"]]
    for guild_id, user_id in ((10, 2), (11, 1), (11, 2)):
        with pytest.raises(ValueError, match="not found"):
            store.owned(own["id"], guild_id, user_id)
    with pytest.raises(ValueError, match="not found"):
        store.owned("unknown", 10, 1)
    assert store.get("unknown") is None


def test_per_user_queue_limit_spans_guilds_and_releases_after_terminal_state(store):
    first = create(store)
    create(store, guild_id=11)
    with pytest.raises(ValueError, match="queue is full"):
        create(store)
    store.update(first["id"], status="completed")
    assert create(store)["status"] == "queued"


def test_global_queue_bound_counts_queued_and_running(store):
    for user_id in range(1, 21):
        job = create(store, user_id=user_id)
        if user_id % 2 == 0:
            store.update(job["id"], status="running")
    with pytest.raises(ValueError, match="queue is full"):
        create(store, user_id=21)
    assert len(store.pending()) == 10
    store.update(job["id"], status="cancelled")
    create(store, user_id=21)


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "timeout", "interrupted"])
def test_terminal_results_await_delivery_and_delivery_is_persistent(store, status):
    job = create(store)
    store.update(job["id"], status=status, answer="answer", artifacts=["artifact"])
    assert store.pending() == []
    result = store.undelivered()[0]
    assert result["status"] == status
    assert result["answer"] == "answer"
    assert json.loads(result["artifacts"]) == ["artifact"]
    store.update(job["id"], delivered=True)
    assert store.undelivered() == []
    assert store.get(job["id"])["status"] == status


def test_bad_status_rejected_without_changing_record(store):
    job = create(store)
    with pytest.raises(ValueError, match="status"):
        store.update(job["id"], status="invented", answer="lost")
    assert store.get(job["id"]) == job


def test_prompt_and_result_bounds_and_continuation_metadata(store):
    for prompt in ("", " \n", "x" * 16001):
        with pytest.raises(ValueError):
            store.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30, prompt=prompt)
    first = create(store)
    store.update(first["id"], status="completed", answer="x" * 25000)
    assert len(store.get(first["id"])["answer"]) == 24000
    continued = create(store, parent_id=first["id"])
    assert continued["parent_id"] == first["id"]


def test_owned_history_returns_only_latest_ten(store):
    ids = []
    for _ in range(12):
        job = create(store)
        ids.append(job["id"])
        store.update(job["id"], status="completed")
    history = store.list_owned(10, 1)
    assert [j["id"] for j in history] == list(reversed(ids))[:10]
