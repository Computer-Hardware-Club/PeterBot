import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from peterbot.agent_jobs import DELIVERY_ATTEMPT_LIMIT, JobStore


@pytest.fixture
def store(tmp_path):
    value = JobStore(str(tmp_path / "jobs.sqlite"))
    yield value
    value.close()


def create(store, user_id=1, guild_id=10, **kwargs):
    return store.create(guild_id=guild_id, user_id=user_id, channel_id=20,
                        source_message_id=30, prompt="Research hardware options", **kwargs)


def test_legacy_delivered_jobs_keep_their_receipt_during_migration(tmp_path):
    path = tmp_path / 'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('''CREATE TABLE jobs (
            id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            prompt TEXT NOT NULL, parent_id TEXT, status TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '', artifacts TEXT NOT NULL DEFAULT '[]',
            delivered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)''')
        for job_id, delivered in (('already-sent', 1), ('needs-send', 0)):
            db.execute('''INSERT INTO jobs
                (id,guild_id,user_id,channel_id,source_message_id,prompt,status,
                 answer,delivered,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (job_id, 10, 1, 20, 30, 'Legacy work', 'completed',
                 'Done', delivered, '2026-09-01', '2026-09-01'))
    store = JobStore(str(path))
    try:
        assert store.get('already-sent')['delivery_status'] == 'delivered'
        assert store.get('needs-send')['delivery_status'] == 'pending'
        assert [job['id'] for job in store.undelivered()] == ['needs-send']
    finally:
        store.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE jobs SET delivery_status='pending' WHERE id='already-sent'")
    reopened = JobStore(str(path))
    try:
        assert reopened.get('already-sent')['delivery_status'] == 'delivered'
    finally:
        reopened.close()


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


def test_preparing_job_is_never_executable_or_deliverable(store):
    # Regression for the submission race: a job persisted as `preparing`
    # while its acknowledgement send is in flight used to be picked up by the
    # negative-list `undelivered()` predicate and marked delivered while empty.
    job = create(store, ready=False)
    assert store.pending() == []
    assert store.undelivered() == []
    assert store.claim(job["id"]) is False
    store.update(job["id"], status="queued")
    assert [j["id"] for j in store.pending()] == [job["id"]]


def test_claim_is_atomic_and_only_promotes_queued(store):
    job = create(store)
    assert store.claim(job["id"]) is True
    assert store.claim(job["id"]) is False
    assert store.get(job["id"])["status"] == "running"
    cancelled = create(store, user_id=2)
    store.update(cancelled["id"], status="cancelled")
    assert store.claim(cancelled["id"]) is False


def test_concurrent_claims_across_connections_admit_exactly_one(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    job = create(first)
    second = JobStore(path)
    try:
        assert [second.claim(job["id"]), first.claim(job["id"])] == [True, False]
        assert first.get(job["id"])["status"] == "running"
    finally:
        second.close()


def test_completion_cannot_overwrite_a_terminal_cancellation(store):
    job = create(store)
    store.update(job["id"], status="running")
    assert store.transition(job["id"], to="cancelled", answer="Task cancelled.") is True
    assert store.transition(job["id"], to="completed", answer="late", artifacts=[{"name": "a.md"}]) is False
    after = store.get(job["id"])
    assert after["status"] == "cancelled"
    assert after["answer"] == "Task cancelled."
    assert json.loads(after["artifacts"]) == []


def test_transitions_enforce_the_legal_state_machine(store):
    job = create(store)
    with pytest.raises(ValueError, match="status"):
        store.transition(job["id"], to="invented")
    assert store.transition(job["id"], to="completed", answer="x") is False
    assert store.transition(job["id"], to="running") is True
    assert store.transition(job["id"], to="interrupted") is True
    assert store.transition(job["id"], to="running") is False
    assert store.get(job["id"])["status"] == "interrupted"


def test_ingress_reservations_deduplicate_discord_replays(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    store = JobStore(path)
    try:
        assert store.claim_ingress(10, 30) is None
        assert store.claim_ingress(10, 30) == ""
        job = store.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                           prompt="Research hardware options", ingress=(10, 30))
        assert store.claim_ingress(10, 30) == job["id"]
        assert store.find_by_ingress(10, 30)["id"] == job["id"]
        store.release_ingress(10, 30)
        assert store.claim_ingress(10, 30) == job["id"]
        assert store.claim_ingress(10, 31) is None
        store.release_ingress(10, 31)
        assert store.claim_ingress(10, 31) is None
    finally:
        store.close()


def test_restart_releases_unbound_ingress_reservations(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    assert first.claim_ingress(10, 30) is None
    first.close()
    restarted = JobStore(path)
    try:
        assert restarted.claim_ingress(10, 30) is None
    finally:
        restarted.close()


def test_job_and_ingress_binding_commit_atomically(tmp_path):
    # A crash directly after the job insert must not leave the reservation
    # unbound: restart deletes unbound reservations, and a replayed Discord
    # event would then create a second job and thread.
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    assert first.claim_ingress(10, 30) is None
    job = first.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                       prompt="Research hardware options", ingress=(10, 30))
    first.close()  # simulated crash immediately after the committed insert
    restarted = JobStore(path)
    try:
        assert restarted.claim_ingress(10, 30) == job["id"]
        assert restarted.find_by_ingress(10, 30)["id"] == job["id"]
    finally:
        restarted.close()


def test_ingress_binding_without_prior_reservation(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite"))
    try:
        job = store.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                           prompt="Research hardware options", ingress=(10, 30))
        assert store.claim_ingress(10, 30) == job["id"]
    finally:
        store.close()


def test_restart_fails_unfinished_submission_but_keeps_queued(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    preparing = create(first, ready=False)
    queued = create(first, user_id=2)
    first.close()
    restarted = JobStore(path)
    try:
        failed = restarted.get(preparing["id"])
        assert failed["status"] == "failed"
        assert "interrupted" in failed["answer"]
        assert restarted.get(queued["id"])["status"] == "queued"
        assert [j["id"] for j in restarted.undelivered()] == [preparing["id"]]
    finally:
        restarted.close()


def test_capacity_reserves_slots_for_preparing_jobs(store):
    create(store, ready=False)
    create(store, ready=False)
    with pytest.raises(ValueError, match="queue is full"):
        create(store)


def test_capacity_is_atomic_across_gateway_connections(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    seed = JobStore(path)
    seed.close()
    ready = Barrier(3)

    def submit(index):
        connection = JobStore(path)
        try:
            ready.wait(timeout=5)
            try:
                connection.create(guild_id=10, user_id=1, channel_id=20,
                                  source_message_id=100 + index, prompt="work")
                return "accepted"
            except ValueError:
                return "full"
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(submit, range(3)))
    assert sorted(outcomes) == ["accepted", "accepted", "full"]


def test_delivery_cursors_and_receipts_never_rewind(store):
    job = create(store)
    store.update(job["id"], status="completed", answer="answer")
    assert store.begin_delivery(job["id"]) is True
    assert store.begin_delivery(job["id"]) is False
    assert store.advance_delivery(job["id"], cursor=1, receipts=["111"]) is True
    assert store.advance_delivery(job["id"], cursor=1, receipts=["rewritten"]) is False
    assert store.advance_delivery(job["id"], cursor=3, receipts=["111", "222", "333"]) is True
    after = store.get(job["id"])
    assert after["delivery_cursor"] == 3
    assert json.loads(after["delivery_receipts"]) == ["111", "222", "333"]
    store.complete_delivery(job["id"])
    assert store.undelivered() == []
    assert store.get(job["id"]) == {**after, "delivered": 1, "delivery_status": "delivered",
                                    "updated_at": store.get(job["id"])["updated_at"]}
    assert store.begin_delivery(job["id"]) is False


def test_withheld_delivery_keeps_the_answer_but_stops_retries(store):
    job = create(store)
    store.update(job["id"], status="completed", answer="private result")
    assert store.begin_delivery(job["id"]) is True
    store.withhold_delivery(job["id"])
    assert store.undelivered() == []
    after = store.get(job["id"])
    assert after["delivery_status"] == "withheld"
    assert after["delivered"] == 1
    assert after["answer"] == "private result"
    assert after["status"] == "completed"


def test_transient_delivery_failures_are_bounded_and_exhaustion_is_explicit(store):
    job = create(store)
    store.update(job["id"], status="completed", answer="answer")
    for attempt in range(DELIVERY_ATTEMPT_LIMIT):
        assert store.begin_delivery(job["id"]) is True
        expected = "exhausted" if attempt == DELIVERY_ATTEMPT_LIMIT - 1 else "pending"
        assert store.note_delivery_failure(job["id"]) == expected
    after = store.get(job["id"])
    assert after["delivery_status"] == "exhausted"
    assert after["delivery_attempts"] == DELIVERY_ATTEMPT_LIMIT
    assert store.undelivered() == []
    assert after["delivered"] == 0
    assert after["answer"] == "answer"


def test_crash_after_receipt_freezes_unknown_without_resending(tmp_path):
    # Part 3's send may have landed before the process died. That outcome is
    # frozen for reconciliation, never silently replayed.
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    job = create(first)
    first.update(job["id"], status="completed", answer="multi-part answer")
    assert first.begin_delivery(job["id"]) is True
    assert first.advance_delivery(job["id"], cursor=2, receipts=["1", "2"]) is True
    first.close()  # crash after two acknowledged parts, mid third send
    restarted = JobStore(path)
    try:
        after = restarted.get(job["id"])
        assert after["delivery_status"] == "unknown"
        assert after["delivery_cursor"] == 2
        assert restarted.undelivered() == []  # never auto-replayed
        assert restarted.begin_delivery(job["id"]) is False
        assert restarted.reconcile_unknown_delivery(job["id"], retry=True) is True
        assert [row["id"] for row in restarted.undelivered()] == [job["id"]]
        # Retry resumes strictly after the acknowledged parts.
        assert restarted.get(job["id"])["delivery_cursor"] == 2
        assert restarted.reconcile_unknown_delivery(job["id"], retry=True) is False
    finally:
        restarted.close()


def test_uncertain_first_send_requires_operator_receipt(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    first = JobStore(path)
    job = create(first)
    first.update(job["id"], status="completed", answer="answer")
    assert first.begin_delivery(job["id"]) is True
    first.close()  # crash while the first send may have reached Discord
    restarted = JobStore(path)
    try:
        after = restarted.get(job["id"])
        assert after["delivery_status"] == "unknown"
        assert after["delivery_cursor"] == 0
        assert after["answer"] == "answer"
        assert restarted.undelivered() == []
        with pytest.raises(ValueError, match="confirmed Discord message ID"):
            restarted.reconcile_unknown_delivery(job["id"], retry=False)
        assert restarted.get(job["id"])["delivery_status"] == "unknown"
        # The operator found the message on Discord and supplies its ID.
        assert restarted.reconcile_unknown_delivery(job["id"], retry=False, confirmed_message_id=888) is True
        final = restarted.get(job["id"])
        assert final["delivery_status"] == "pending" and final["delivered"] == 0
        assert final["delivery_cursor"] == 1
        assert json.loads(final["delivery_receipts"]) == ["888"]
        assert [row["id"] for row in restarted.undelivered()] == [job["id"]]
    finally:
        restarted.close()


def test_abandoned_submission_is_terminal_and_undelivered(store):
    job = create(store, ready=False)
    store.abandon_submission(job["id"], "Task submission failed before execution.")
    after = store.get(job["id"])
    assert after["status"] == "failed"
    assert after["delivered"] == 1
    assert store.undelivered() == []
    assert store.transition(job["id"], to="queued") is False


def test_project_binding_survives_restart_and_cannot_change(tmp_path):
    path = str(tmp_path / 'jobs.sqlite')
    first = JobStore(path)
    project_id = 'a' * 32
    job = create(first, project_id=project_id)
    assert first.get(job['id'])['project_id'] == project_id
    assert first.link_project(job['id'], project_id)
    assert not first.link_project(job['id'], 'b' * 32)
    first.close()
    reopened = JobStore(path)
    try:
        assert reopened.get(job['id'])['project_id'] == project_id
        unbound = create(reopened)
        assert reopened.link_project(unbound['id'], 'b' * 32)
        assert reopened.get(unbound['id'])['project_id'] == 'b' * 32
        with pytest.raises(ValueError, match='project id'):
            create(reopened, project_id='../bad')
    finally:
        reopened.close()


def test_progress_events_are_fixed_ordered_and_terminal_frozen(store):
    job = create(store)
    assert job['stage'] == 'queued'
    assert store.claim(job['id'])
    assert store.get(job['id'])['stage'] == 'starting'
    assert store.update_progress(job['id'], seq=1, stage='researching')
    assert not store.update_progress(job['id'], seq=1, stage='running_code')
    assert store.get(job['id'])['stage'] == 'researching'
    with pytest.raises(ValueError):
        store.update_progress(job['id'], seq=2, stage='Peter says SECRET')
    with pytest.raises(ValueError):
        store.update_progress(job['id'], seq=0, stage='editing_files')
    assert store.update_progress(job['id'], seq=2, stage='running_code')
    assert store.transition(job['id'], to='completed', answer='Done')
    assert store.get(job['id'])['stage'] == 'completed'
    assert not store.update_progress(job['id'], seq=3, stage='researching')


def test_long_fast_answer_has_durable_delivery_without_queue_capacity(store):
    for user_id in range(1, 21):
        create(store, user_id=user_id)
    answer = 'useful detail ' * 400
    reply = store.record_fast_answer(guild_id=10, user_id=99, channel_id=20,
        source_message_id=900, prompt='explain carefully', answer=answer)
    assert reply['status'] == 'completed' and reply['delivery_status'] == 'pending'
    assert reply['stage'] == 'completed' and reply['answer'] == answer
    assert store.record_fast_answer(guild_id=10, user_id=99, channel_id=20,
        source_message_id=900, prompt='explain carefully', answer=answer)['id'] == reply['id']
    assert [item['id'] for item in store.undelivered()] == [reply['id']]
