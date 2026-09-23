"""PETER-16: private diagnostics, dry-run-first retention, and snapshot staging checks."""
import json
import socket
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from deploy.housekeeping import main as housekeeping_main
from peterbot.agent_jobs import JobStore
from peterbot.agent_memory import ScopedMemoryStore
from peterbot.agent_policy import AgentPolicy, ControlIntent, Principal
from peterbot.announcement_outbox import AnnouncementOutbox
from peterbot.club_state import ClubStateStore
from peterbot.conversation_store import ConversationStore
from peterbot.foreground import ForegroundScheduler
from peterbot.ops_metrics import MetricStore
from peterbot.operator_ops import (
    RetentionConfig, component_health, diagnose, retention_apply, retention_plan,
)
from peterbot.project_store import ProjectStore
from peterbot.style_state import StyleStore

GUILD = 1700000000000000001
USER = 1700000000000000002
OFFICER = 1700000000000000003
CHANNEL = 1700000000000000004
TARGET = 1700000000000000005
ROLE = 1700000000000000006
RECEIPT = 1700000000000000007
NOW = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)

CANARY_PROMPT = "CANARY-PROMPT wire the rover motors"
CANARY_ANSWER = "CANARY-ANSWER solder pin seven"
CANARY_MEMORY = "CANARY-MEMORY favorite soldering iron"
CANARY_MESSAGE = "CANARY-ANNOUNCEMENT meeting moved"
CANARY_PROVENANCE = "CANARY-PROVENANCE from task nine"

POLICY = AgentPolicy(allowed_guild_ids=frozenset({GUILD}), officer_role_ids=frozenset({ROLE}),
                     control_channel_ids=frozenset({CHANNEL}))
OFFICER_PRINCIPAL = Principal(GUILD, OFFICER, CHANNEL, (ROLE,))
MEMBER_PRINCIPAL = Principal(GUILD, USER, CHANNEL)

CONFIG = RetentionConfig(conversations_days=30, metrics_days=14,
                         terminal_jobs_days=45, settled_receipts_days=60)


def test_split_live_layout_finds_foreground_and_nested_hermes_stores(tmp_path):
    state = tmp_path / 'data'
    hermes = state / 'hermes'
    hermes.mkdir(parents=True)
    fg = ForegroundScheduler(str(state / 'foreground.sqlite3'))
    fg.close_sync()
    JobStore(str(hermes / 'tasks.sqlite3')).close()
    ConversationStore(str(hermes / 'conversations.sqlite3')).db.close()
    ProjectStore(hermes / 'projects').close()
    report = diagnose(state)
    for name in ('foreground', 'jobs', 'conversation', 'projects'):
        assert report['components'][name]['status'] == 'ok'
    plan = retention_plan(state, CONFIG)
    assert plan['categories']['conversations']['status'] == 'measured'


def iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def age_rows(path, sql: str, *params) -> None:
    with sqlite3.connect(path) as db:
        db.execute(sql, params)


def seed_jobs(state):
    store = JobStore(str(state / "jobs.sqlite3"))
    made = {}
    def fresh(key, source):
        return store.create(guild_id=GUILD, user_id=USER, channel_id=CHANNEL,
                            source_message_id=source, prompt=CANARY_PROMPT)
    settled_old = []
    for key, source, settle in (("delivered_old", 501, "complete"),
                                ("withheld_old", 502, "withhold")):
        job = fresh(key, source)
        store.claim(job["id"])
        store.transition(job["id"], to="completed", answer=CANARY_ANSWER)
        store.begin_delivery(job["id"])
        if settle == "complete":
            store.complete_delivery(job["id"])
        else:
            store.withhold_delivery(job["id"])
        made[key] = job["id"]
        settled_old.append(job["id"])
    ambiguous = fresh("unknown_old", 503)
    store.claim(ambiguous["id"])
    store.transition(ambiguous["id"], to="completed", answer=CANARY_ANSWER)
    store.begin_delivery(ambiguous["id"])
    store.mark_delivery_unknown(ambiguous["id"])
    made["unknown_old"] = ambiguous["id"]
    starved = fresh("exhausted_old", 504)
    store.claim(starved["id"])
    store.transition(starved["id"], to="completed", answer=CANARY_ANSWER)
    for _ in range(11):
        assert store.begin_delivery(starved["id"])
        assert store.note_delivery_failure(starved["id"]) == "pending"
    assert store.begin_delivery(starved["id"])
    assert store.note_delivery_failure(starved["id"]) == "exhausted"
    made["exhausted_old"] = starved["id"]
    active = fresh("queued_old", 505)
    made["queued_old"] = active["id"]
    recent = fresh("delivered_recent", 506)
    store.claim(recent["id"])
    store.transition(recent["id"], to="completed", answer=CANARY_ANSWER)
    store.begin_delivery(recent["id"])
    store.complete_delivery(recent["id"])
    made["delivered_recent"] = recent["id"]
    store.close()
    jobs_db = state / "jobs.sqlite3"
    for key in ("delivered_old", "withheld_old", "unknown_old", "exhausted_old", "queued_old"):
        age_rows(jobs_db, "UPDATE jobs SET created_at=?, updated_at=? WHERE id=?",
                 iso(200), iso(200), made[key])
    return made


def seed_outbox(state):
    outbox = AnnouncementOutbox(state / "outbox.sqlite3", POLICY, {GUILD: frozenset({TARGET})})
    ids = {}
    for key, source in (("sent_old", 601), ("sent_recent", 602), ("unknown", 603), ("pending", 604)):
        intent = ControlIntent(GUILD, OFFICER, CHANNEL, source, "announcement")
        record = outbox.propose(OFFICER_PRINCIPAL, intent, target_channel_id=TARGET,
                                content=CANARY_MESSAGE, channel_is_private=True)
        if key != "pending":
            outbox.begin_send(record["id"], OFFICER_PRINCIPAL, intent, channel_is_private=True)
            if key == "unknown":
                outbox.mark_unknown(record["id"])
            else:
                outbox.mark_sent(record["id"], RECEIPT)
        ids[key] = record["id"]
    outbox.close()
    age_rows(state / "outbox.sqlite3",
             "UPDATE announcements SET created_at=?, updated_at=? WHERE id=?",
             iso(200), iso(200), ids["sent_old"])
    return ids


def seed_conversation(state):
    store = ConversationStore(str(state / "conversation.sqlite3"))
    store.append_turn(guild_id=GUILD, user_id=USER, channel_id=CHANNEL, source_message_id=701,
                      audience="private", prompt=CANARY_PROMPT, answer=CANARY_ANSWER)
    store.append_turn(guild_id=GUILD, user_id=USER, channel_id=CHANNEL, source_message_id=702,
                      audience="private", prompt="current prompt", answer="current answer")
    store.db.close()
    age_rows(state / "conversation.sqlite3",
             "UPDATE conversation_turns SET created_at=? WHERE source_message_id=701", iso(200))


def seed_metrics(state):
    store = MetricStore(state / "metrics.sqlite3")
    store.record("model", "ok", 120)
    store.record("model", "ok", 130)
    store.db.close()
    age_rows(state / "metrics.sqlite3", "UPDATE stage_metrics SET at=? WHERE id=1", iso(200))


def seed_memory(state):
    store = ScopedMemoryStore(state / "memory.sqlite3", POLICY)
    record = store.create(MEMBER_PRINCIPAL, scope="personal", content=CANARY_MEMORY,
                          source_message_id=801)
    store.delete(MEMBER_PRINCIPAL, record["id"], source_message_id=802, expected_version=1)


def seed_audit(state):
    ClubStateStore(state / "club.sqlite3", POLICY)
    with sqlite3.connect(state / "club.sqlite3") as db:
        db.execute("INSERT INTO club_guild_state VALUES (?,?,?)", (GUILD, 3, iso(400)))
        db.execute("""INSERT INTO club_revisions
                      (guild_id,version,source_message_id,actor_id,action,operation,
                       before_state,after_state,created_at)
                      VALUES (?,?,?,?,?,?,?,?,?)""",
                   (GUILD, 3, 901, OFFICER, "club_fact", "set", "{}", "{}", iso(400)))
    style = StyleStore(state / "style.sqlite3", POLICY)
    style.db.execute("INSERT INTO style VALUES (?,?,?,?)", (GUILD, 2, "{}", iso(400)))
    style.db.execute("INSERT INTO style_revisions VALUES (?,?,?,?,?,?,?,?)",
                     (GUILD, 2, OFFICER, 902, "edit", "{}", "{}", iso(400)))
    style.db.commit()
    style.close()


def seed_foreground(state):
    scheduler = ForegroundScheduler(str(state / "foreground.sqlite3"))
    scheduler.enqueue(kind="task", guild_id=GUILD, user_id=USER, channel_id=CHANNEL,
                      source_message_id=903, job_id="job-fg")
    claimed = scheduler.claim_next()
    assert claimed is not None and claimed["id"]
    scheduler.release_worker(claimed["id"], confirmed=False, event="worker-stopped")
    scheduler.db.close()


def seed_project(state):
    store = ProjectStore(state / "projects")
    project = store.create_project(MEMBER_PRINCIPAL, name="rover firmware", task_id="task-p1")
    store.save(MEMBER_PRINCIPAL, project["id"], task_id="task-p1",
               files={"firmware/main.c": b"void main(void){}"}, provenance=CANARY_PROVENANCE)
    store.close()
    return project["id"]


@pytest.fixture
def state(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir()
    return directory


def seed_full(state):
    seed_jobs(state)
    seed_outbox(state)
    seed_conversation(state)
    seed_metrics(state)
    seed_memory(state)
    seed_audit(state)
    seed_foreground(state)
    seed_project(state)


def test_diagnose_is_private_and_aggregate_only(state):
    seed_full(state)
    report = diagnose(state, now=NOW)
    dump = json.dumps(report)
    for canary in (CANARY_PROMPT, CANARY_ANSWER, CANARY_MEMORY, CANARY_MESSAGE,
                   CANARY_PROVENANCE, "firmware/main.c", "rover firmware", "token"):
        assert canary not in dump
    for identifier in (GUILD, USER, OFFICER, CHANNEL, TARGET, RECEIPT, 501, 701):
        assert str(identifier) not in dump
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        digest = db.execute("SELECT sha256 FROM files LIMIT 1").fetchone()[0]
    assert digest not in dump
    assert report["jobs"]["by_status"]["completed"] == 5
    assert report["jobs"]["by_status"]["queued"] == 1
    assert report["jobs"]["delivery"] == {"pending": 1, "delivered": 2, "exhausted": 1,
                                          "unknown": 1, "withheld": 1}
    assert report["jobs"]["oldest_active_age_seconds"] > 0
    assert report["foreground"]["cleanup_unknown"] == 1
    assert report["outbox"]["by_status"] == {"pending": 1, "sent": 2, "unknown": 1}
    assert report["memory"] == {"active": 0, "soft_forgotten": 1, "revisions": 2}
    assert report["audit"]["club"]["max_version"] == 3
    assert report["audit"]["style"]["revisions"] == 1
    assert report["projects"]["live"] == 1 and report["projects"]["files"] == 1
    assert report["projects"]["blob_files"] == 1


def test_revision_reported_from_environment(state, monkeypatch):
    monkeypatch.setenv("PETERBOT_REVISION", "deadbeefcafe0000")
    assert diagnose(state, now=NOW)["revision"] == "deadbeefcafe0000"
    monkeypatch.delenv("PETERBOT_REVISION")
    assert diagnose(state, now=NOW)["revision"] == "unknown"


def test_offline_and_degraded_states_are_distinct(state):
    health = component_health(state)
    assert all(entry["status"] == "offline" for entry in health.values())
    sqlite3.connect(state / "jobs.sqlite3").close()  # valid database, wrong shape
    (state / "metrics.sqlite3").write_text("this is not a database")
    with sqlite3.connect(state / "conversation.sqlite3") as db:
        db.execute("CREATE TABLE conversation_turns (x)")
        db.execute("PRAGMA user_version=99")
    health = component_health(state)
    assert health["jobs"] == {"status": "degraded", "reason": "missing_tables"}
    assert health["metrics"] == {"status": "degraded", "reason": "corrupt_or_unreadable"}
    assert health["conversation"] == {"status": "degraded", "reason": "schema_newer"}
    assert health["club"]["status"] == "offline"
    plan = retention_plan(state, CONFIG, now=NOW)
    assert plan["categories"]["terminal_jobs"]["status"] == "degraded"
    assert plan["categories"]["terminal_jobs"]["eligible"] is None
    assert plan["categories"]["settled_receipts"] == {"status": "offline", "eligible": 0}


def test_retention_plan_is_dry_run(state):
    seed_full(state)
    plan = retention_plan(state, CONFIG, now=NOW)
    assert plan["dry_run"] is True
    assert plan["include_projects"] is False
    assert plan["projects"] == {"status": "excluded"}
    assert plan["categories"]["conversations"]["eligible"] == 1
    assert plan["categories"]["metrics"]["eligible"] == 1
    assert plan["categories"]["terminal_jobs"]["eligible"] == 2  # delivered + withheld only
    assert plan["categories"]["settled_receipts"]["eligible"] == 1
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 6
    with sqlite3.connect(state / "conversation.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 2


def test_retention_apply_removes_settled_rows_and_keeps_everything_protected(state):
    seed_full(state)
    result = retention_apply(state, CONFIG, now=NOW)
    assert result["dry_run"] is False
    assert result["deleted"] == {"conversations": 1, "metrics": 1,
                                 "terminal_jobs": 2, "settled_receipts": 1}
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        surviving = {row[:2]: row[2] for row in db.execute(
            "SELECT status, delivery_status, COUNT(*) FROM jobs "
            "GROUP BY status, delivery_status")}
        assert surviving == {("queued", "pending"): 1, ("completed", "unknown"): 1,
                             ("completed", "exhausted"): 1, ("completed", "delivered"): 1}
    with sqlite3.connect(state / "outbox.sqlite3") as db:
        statuses = dict(db.execute("SELECT status, COUNT(*) FROM announcements GROUP BY status"))
        assert statuses == {"pending": 1, "sent": 1, "unknown": 1}  # only the recent 'sent' survives
    with sqlite3.connect(state / "conversation.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 1
    with sqlite3.connect(state / "metrics.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM stage_metrics").fetchone()[0] == 1
    with sqlite3.connect(state / "memory.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0] == 2
    with sqlite3.connect(state / "club.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM club_revisions").fetchone()[0] == 1
    with sqlite3.connect(state / "style.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM style_revisions").fetchone()[0] == 1


def test_retention_apply_refuses_degraded_store_before_writing(state):
    seed_conversation(state)
    (state / "metrics.sqlite3").write_text("this is not a database")
    with pytest.raises(ValueError, match="degraded metrics store"):
        retention_apply(state, CONFIG, now=NOW)
    with sqlite3.connect(state / "conversation.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0] == 2


def test_retention_apply_skips_offline_stores(state):
    seed_conversation(state)
    result = retention_apply(state, CONFIG, now=NOW)
    assert result["deleted"] == {"conversations": 1}
    assert result["skipped"]["terminal_jobs"] == "offline"
    assert result["skipped"]["settled_receipts"] == "offline"


def test_project_sweep_only_under_its_own_policy(state):
    seed_project(state)
    aged = RetentionConfig()
    plan = retention_plan(state, aged, now=NOW)
    assert plan["projects"] == {"status": "excluded"}
    age_rows(state / "projects" / "projects.sqlite", "UPDATE versions SET created_at=?",
             int(NOW.timestamp()) - 200 * 86400)
    age_rows(state / "projects" / "projects.sqlite",
             "UPDATE projects SET created_at=?, updated_at=?",
             int(NOW.timestamp()) - 200 * 86400, int(NOW.timestamp()) - 200 * 86400)
    plan = retention_plan(state, RetentionConfig(include_projects=True), now=NOW)
    assert plan["projects"]["status"] == "measured"
    assert plan["projects"]["policy_days"] == 90
    assert plan["projects"]["versions_aged"] == 1
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 1  # plan mutates nothing
    result = retention_apply(state, RetentionConfig(include_projects=True), now=NOW)
    assert result["projects"]["versions_removed"] == 1
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 0
    assert not list((state / "projects" / "blobs").glob("*/*"))


def test_no_network_or_model_calls_in_housekeeping(state, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("housekeeping attempted a network call")
    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)
    monkeypatch.setattr(socket, "getaddrinfo", explode)
    seed_full(state)
    diagnose(state, now=NOW)
    retention_plan(state, CONFIG, now=NOW)
    assert housekeeping_main(["diagnose", str(state)]) == 0
    assert housekeeping_main(["retention", str(state)]) == 0
    assert housekeeping_main(["retention", str(state), "--apply",
                              "--backup-destination", str(state.parent / "snap")]) == 0
    assert housekeeping_main(["check-snapshot", str(state.parent / "snap"),
                              str(state.parent / "staging")]) == 0


def test_cli_diagnose_and_dry_run_default(tmp_path, state, capsys):
    seed_full(state)
    assert housekeeping_main(["diagnose", str(state)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["components"]["jobs"]["status"] == "ok"
    assert housekeeping_main(["retention", str(state)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 6


def test_cli_apply_requires_backup(tmp_path, state, capsys):
    seed_full(state)
    with pytest.raises(SystemExit) as exit_info:
        housekeeping_main(["retention", str(state), "--apply"])
    assert exit_info.value.code == 2
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 6


def test_cli_check_snapshot_restores_staging_with_project_bytes_and_unknown_receipt(tmp_path, state):
    seed_full(state)
    snapshot = tmp_path / "snapshot"
    seed_digest = None
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        seed_digest = db.execute("SELECT sha256 FROM files").fetchone()[0]
    from deploy.state_backup import backup
    backup(state, snapshot)
    staging = tmp_path / "staging"
    assert housekeeping_main(["check-snapshot", str(snapshot), str(staging)]) == 0
    assert (staging / "projects" / "blobs" / seed_digest[:2] / seed_digest).read_bytes() \
        == b"void main(void){}"
    report = diagnose(staging)
    assert report["outbox"]["by_status"]["unknown"] == 1
    assert report["projects"]["live"] == 1
    missing = tmp_path / "state-missing-blob"
    missing.mkdir()
    import shutil
    shutil.copytree(state / "projects", missing / "projects")
    (missing / "projects" / "blobs" / seed_digest[:2] / seed_digest).unlink()
    broken = tmp_path / "broken-snapshot"
    backup(missing, broken)
    refused = tmp_path / "never-created"
    with pytest.raises(ValueError, match="missing or mismatched blob"):
        housekeeping_main(["check-snapshot", str(broken), str(refused)])
    assert not refused.exists()
