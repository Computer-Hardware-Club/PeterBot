import json
from pathlib import Path
import sqlite3

import pytest

from deploy.state_backup import backup, restore, verify


def test_live_wal_backup_and_restore(tmp_path):
    source = tmp_path / "state"
    source.mkdir()
    (source / "reminders.json").write_text('[{"id": 1}]')
    database = source / "hermes" / "tasks.sqlite3"
    database.parent.mkdir()
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE jobs (id INTEGER)")
        db.execute("INSERT INTO jobs VALUES (42)")
        db.commit()
        snapshot = tmp_path / "snapshot"
        backup(source, snapshot)
        entries = verify(snapshot)
        assert {entry["path"] for entry in entries} == {"reminders.json", "hermes/tasks.sqlite3"}
        restored = tmp_path / "restored"
        restore(snapshot, restored)
        assert json.loads((restored / "reminders.json").read_text()) == [{"id": 1}]
        assert not (restored / "hermes" / "tasks.sqlite3-wal").exists()
        with sqlite3.connect(restored / "hermes" / "tasks.sqlite3") as copy:
            assert copy.execute("SELECT id FROM jobs").fetchall() == [(42,)]


def test_backup_rejects_links_and_restore_rejects_tampering(tmp_path):
    source = tmp_path / "state"
    source.mkdir()
    (source / "memory.txt").write_text("club memory")
    (source / "link.txt").symlink_to("memory.txt")
    with pytest.raises(ValueError, match="link"):
        backup(source, tmp_path / "bad")
    (source / "link.txt").unlink()
    backup(source, tmp_path / "good")
    (tmp_path / "good" / "files" / "memory.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        restore(tmp_path / "good", tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_restore_refuses_existing_directory_and_traversal(tmp_path):
    source = tmp_path / "state"
    source.mkdir()
    (source / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="must not exist inside"):
        backup(source, source / "nested-snapshot")
    snapshot = tmp_path / "snapshot"
    backup(source, snapshot)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ValueError, match="must not exist"):
        restore(snapshot, destination)
    broken_link = tmp_path / "broken-link"
    broken_link.symlink_to("missing")
    with pytest.raises(ValueError, match="must not exist"):
        restore(snapshot, broken_link)
    manifest = snapshot / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        verify(snapshot)


def build_project(tmp_path):
    """Seed one real project (manifest + content-addressed blob) in state/projects."""
    from peterbot.agent_policy import Principal
    from peterbot.project_store import ProjectStore

    principal = Principal(10, 1, 20)
    store = ProjectStore(tmp_path / "state" / "projects")
    project = store.create_project(principal, name="robot firmware", task_id="task-1")
    saved = store.save(principal, project["id"], task_id="task-1",
                       files={"firmware/main.c": b"int main(void){return 0;}"},
                       provenance="synthetic backup test")
    store.close()
    return project["id"], saved


def find_blob(state, digest):
    return state / "projects" / "blobs" / digest[:2] / digest


def test_project_round_trip_and_blob_integrity(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    project_id, _saved = build_project(tmp_path)
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        digest = db.execute("SELECT sha256 FROM files WHERE project_id=?",
                            (project_id,)).fetchone()[0]
    assert find_blob(state, digest).is_file()
    snapshot = tmp_path / "snapshot"
    backup(state, snapshot)
    restored = tmp_path / "restored"
    restore(snapshot, restored)
    assert find_blob(restored, digest).read_bytes() == b"int main(void){return 0;}"
    with sqlite3.connect(restored / "projects" / "projects.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_missing_blob_fails_snapshot_verification(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    project_id, _ = build_project(tmp_path)
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        digest = db.execute("SELECT sha256 FROM files WHERE project_id=?",
                            (project_id,)).fetchone()[0]
    # An interrupted store-side GC can remove bytes while the live manifest
    # still references them; the databases stay consistent, so only the
    # snapshot cross-check can prove the backup is restorable.
    find_blob(state, digest).unlink()
    snapshot = tmp_path / "snapshot"
    backup(state, snapshot)
    with pytest.raises(ValueError, match="missing or mismatched blob"):
        verify(snapshot)
    with pytest.raises(ValueError, match="missing or mismatched blob"):
        restore(snapshot, tmp_path / "restored")


def test_hash_mismatched_blob_fails_snapshot_verification(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    project_id, _ = build_project(tmp_path)
    with sqlite3.connect(state / "projects" / "projects.sqlite") as db:
        digest = db.execute("SELECT sha256 FROM files WHERE project_id=?",
                            (project_id,)).fetchone()[0]
    blob = find_blob(state, digest)
    blob.write_bytes(blob.read_bytes() + b"tampered")
    snapshot = tmp_path / "snapshot"
    backup(state, snapshot)
    with pytest.raises(ValueError, match="missing or mismatched blob"):
        verify(snapshot)


def test_nested_hermes_project_manifest_is_checked_in_parent_backup(tmp_path):
    from peterbot.agent_policy import Principal
    from peterbot.project_store import ProjectStore

    state = tmp_path / 'data'
    store = ProjectStore(state / 'hermes' / 'projects')
    actor = Principal(10, 1, 20)
    project = store.create_project(actor, name='edigits', task_id='task-1')
    store.save(actor, project['id'], task_id='task-1',
               files={'src/main.rs': b'fn main() {}'}, provenance='synthetic fixture')
    store.close()
    snapshot = tmp_path / 'snapshot'
    backup(state, snapshot)
    entries = verify(snapshot)
    blob = next(item for item in entries if item['path'].startswith('hermes/projects/blobs/'))
    assert 'hermes/projects/projects.sqlite' in {item['path'] for item in entries}
    (snapshot / 'files' / blob['path']).unlink()
    manifest_path = snapshot / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'] = [entry for entry in manifest['files'] if entry['path'] != blob['path']]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='missing or mismatched blob'):
        verify(snapshot)


def test_restore_preserves_queued_job_and_unknown_outbox_receipt(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    with sqlite3.connect(state / "outbox.sqlite3") as db:
        db.execute("""CREATE TABLE announcements (
            id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, actor_user_id INTEGER NOT NULL,
            source_channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            target_channel_id INTEGER NOT NULL, content TEXT NOT NULL,
            content_hash TEXT NOT NULL, nonce TEXT NOT NULL, status TEXT NOT NULL,
            discord_message_id INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            UNIQUE(guild_id, source_message_id))""")
        db.execute("""INSERT INTO announcements VALUES
            ('act-1',10,1,20,40,30,'announcement text','hash','nonce','unknown',NULL,1,'2026-01-01T00:00:00+00:00','2026-01-02T00:00:00+00:00')""")
    with sqlite3.connect(state / "jobs.sqlite3") as db:
        db.execute("""CREATE TABLE jobs (
            id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            prompt TEXT NOT NULL, parent_id TEXT, status TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '', artifacts TEXT NOT NULL DEFAULT '[]',
            delivered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, delivery_status TEXT NOT NULL DEFAULT 'pending')""")
        db.execute("""INSERT INTO jobs VALUES
            ('job-1',10,1,20,41,'queued prompt',NULL,'queued','', '[]',0,
             '2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00','pending')""")
    snapshot = tmp_path / "snapshot"
    backup(state, snapshot)
    restored = tmp_path / "restored"
    restore(snapshot, restored)
    with sqlite3.connect(restored / "outbox.sqlite3") as db:
        row = db.execute("SELECT status, discord_message_id FROM announcements").fetchone()
        assert row == ("unknown", None)
        assert db.execute("SELECT COUNT(*) FROM announcements WHERE status='sent'").fetchone()[0] == 0
    with sqlite3.connect(restored / "jobs.sqlite3") as db:
        assert db.execute("SELECT status, delivered FROM jobs").fetchone() == ("queued", 0)
