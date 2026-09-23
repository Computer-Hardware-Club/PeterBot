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
