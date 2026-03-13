import json
from datetime import datetime, timedelta
from pathlib import Path

from peterbot.reminders import ReminderManager, parse_reminder_time


def test_parse_reminder_time_core_formats() -> None:
    now = datetime(2026, 3, 10, 12, 0, 0)
    assert parse_reminder_time("in 45 minutes", now=now) == datetime(2026, 3, 10, 12, 45, 0)
    assert parse_reminder_time("tomorrow at 9:15 PM", now=now) == datetime(2026, 3, 11, 21, 15, 0)
    assert parse_reminder_time("03/10/26 2:30 PM", now=now) == datetime(2026, 3, 10, 14, 30, 0)
    assert parse_reminder_time("in 0 minutes", now=now) is None


def test_reminder_manager_uses_stable_data_directory(tmp_path) -> None:
    manager = ReminderManager(data_dir=str(tmp_path))
    manager.add_reminder(
        user_id=42,
        message="check persistence path",
        remind_time=datetime(2026, 3, 12, 9, 30, 0),
    )
    assert Path(manager.reminders_file) == (tmp_path / "reminders.json")
    assert Path(manager.reminders_file).exists()


def test_atomic_save_and_load_round_trip(tmp_path) -> None:
    manager = ReminderManager(data_dir=str(tmp_path))
    manager.add_reminder(
        user_id=7,
        message="round trip reminder",
        remind_time=datetime(2026, 3, 15, 10, 0, 0),
    )

    created_files = sorted(path.name for path in tmp_path.iterdir())
    assert created_files == ["reminders.json"]

    reloaded = ReminderManager(data_dir=str(tmp_path))
    reloaded.load_reminders()
    assert len(reloaded.reminders) == 1
    assert reloaded.reminders[0]["user_id"] == 7
    assert reloaded.reminders[0]["message"] == "round trip reminder"
    assert reloaded.reminders[0]["remind_time"] == datetime(2026, 3, 15, 10, 0, 0)


def test_legacy_fallback_reads_old_files_once(tmp_path, monkeypatch) -> None:
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

    manager = ReminderManager(data_dir=str(data_dir))
    manager.load_reminders()
    downtime = manager.get_downtime()

    assert len(manager.reminders) == 1
    assert manager.reminders[0]["message"] == "legacy reminder"
    assert downtime is not None
    assert not (legacy_dir / "bot_shutdown.json").exists()
