"""Durable task queue. Only the trusted gateway opens this database."""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            prompt TEXT NOT NULL, parent_id TEXT, status TEXT NOT NULL,
            answer TEXT NOT NULL DEFAULT '', artifacts TEXT NOT NULL DEFAULT '[]',
            delivered INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(jobs)')}
        for name, definition in {'input_files':"TEXT NOT NULL DEFAULT '[]'", 'delivery_cursor':'INTEGER NOT NULL DEFAULT 0'}.items():
            if name not in columns:
                self.db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {definition}')
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',answer='Task submission was interrupted.',updated_at=? WHERE status='preparing'",(now(),))
            self.db.execute("UPDATE jobs SET status='interrupted', answer='The gateway restarted during this task. Use /continue_task to resume from the saved objective.', updated_at=? WHERE status='running'", (now(),))

    def create(self, *, guild_id: int, user_id: int, channel_id: int,
               source_message_id: int, prompt: str, parent_id: str | None = None, input_files: list | None = None, ready: bool = True) -> dict:
        if not prompt.strip() or len(prompt) > 16000:
            raise ValueError('Please use a task description between 1 and 16,000 characters.')
        self.check_capacity(user_id)
        job_id = uuid.uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO jobs (id,guild_id,user_id,channel_id,source_message_id,prompt,parent_id,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (job_id,guild_id,user_id,channel_id,source_message_id,prompt,parent_id,'queued' if ready else 'preparing',now(),now()))
            self.db.execute('UPDATE jobs SET input_files=? WHERE id=?',(json.dumps(input_files or []),job_id))
        return self.get(job_id)

    def check_capacity(self, user_id: int) -> None:
        count = self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
        own = self.db.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND status IN ('queued','running')", (user_id,)).fetchone()[0]
        if count >= 20 or own >= 2:
            raise ValueError('The task queue is full for now. Finish or cancel an existing task first.')

    def latest_for_thread(self, guild_id: int, user_id: int, channel_id: int) -> dict | None:
        row=self.db.execute('SELECT * FROM jobs WHERE guild_id=? AND user_id=? AND channel_id=? ORDER BY created_at DESC LIMIT 1',(guild_id,user_id,channel_id)).fetchone()
        return dict(row) if row else None

    def get(self, job_id: str) -> dict | None:
        row = self.db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        return dict(row) if row else None

    def owned(self, job_id: str, guild_id: int, user_id: int) -> dict:
        job = self.get(job_id)
        if not job or job['guild_id'] != guild_id or job['user_id'] != user_id:
            raise ValueError('Task not found for your account in this server.')
        return job

    def list_owned(self, guild_id: int, user_id: int) -> list[dict]:
        return [dict(r) for r in self.db.execute('SELECT id,status,channel_id,created_at FROM jobs WHERE guild_id=? AND user_id=? ORDER BY created_at DESC LIMIT 10',(guild_id,user_id))]

    def pending(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 20")]

    def undelivered(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM jobs WHERE delivered=0 AND status NOT IN ('queued','running') ORDER BY created_at LIMIT 20")]

    def update(self, job_id: str, *, status: str | None = None, answer: str | None = None,
               artifacts: list | None = None, delivered: bool | None = None, delivery_cursor: int | None = None) -> None:
        fields = {'updated_at': now()}
        if status is not None:
            if status not in {'queued','running','completed','failed','cancelled','timeout','interrupted'}:
                raise ValueError('Invalid task status')
            fields['status'] = status
        if answer is not None:
            fields['answer'] = answer[:24000]
        if artifacts is not None:
            fields['artifacts'] = json.dumps(artifacts)
        if delivery_cursor is not None:
            fields['delivery_cursor']=delivery_cursor
        if delivered is not None:
            fields['delivered'] = int(delivered)
        with self.db:
            self.db.execute('UPDATE jobs SET '+','.join(f'{key}=?' for key in fields)+' WHERE id=?', (*fields.values(),job_id))

    def close(self) -> None:
        self.db.close()
