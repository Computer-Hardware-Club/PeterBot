"""Durable, audience-scoped final conversation turns.

Only final user-visible text belongs here. Reasoning, raw tool output, and
Discord bystander messages are deliberately absent. The caller must verify
current Discord access before reading or appending a turn.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
AUDIENCES = frozenset({"public", "private", "officer"})
MAX_PROMPT = 16_000
MAX_ANSWER = 24_000
MAX_CONTEXT_CHARS = 8_000
MAX_CONTEXT_TURNS = 6
MAX_PINNED_CHARS = 2_000
CONSTRAINT_MARKERS = ("must ", "only ", "don't ", "do not ", "never ", "keep ", "please ")


def _valid_id(value: object) -> bool:
    return type(value) is int and 0 < value <= 2**63 - 1


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _constraints(prompt: str) -> list[str]:
    """Keep explicit user text verbatim; never synthesize a constraint."""
    lines = [line.strip() for line in prompt.splitlines()]
    return [line[:500] for line in lines if line and any(marker in line.lower() for marker in CONSTRAINT_MARKERS)]


class ConversationStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ValueError("Conversation database is newer than this gateway")
        if version == 0:
            with self.db:
                self.db.execute("""CREATE TABLE IF NOT EXISTS conversation_turns (
                    guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
                    audience TEXT NOT NULL CHECK(audience IN ('public','private','officer')),
                    prompt TEXT NOT NULL, answer TEXT NOT NULL,
                    task_id TEXT, project_id TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, source_message_id)
                )""")
                self.db.execute("""CREATE INDEX IF NOT EXISTS conversation_scope_recent
                    ON conversation_turns (guild_id,user_id,channel_id,audience,created_at DESC)""")
                self.db.execute("PRAGMA user_version=1")

    def append_turn(self, *, guild_id: int, user_id: int, channel_id: int,
                    source_message_id: int, audience: str, prompt: str, answer: str,
                    task_id: str | None = None, project_id: str | None = None) -> bool:
        if not all(_valid_id(v) for v in (guild_id, user_id, channel_id, source_message_id)):
            raise ValueError("Invalid Discord identity")
        if audience not in AUDIENCES:
            raise ValueError("Invalid audience")
        if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip():
            raise ValueError("A final user turn and answer are required")
        if len(prompt) > MAX_PROMPT or len(answer) > MAX_ANSWER:
            raise ValueError("Conversation turn exceeds storage limit")
        if task_id is not None and (not isinstance(task_id, str) or len(task_id) > 64):
            raise ValueError("Invalid task id")
        if project_id is not None and (not isinstance(project_id, str) or len(project_id) > 64):
            raise ValueError("Invalid project id")
        with self.db:
            cursor = self.db.execute("""INSERT OR IGNORE INTO conversation_turns
                (guild_id,user_id,channel_id,source_message_id,audience,prompt,answer,task_id,project_id,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (guild_id,user_id,channel_id,source_message_id,audience,prompt,answer,
                 task_id,project_id,_utcnow()))
        return cursor.rowcount == 1

    def context(self, *, guild_id: int, user_id: int, channel_id: int,
                audience: str, max_chars: int = MAX_CONTEXT_CHARS) -> list[dict[str, str]]:
        """Return bounded text from exactly one verified requester and audience.

        The oldest request and later explicit constraint lines are kept verbatim
        when ordinary recent-turn truncation would lose them. This is a
        deterministic compact record, with no model call and no claim of
        progress on tasks that did not finish.
        """
        if not all(_valid_id(v) for v in (guild_id, user_id, channel_id)) or audience not in AUDIENCES:
            raise ValueError("Invalid conversation scope")
        if not 500 <= max_chars <= MAX_CONTEXT_CHARS:
            raise ValueError("Invalid context limit")
        rows = list(self.db.execute("""SELECT prompt,answer FROM conversation_turns
            WHERE guild_id=? AND user_id=? AND channel_id=? AND audience=?
            ORDER BY created_at DESC,source_message_id DESC LIMIT 64""",
            (guild_id,user_id,channel_id,audience)))
        if not rows:
            return []
        rows.reverse()
        pinned: list[str] = []
        for row in rows:
            for line in _constraints(row["prompt"]):
                if line not in pinned and sum(map(len, pinned)) + len(line) <= MAX_PINNED_CHARS:
                    pinned.append(line)
        selected = rows[-MAX_CONTEXT_TURNS:]
        prefix: list[dict[str, str]] = []
        if rows[0] not in selected:
            content = "Original request (verbatim excerpt): " + rows[0]["prompt"][:1000]
            prefix.append({"role": "user", "content": content[:max_chars // 5]})
        if pinned:
            content = "Earlier explicit user constraints (verbatim): " + json.dumps(pinned, ensure_ascii=False)
            prefix.append({"role": "user", "content": content[:max_chars // 5]})
        remaining = max_chars - sum(len(item["content"]) for item in prefix)
        recent: list[dict[str, str]] = []
        for row in reversed(selected):
            if remaining < 100:
                break
            # Allocate newest turns first. Split the last available budget
            # between question and answer so neither disappears at the edge.
            prompt_limit = min(2000, max(50, remaining // 2))
            prompt = row["prompt"][:prompt_limit]
            answer = row["answer"][:min(2000, remaining - len(prompt))]
            recent[0:0] = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]
            remaining -= len(prompt) + len(answer)
        return prefix + recent

    def delete_before(self, cutoff: datetime) -> int:
        """Retention entry point; caller owns the policy and backup cadence."""
        if cutoff.tzinfo is None:
            raise ValueError("Cutoff must have a timezone")
        with self.db:
            result = self.db.execute("DELETE FROM conversation_turns WHERE created_at<?",
                                     (cutoff.astimezone(timezone.utc).isoformat(),))
        return result.rowcount
