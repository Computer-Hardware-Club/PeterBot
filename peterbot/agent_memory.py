"""SQLite memory with actor-bound access and an append-only revision ledger.

All methods require a fresh gateway-supplied Principal. There is intentionally no
owner/guild override and no agent-facing audit modification or raw SQL method.
Records describe facts; they never participate in authorization. The database
must live outside the agent execution sandbox, accessible only by the gateway.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import UUID, uuid4

from .agent_policy import AgentPolicy, Principal, _valid_id


class MemoryConflict(ValueError):
    """The caller must read the current version before retrying its edit."""


class ScopedMemoryStore:
    MAX_CONTENT_CHARS = 4000
    MAX_QUERY_CHARS = 200
    MAX_RESULTS = 50
    MAX_RECORDS_PER_SCOPE = 1000

    def __init__(self, path: str | Path, policy: AgentPolicy) -> None:
        self.path = Path(path)
        self.policy = policy
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(write=True) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    guild_id INTEGER NOT NULL,
                    scope TEXT NOT NULL CHECK(scope IN ('personal', 'club')),
                    owner_user_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    actor_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS memories_access
                    ON memories(guild_id, scope, owner_user_id, deleted);
                CREATE TABLE IF NOT EXISTS memory_revisions (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    guild_id INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    owner_user_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    actor_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL CHECK(operation IN ('create', 'update', 'delete')),
                    PRIMARY KEY(memory_id, version)
                );
                CREATE TRIGGER IF NOT EXISTS revisions_no_update
                    BEFORE UPDATE ON memory_revisions
                    BEGIN SELECT RAISE(ABORT, 'revision ledger is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS revisions_no_delete
                    BEFORE DELETE ON memory_revisions
                    BEGIN SELECT RAISE(ABORT, 'revision ledger is append-only'); END;
            """)

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout = 10000")
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _validate_content(cls, content: str) -> None:
        if not isinstance(content, str) or not content.strip() or len(content) > cls.MAX_CONTENT_CHARS:
            raise ValueError(f"content must contain 1–{cls.MAX_CONTENT_CHARS} characters")

    @staticmethod
    def _validate_source(source_message_id: int) -> None:
        if not _valid_id(source_message_id):
            raise ValueError("source_message_id must be a Discord message ID")

    @staticmethod
    def _validate_memory_id(memory_id: str) -> None:
        if not isinstance(memory_id, str) or len(memory_id) != 36:
            raise ValueError("memory_id must be a canonical UUID")
        try:
            if str(UUID(memory_id)) != memory_id:
                raise ValueError
        except ValueError:
            raise ValueError("memory_id must be a canonical UUID") from None

    def _read_visible(
        self, conn: sqlite3.Connection, principal: Principal, memory_id: str
    ) -> sqlite3.Row | None:
        # Apply visibility in SQL, before any row contents enter application code.
        return conn.execute(
            """SELECT * FROM memories WHERE id = ? AND guild_id = ? AND deleted = 0
               AND (scope = 'club' OR (scope = 'personal' AND owner_user_id = ?))""",
            (memory_id, principal.guild_id, principal.user_id),
        ).fetchone()

    @staticmethod
    def _record_revision(conn: sqlite3.Connection, record: dict, operation: str) -> None:
        conn.execute(
            """INSERT INTO memory_revisions
               (memory_id, version, guild_id, scope, owner_user_id, content,
                source_message_id, actor_id, timestamp, operation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (record["id"], record["version"], record["guild_id"], record["scope"],
             record["owner_user_id"], record["content"], record["source_message_id"],
             record["actor_id"], record["updated_at"], operation),
        )

    def create(
        self, principal: Principal, *, scope: str, content: str, source_message_id: int
    ) -> dict:
        self.policy.require_memory_access(principal, scope, write=True)
        self._validate_content(content)
        self._validate_source(source_message_id)
        timestamp = self._timestamp()
        record = dict(
            id=str(uuid4()), guild_id=principal.guild_id, scope=scope,
            owner_user_id=principal.user_id if scope == "personal" else 0,
            content=content, source_message_id=source_message_id,
            actor_id=principal.user_id, created_at=timestamp, updated_at=timestamp,
            version=1, deleted=0,
        )
        with self._connection(write=True) as conn:
            count = conn.execute(
                "SELECT count(*) FROM memories WHERE guild_id = ? AND scope = ? AND owner_user_id = ? AND deleted = 0",
                (principal.guild_id, scope, record["owner_user_id"]),
            ).fetchone()[0]
            if count >= self.MAX_RECORDS_PER_SCOPE:
                raise ValueError("Memory scope is full; update or delete existing records")
            conn.execute(
                """INSERT INTO memories (id, guild_id, scope, owner_user_id, content,
                   source_message_id, actor_id, created_at, updated_at, version, deleted)
                   VALUES (:id, :guild_id, :scope, :owner_user_id, :content,
                   :source_message_id, :actor_id, :created_at, :updated_at, :version, :deleted)""",
                record,
            )
            self._record_revision(conn, record, "create")
        return record

    def get(self, principal: Principal, memory_id: str) -> dict | None:
        self.policy.require_guild(principal)
        self._validate_memory_id(memory_id)
        with self._connection() as conn:
            row = self._read_visible(conn, principal, memory_id)
            return dict(row) if row is not None else None

    def search(
        self, principal: Principal, *, scope: str, query: str = "", limit: int = 20
    ) -> list[dict]:
        self.policy.require_memory_access(principal, scope)
        if not isinstance(query, str) or len(query) > self.MAX_QUERY_CHARS:
            raise ValueError(f"query must be at most {self.MAX_QUERY_CHARS} characters")
        if type(limit) is not int or not 1 <= limit <= self.MAX_RESULTS:
            raise ValueError(f"limit must be between 1 and {self.MAX_RESULTS}")
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM memories WHERE guild_id = ? AND scope = ? AND owner_user_id = ?
                   AND deleted = 0 AND content LIKE ? ESCAPE '\\'
                   ORDER BY updated_at DESC, id LIMIT ?""",
                (principal.guild_id, scope, principal.user_id if scope == "personal" else 0,
                 f"%{escaped}%", limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def update(
        self, principal: Principal, memory_id: str, *, content: str,
        source_message_id: int, expected_version: int
    ) -> dict:
        self._validate_content(content)
        return self._mutate(principal, memory_id, content=content,
                            source_message_id=source_message_id, expected_version=expected_version)

    def delete(
        self, principal: Principal, memory_id: str, *, source_message_id: int,
        expected_version: int
    ) -> None:
        """Soft-delete a visible record while retaining its previous contents in audit."""
        self._mutate(principal, memory_id, content=None,
                     source_message_id=source_message_id, expected_version=expected_version)

    def _mutate(
        self, principal: Principal, memory_id: str, *, content: str | None,
        source_message_id: int, expected_version: int
    ) -> dict:
        self.policy.require_guild(principal)
        self._validate_memory_id(memory_id)
        self._validate_source(source_message_id)
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive integer")
        with self._connection(write=True) as conn:
            row = self._read_visible(conn, principal, memory_id)
            if row is None:
                raise KeyError("Memory record not found")
            self.policy.require_memory_access(principal, row["scope"], write=True)
            if row["version"] != expected_version:
                raise MemoryConflict("Memory changed; read the current version before editing")
            record = dict(row)
            record.update(
                content=row["content"] if content is None else content,
                source_message_id=source_message_id, actor_id=principal.user_id,
                updated_at=self._timestamp(), version=expected_version + 1,
                deleted=int(content is None),
            )
            conn.execute(
                """UPDATE memories SET content=:content, source_message_id=:source_message_id,
                   actor_id=:actor_id, updated_at=:updated_at, version=:version, deleted=:deleted
                   WHERE id=:id""", record,
            )
            self._record_revision(conn, record, "delete" if content is None else "update")
            return record
