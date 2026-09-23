"""Durable intent and receipt ledger for one authorized club announcement.

The nonce is a short-window Discord safeguard when a sender uses
``enforce_nonce``. The outbox's persisted states and operator reconciliation
remain necessary because Discord's nonce window lasts only a few minutes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import secrets
import sqlite3
from datetime import datetime, timezone

from .agent_policy import AgentPolicy, ControlIntent, PolicyDenied, Principal


MENTION = re.compile(r"@(?:everyone|here)\b|<@!?\d+>|<@&\d+>", re.I)
MAX_ATTEMPTS = 8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OutboxConflict(ValueError):
    pass


class AnnouncementOutbox:
    def __init__(self, path: str | Path, policy: AgentPolicy,
                 destinations: dict[int, frozenset[int]]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            self.db.close()
            raise ValueError("Announcement database is newer than this gateway")
        if version == 0:
            with self.db:
                self.db.execute("""CREATE TABLE IF NOT EXISTS announcements (
                    id TEXT PRIMARY KEY, guild_id INTEGER NOT NULL, actor_user_id INTEGER NOT NULL,
                    source_channel_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
                    target_channel_id INTEGER NOT NULL, content TEXT NOT NULL,
                    content_hash TEXT NOT NULL, nonce TEXT NOT NULL, status TEXT NOT NULL,
                    discord_message_id INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(guild_id, source_message_id))""")
                self.db.execute("PRAGMA user_version=1")
        with self.db:
            # The process died while Discord may have accepted the send.
            self.db.execute("UPDATE announcements SET status='unknown',updated_at=?"
                            " WHERE status='sending'", (_now(),))
        self.policy = policy
        self.destinations = destinations

    def _authorize(self, principal: Principal, intent: ControlIntent,
                   target_channel_id: int, private: bool) -> None:
        self.policy.require_control(principal, intent, channel_is_private=private)
        if intent.action != "announcement":
            raise PolicyDenied("This request does not authorize an announcement")
        if type(target_channel_id) is not int or target_channel_id not in self.destinations.get(principal.guild_id, frozenset()):
            raise PolicyDenied("That announcement destination is not configured")

    @staticmethod
    def _validate_content(content: str) -> str:
        if not isinstance(content, str) or not 1 <= len(content.strip()) <= 1800:
            raise ValueError("An announcement needs 1 to 1,800 characters")
        content = content.strip()
        if MENTION.search(content):
            raise ValueError("Mass, role and user mentions are disabled for announcements")
        return content

    def propose(self, principal: Principal, intent: ControlIntent, *,
                target_channel_id: int, content: str, channel_is_private: bool) -> dict:
        """Persist one explicit public payload before any Discord send."""
        self._authorize(principal, intent, target_channel_id, channel_is_private)
        content = self._validate_content(content)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            prior = self.db.execute("SELECT * FROM announcements WHERE guild_id=? AND source_message_id=?",
                                    (principal.guild_id, intent.source_message_id)).fetchone()
            if prior is not None:
                if (prior["actor_user_id"], prior["source_channel_id"], prior["target_channel_id"], prior["content_hash"]) != (
                        principal.user_id, principal.channel_id, target_channel_id, digest):
                    raise OutboxConflict("That request already has a different announcement intent")
                return dict(prior)
            action_id = secrets.token_hex(16)
            nonce = action_id[:24]
            timestamp = _now()
            self.db.execute("INSERT INTO announcements VALUES (?,?,?,?,?,?,?,?,?,?,NULL,0,?,?)",
                            (action_id, principal.guild_id, principal.user_id, principal.channel_id,
                             intent.source_message_id, target_channel_id, content, digest, nonce,
                             "pending", timestamp, timestamp))
        return self.get(action_id)

    def get(self, action_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM announcements WHERE id=?", (action_id,)).fetchone()
        return dict(row) if row else None

    def pending(self) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM announcements WHERE status='pending' ORDER BY created_at,id LIMIT 20")]

    def begin_send(self, action_id: str, principal: Principal, intent: ControlIntent,
                   *, channel_is_private: bool) -> bool:
        record = self.get(action_id)
        if record is None:
            return False
        self._authorize(principal, intent, record["target_channel_id"], channel_is_private)
        if (record["guild_id"], record["actor_user_id"], record["source_channel_id"],
                record["source_message_id"]) != (
                principal.guild_id, principal.user_id, principal.channel_id, intent.source_message_id):
            raise PolicyDenied("This announcement belongs to a different verified request")
        with self.db:
            changed = self.db.execute("UPDATE announcements SET status='sending',updated_at=?"
                                      " WHERE id=? AND status='pending'", (_now(), action_id))
        return changed.rowcount == 1

    def mark_sent(self, action_id: str, discord_message_id: int) -> bool:
        if type(discord_message_id) is not int or discord_message_id <= 0:
            raise ValueError("A Discord message receipt is required")
        with self.db:
            changed = self.db.execute("UPDATE announcements SET status='sent',discord_message_id=?,updated_at=?"
                                      " WHERE id=? AND status='sending'",
                                      (discord_message_id, _now(), action_id))
        return changed.rowcount == 1

    def mark_unknown(self, action_id: str) -> bool:
        with self.db:
            changed = self.db.execute("UPDATE announcements SET status='unknown',updated_at=?"
                                      " WHERE id=? AND status='sending'", (_now(), action_id))
        return changed.rowcount == 1

    def mark_denied(self, action_id: str) -> bool:
        """Block a queued intent before any send attempt has begun."""
        with self.db:
            changed = self.db.execute("UPDATE announcements SET status='denied',updated_at=?"
                                      " WHERE id=? AND status='pending'", (_now(), action_id))
        return changed.rowcount == 1

    def rejected_retry(self, action_id: str) -> str:
        """A known unsent response such as HTTP 429 can retry, with a bound."""
        with self.db:
            self.db.execute("UPDATE announcements SET attempts=attempts+1,updated_at=?,"
                            " status=CASE WHEN attempts+1>=? THEN 'failed' ELSE 'pending' END"
                            " WHERE id=? AND status='sending'", (_now(), MAX_ATTEMPTS, action_id))
        row = self.get(action_id)
        return row["status"] if row else "missing"

    def reconcile_unknown(self, action_id: str, *, confirmed_message_id: int | None = None,
                          retry_after_check: bool = False) -> bool:
        if retry_after_check == (confirmed_message_id is not None):
            raise ValueError("Choose either a verified Discord receipt or a checked retry")
        if confirmed_message_id is not None and (type(confirmed_message_id) is not int or confirmed_message_id <= 0):
            raise ValueError("Invalid Discord receipt")
        with self.db:
            if retry_after_check:
                changed = self.db.execute("UPDATE announcements SET status='pending',updated_at=?"
                                          " WHERE id=? AND status='unknown'", (_now(), action_id))
            else:
                changed = self.db.execute("UPDATE announcements SET status='sent',discord_message_id=?,updated_at=?"
                                          " WHERE id=? AND status='unknown'",
                                          (confirmed_message_id, _now(), action_id))
        return changed.rowcount == 1

    def receipt_url(self, action_id: str) -> str | None:
        record = self.get(action_id)
        if not record or record["status"] != "sent" or not record["discord_message_id"]:
            return None
        return (f"https://discord.com/channels/{record['guild_id']}/"
                f"{record['target_channel_id']}/{record['discord_message_id']}")

    def close(self) -> None:
        self.db.close()
