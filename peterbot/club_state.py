"""Durable, audience-scoped club facts and the published officer roster.

Separation of concerns enforced here:

- Published office facts are *statements about the club*, never authorization.
  This module never grants a Discord permission; ``AgentPolicy`` never reads it.
  Only gateway-supplied ``Principal``/``ControlIntent`` objects built from
  verified Discord IDs participate in control decisions.
- Mutations require a source-bound ``ControlIntent`` whose action class matches
  the edit (``roster`` for offices, ``club_fact`` for facts), a private
  configured control channel, and an optimistic expected version.
- Replaying the same source message returns the recorded result; it never
  applies twice and never applies under a different actor.
- Holders are stable member IDs supplied by the trusted gateway directory.
  Names are resolved against that directory only; ambiguity or an unknown name
  raises instead of inventing an officer. Display names/labels are decorative.
- Facts carry an explicit ``public``/``private`` classification. Private rows
  never enter ``snapshot()``/``chat_context()`` output, which is the only path
  intended for model context. Private officer-channel discussion is not
  published automatically; only the explicitly requested fact is stored.
- Undo reverts the latest committed change of the matching action class by
  restoring its recorded before-state; history stays append-only.

The database must live outside the agent sandbox, writable only by the
gateway process (same boundary as ``ScopedMemoryStore``).
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

from .agent_policy import AgentPolicy, ControlIntent, PolicyDenied, Principal, _valid_id
from .knowledge import (build_knowledge_excerpt, chunk_is_expired, chunk_is_superseded,
                        rank_knowledge_chunks, tokenize_relevance)


class ClubStateConflict(ValueError):
    """The caller read a stale version; re-read before retrying."""


class UnresolvedIdentityError(ValueError):
    """A requested name has no stable member ID in the trusted directory."""


class AmbiguousIdentityError(ValueError):
    """A requested name maps to several member IDs; ask instead of guessing."""


OFFICE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
FACT_KEY_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")

OFFICER_ROSTER_SUPERSESSION_KEY = "officer_roster"
# Fact key -> static heading slugs it supersedes (the live roster wins over an
# older "Officers" section in the static knowledge file).
SUPERSESSION_ALIASES: Mapping[str, frozenset[str]] = {
    OFFICER_ROSTER_SUPERSESSION_KEY: frozenset({"officer", "officers", "officer_roster"}),
}


@dataclass(frozen=True)
class OfficerRequest:
    """One requested office from natural language, before ID resolution."""

    office: str
    name: str


@dataclass(frozen=True)
class OfficeAssignment:
    """One resolved office: a stable member ID, plus a decorative label."""

    office: str
    holder_user_id: int
    holder_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.office, str) or not OFFICE_SLUG.match(self.office):
            raise ValueError("office must be a lowercase slug (e.g. 'vice_president')")
        if not _valid_id(self.holder_user_id):
            raise ValueError("holder_user_id must be a Discord member ID")
        if not isinstance(self.holder_label, str) or not self.holder_label.strip():
            raise ValueError("holder_label is required for the published roster")
        if len(self.holder_label) > 96 or any(ord(c) < 32 for c in self.holder_label):
            raise ValueError("holder_label must be a short printable name")


@dataclass(frozen=True)
class ClubSnapshot:
    """Bounded, audience-safe view intended for model context."""

    version: int
    offices: tuple[dict, ...]
    facts: tuple[dict, ...]
    text: str


def normalize_person_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("person names must be strings")
    return " ".join(name.casefold().split())


def resolve_officers(
    requests: Sequence[OfficerRequest],
    directory: Mapping[str, int],
) -> tuple[OfficeAssignment, ...]:
    """Resolve requested names to stable member IDs from a trusted directory.

    ``directory`` maps member names to Discord member IDs and must be built by
    the gateway from verified membership data. The first case-insensitive name
    match wins; two distinct IDs under one normalized name is ambiguous and is
    rejected without guessing.
    """
    if not isinstance(directory, Mapping):
        raise ValueError("directory must map member names to Discord IDs")
    by_name: dict[str, dict[int, str]] = {}
    for raw_name, user_id in directory.items():
        if not _valid_id(user_id):
            raise ValueError("directory member IDs must be Discord IDs")
        key = normalize_person_name(raw_name)
        if not key:
            continue
        by_name.setdefault(key, {})[int(user_id)] = str(raw_name)
    assignments: list[OfficeAssignment] = []
    seen_offices: set[str] = set()
    for request in requests:
        if not isinstance(request, OfficerRequest):
            raise ValueError("requests must be OfficerRequest instances")
        candidates = by_name.get(normalize_person_name(request.name), {})
        if not candidates:
            raise UnresolvedIdentityError(
                f"No club member named {request.name!r} in the server directory; "
                "refusing to invent an officer")
        if len(candidates) > 1:
            raise AmbiguousIdentityError(
                f"{request.name!r} matches several members "
                f"({', '.join(sorted(candidates.values()))}); ask which one")
        holder_user_id, holder_label = next(iter(candidates.items()))
        if request.office in seen_offices:
            raise ValueError(f"office {request.office!r} requested twice")
        seen_offices.add(request.office)
        assignments.append(OfficeAssignment(request.office, holder_user_id, holder_label))
    return tuple(assignments)


class ClubStateStore:
    MAX_OFFICES = 24
    MAX_FACTS = 128
    MAX_VALUE_CHARS = 600
    MAX_TERM_CHARS = 96
    MAX_ITEMS = 40

    def __init__(self, path: str | Path, policy: AgentPolicy) -> None:
        self.path = Path(path)
        self.policy = policy
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection(write=True) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS club_guild_state (
                    guild_id INTEGER PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS club_offices (
                    guild_id INTEGER NOT NULL,
                    office TEXT NOT NULL,
                    holder_user_id INTEGER NOT NULL,
                    holder_label TEXT NOT NULL,
                    term TEXT NOT NULL,
                    effective_from TEXT,
                    expires_at TEXT,
                    actor_id INTEGER NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, office)
                );
                CREATE TABLE IF NOT EXISTS club_facts (
                    guild_id INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    visibility TEXT NOT NULL CHECK(visibility IN ('public', 'private')),
                    actor_id INTEGER NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, key)
                );
                CREATE TABLE IF NOT EXISTS club_revisions (
                    guild_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    source_message_id INTEGER NOT NULL,
                    actor_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('club_fact', 'roster')),
                    operation TEXT NOT NULL,
                    before_state TEXT NOT NULL,
                    after_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, version),
                    UNIQUE (guild_id, source_message_id)
                );
                CREATE TRIGGER IF NOT EXISTS club_revisions_no_update
                    BEFORE UPDATE ON club_revisions
                    BEGIN SELECT RAISE(ABORT, 'club revision ledger is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS club_revisions_no_delete
                    BEFORE DELETE ON club_revisions
                    BEGIN SELECT RAISE(ABORT, 'club revision ledger is append-only'); END;
            """)

    # ---------------------------------------------------------------- plumbing

    @contextmanager
    def _connection(self, *, write: bool = False):
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

    def _require_guild(self, guild_id: object) -> None:
        # Reads are guild-scoped fail-closed like writes: an unconfigured or
        # non-guild ID can never observe club state, even by guessing IDs.
        if not _valid_id(guild_id) or guild_id not in self.policy.allowed_guild_ids:
            raise PolicyDenied("Club state is not available for this server")

    def _require_control(self, principal: Principal, intent: ControlIntent,
                         *, channel_is_private: bool, action: str) -> None:
        self.policy.require_control(principal, intent, channel_is_private=channel_is_private)
        if intent.action != action:
            raise PolicyDenied(
                f"This control request authorizes {intent.action!r}, not {action!r}")

    def _load_state(self, conn: sqlite3.Connection, guild_id: int) -> dict:
        offices = [dict(row) for row in conn.execute(
            "SELECT * FROM club_offices WHERE guild_id = ? ORDER BY office", (guild_id,))]
        facts = [dict(row) for row in conn.execute(
            "SELECT * FROM club_facts WHERE guild_id = ? ORDER BY key", (guild_id,))]
        return {"offices": offices, "facts": facts}

    def _current_version(self, conn: sqlite3.Connection, guild_id: int) -> int:
        row = conn.execute(
            "SELECT version FROM club_guild_state WHERE guild_id = ?", (guild_id,)).fetchone()
        return 0 if row is None else row["version"]

    @staticmethod
    def _validate_term(term: object, effective_from: object, expires_at: object) -> str:
        if not isinstance(term, str) or not term.strip() or len(term) > ClubStateStore.MAX_TERM_CHARS \
                or any(ord(c) < 32 for c in term):
            raise ValueError(f"term must be a short single-line label (e.g. 'Fall 2026')")
        dates: list[date] = []
        for name, value in (("effective_from", effective_from), ("expires_at", expires_at)):
            if value is None:
                continue
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError(f"{name} must be an ISO date string or None")
            try:
                dates.append(date.fromisoformat(value))
            except ValueError:
                raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from None
        if len(dates) == 2 and dates[0] > dates[1]:
            raise ValueError("effective_from must not be after expires_at")
        return term.strip()

    def _validate_offices(self, assignments: Sequence[OfficeAssignment]) -> None:
        if not isinstance(assignments, Sequence) or isinstance(assignments, (str, bytes)) \
                or not assignments:
            raise ValueError("at least one OfficeAssignment is required")
        if len(assignments) > self.MAX_OFFICES:
            raise ValueError(f"the roster holds at most {self.MAX_OFFICES} offices")
        seen: set[str] = set()
        for assignment in assignments:
            if not isinstance(assignment, OfficeAssignment):
                raise ValueError("roster entries must be OfficeAssignment instances")
            if assignment.office in seen:
                raise ValueError(f"office {assignment.office!r} appears twice")
            seen.add(assignment.office)

    @staticmethod
    def _validate_fact(key: object, value: object) -> tuple[str, str]:
        if not isinstance(key, str) or not FACT_KEY_SLUG.match(key):
            raise ValueError("fact key must be a lowercase slug (e.g. 'meeting_day')")
        if not isinstance(value, str) or not value.strip() or len(value) > ClubStateStore.MAX_VALUE_CHARS \
                or any(ord(c) < 32 for c in value):
            raise ValueError(f"fact value must be 1–{ClubStateStore.MAX_VALUE_CHARS} printable characters")
        return key, value.strip()

    # -------------------------------------------------------------- mutations

    def _replay_result(self, conn: sqlite3.Connection, principal: Principal,
                       intent: ControlIntent) -> dict | None:
        row = conn.execute(
            "SELECT * FROM club_revisions WHERE guild_id = ? AND source_message_id = ?",
            (principal.guild_id, intent.source_message_id)).fetchone()
        if row is None:
            return None
        if row["actor_id"] != principal.user_id or row["action"] != intent.action:
            raise PolicyDenied(
                "That source message already authorized a different control action")
        return {"version": row["version"],
                "state": json.loads(row["after_state"]),
                "replayed": True}

    def _require_version(self, conn: sqlite3.Connection, guild_id: int,
                         expected_version: int) -> None:
        if self._current_version(conn, guild_id) != expected_version:
            raise ClubStateConflict(
                "Club state changed since it was read; re-read the current version")

    def _commit(self, conn: sqlite3.Connection, before_state: dict, *, principal: Principal,
                intent: ControlIntent, expected_version: int, operation: str) -> dict:
        # Defensive re-check; callers validate before mutating.
        self._require_version(conn, principal.guild_id, expected_version)
        timestamp = self._timestamp()
        version = expected_version + 1
        after = self._load_state(conn, principal.guild_id)
        conn.execute(
            """INSERT INTO club_guild_state (guild_id, version, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(guild_id) DO UPDATE SET version = excluded.version,
               updated_at = excluded.updated_at""",
            (principal.guild_id, version, timestamp))
        conn.execute(
            """INSERT INTO club_revisions
               (guild_id, version, source_message_id, actor_id, action, operation,
                before_state, after_state, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (principal.guild_id, version, intent.source_message_id, principal.user_id,
             intent.action, operation, json.dumps(before_state), json.dumps(after), timestamp))
        return {"version": version, "state": after, "replayed": False}

    def set_officers(self, principal: Principal, intent: ControlIntent,
                     assignments: Sequence[OfficeAssignment], *,
                     channel_is_private: bool, term: str,
                     effective_from: str | None = None, expires_at: str | None = None,
                     replace_all: bool = True, expected_version: int) -> dict:
        """Atomically publish office holders (a reshuffle or a partial amend).

        ``replace_all=True`` replaces the entire current roster, so a shuffle
        supersedes conflicting records instead of accumulating contradictory
        prose. ``role_requirements``/``member_roles`` are gateway-supplied and
        only ever produce advisory notes; they never change policy.
        """
        self._require_control(principal, intent,
                              channel_is_private=channel_is_private, action="roster")
        self._validate_offices(assignments)
        self._validate_term(term, effective_from, expires_at)
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("expected_version must be a nonnegative integer")
        if type(replace_all) is not bool:
            raise ValueError("replace_all must be a boolean")
        with self._connection(write=True) as conn:
            replay = self._replay_result(conn, principal, intent)
            if replay is not None:
                return replay
            self._require_version(conn, principal.guild_id, expected_version)
            before = self._load_state(conn, principal.guild_id)
            if replace_all:
                conn.execute(
                    "DELETE FROM club_offices WHERE guild_id = ?", (principal.guild_id,))
            timestamp = self._timestamp()
            for assignment in assignments:
                conn.execute(
                    """INSERT INTO club_offices
                       (guild_id, office, holder_user_id, holder_label, term,
                        effective_from, expires_at, actor_id, source_message_id, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(guild_id, office) DO UPDATE SET
                       holder_user_id = excluded.holder_user_id,
                       holder_label = excluded.holder_label,
                       term = excluded.term,
                       effective_from = excluded.effective_from,
                       expires_at = excluded.expires_at,
                       actor_id = excluded.actor_id,
                       source_message_id = excluded.source_message_id,
                       updated_at = excluded.updated_at""",
                    (principal.guild_id, assignment.office, assignment.holder_user_id,
                     assignment.holder_label.strip(), term.strip(), effective_from,
                     expires_at, principal.user_id, intent.source_message_id, timestamp))
            return self._commit(conn, before, principal=principal, intent=intent,
                                expected_version=expected_version,
                                operation="officers_set")

    def set_fact(self, principal: Principal, intent: ControlIntent, *,
                 channel_is_private: bool, key: str, value: str, visibility: str,
                 expected_version: int) -> dict:
        """Commit one club fact under an explicit public/private classification."""
        self._require_control(principal, intent,
                              channel_is_private=channel_is_private, action="club_fact")
        key, value = self._validate_fact(key, value)
        if visibility not in ("public", "private"):
            raise ValueError("visibility must be 'public' or 'private'")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("expected_version must be a nonnegative integer")
        with self._connection(write=True) as conn:
            replay = self._replay_result(conn, principal, intent)
            if replay is not None:
                return replay
            self._require_version(conn, principal.guild_id, expected_version)
            before = self._load_state(conn, principal.guild_id)
            existing = conn.execute(
                "SELECT key FROM club_facts WHERE guild_id = ? AND key = ?",
                (principal.guild_id, key)).fetchone()
            if existing is None:
                count = conn.execute(
                    "SELECT count(*) FROM club_facts WHERE guild_id = ?",
                    (principal.guild_id,)).fetchone()[0]
                if count >= self.MAX_FACTS:
                    raise ValueError(f"at most {self.MAX_FACTS} club facts are stored; "
                                     "forget one first")
            conn.execute(
                """INSERT INTO club_facts
                   (guild_id, key, value, visibility, actor_id, source_message_id, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id, key) DO UPDATE SET
                   value = excluded.value, visibility = excluded.visibility,
                   actor_id = excluded.actor_id,
                   source_message_id = excluded.source_message_id,
                   updated_at = excluded.updated_at""",
                (principal.guild_id, key, value, visibility, principal.user_id,
                 intent.source_message_id, self._timestamp()))
            return self._commit(conn, before, principal=principal, intent=intent,
                                expected_version=expected_version, operation="fact_set")

    def forget_fact(self, principal: Principal, intent: ControlIntent, *,
                    channel_is_private: bool, key: str, expected_version: int) -> dict:
        self._require_control(principal, intent,
                              channel_is_private=channel_is_private, action="club_fact")
        if not isinstance(key, str) or not FACT_KEY_SLUG.match(key):
            raise ValueError("fact key must be a lowercase slug")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("expected_version must be a nonnegative integer")
        with self._connection(write=True) as conn:
            replay = self._replay_result(conn, principal, intent)
            if replay is not None:
                return replay
            self._require_version(conn, principal.guild_id, expected_version)
            before = self._load_state(conn, principal.guild_id)
            removed = conn.execute(
                "DELETE FROM club_facts WHERE guild_id = ? AND key = ?",
                (principal.guild_id, key)).rowcount
            if not removed:
                raise KeyError(f"No club fact named {key!r}")
            return self._commit(conn, before, principal=principal, intent=intent,
                                expected_version=expected_version, operation="fact_forget")

    def undo(self, principal: Principal, intent: ControlIntent, *,
             channel_is_private: bool, expected_version: int) -> dict:
        """Revert the latest committed change by restoring its before-state.

        The intent's action class must match the revision being undone, so a
        roster undo cannot silently roll back fact edits (and vice versa).
        """
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive integer")
        with self._connection(write=True) as conn:
            # Control is required before any revision contents are revealed; the
            # action class, however, can only be checked against the revision.
            self.policy.require_control(principal, intent,
                                        channel_is_private=channel_is_private)
            replay = self._replay_result(conn, principal, intent)
            if replay is not None:
                return replay
            revision = conn.execute(
                "SELECT * FROM club_revisions WHERE guild_id = ? AND version = ?",
                (principal.guild_id, expected_version)).fetchone()
            if revision is None:
                raise ValueError("There is no committed change at that version to undo")
            if revision["action"] != intent.action:
                raise PolicyDenied(
                    f"Version {expected_version} was a {revision['action']!r} change; "
                    f"undo it with a {revision['action']!r} control request")
            current = self._current_version(conn, principal.guild_id)
            if current != expected_version:
                raise ClubStateConflict(
                    "Only the latest change can be undone; re-read the current state")
            before = json.loads(revision["before_state"])
            current_state = self._load_state(conn, principal.guild_id)
            timestamp = self._timestamp()
            if intent.action == "roster":
                conn.execute("DELETE FROM club_offices WHERE guild_id = ?",
                             (principal.guild_id,))
                for row in before["offices"]:
                    conn.execute(
                        """INSERT INTO club_offices
                           (guild_id, office, holder_user_id, holder_label, term,
                            effective_from, expires_at, actor_id, source_message_id, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (principal.guild_id, row["office"], row["holder_user_id"],
                         row["holder_label"], row["term"], row["effective_from"],
                         row["expires_at"], row["actor_id"], row["source_message_id"],
                         timestamp))
            else:
                conn.execute("DELETE FROM club_facts WHERE guild_id = ?",
                             (principal.guild_id,))
                for row in before["facts"]:
                    conn.execute(
                        """INSERT INTO club_facts
                           (guild_id, key, value, visibility, actor_id,
                            source_message_id, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (principal.guild_id, row["key"], row["value"], row["visibility"],
                         row["actor_id"], row["source_message_id"], timestamp))
            return self._commit(conn, current_state, principal=principal, intent=intent,
                                expected_version=expected_version, operation="undo")

    # ---------------------------------------------------------------- queries

    def current(self, guild_id: int) -> dict:
        """Full internal state (operator inspection only; includes private)."""
        self._require_guild(guild_id)
        with self._connection() as conn:
            state = self._load_state(conn, guild_id)
            return {"version": self._current_version(conn, guild_id), **state}

    def public_officers(self, guild_id: int, *, today: date | None = None) -> tuple[dict, ...]:
        self._require_guild(guild_id)
        as_of = (today or datetime.now(timezone.utc).date()).isoformat()
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT office, holder_user_id, holder_label, term, effective_from,
                          expires_at FROM club_offices WHERE guild_id = ?
                   AND (effective_from IS NULL OR effective_from <= ?)
                   AND (expires_at IS NULL OR expires_at >= ?)
                   ORDER BY office""", (guild_id, as_of, as_of)).fetchall()
        return tuple(dict(row) for row in rows)

    def public_facts(self, guild_id: int) -> tuple[dict, ...]:
        """Committed public facts only; visibility is enforced in SQL."""
        self._require_guild(guild_id)
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT key, value, updated_at FROM club_facts
                   WHERE guild_id = ? AND visibility = 'public'""", (guild_id,)).fetchall()
        return tuple(dict(row) for row in rows)

    def snapshot(self, guild_id: int, *, query: str = "", static_chunks: Sequence = (),
                 max_items: int = 20, max_chars: int = 2200,
                 today: date | None = None) -> ClubSnapshot:
        """One bounded, fresh, audience-safe fact view for fast chat, /ask and Hermes.

        Reads the latest committed version on every call (no cache to
        invalidate). Offices are always included so a roster never disappears
        behind unrelated edits; other public facts are ranked against the
        query, then by recency. Static knowledge chunks are only background:
        chunks superseded by a live fact are dropped, and chunks past their
        ``expires`` date are dropped, so older file text can never contradict
        a newer committed record. Private rows never appear.
        """
        self._require_guild(guild_id)
        if not isinstance(query, str) or len(query) > 300:
            raise ValueError("query must be at most 300 characters")
        if type(max_items) is not int or not 1 <= max_items <= self.MAX_ITEMS:
            raise ValueError(f"max_items must be between 1 and {self.MAX_ITEMS}")
        if type(max_chars) is not int or not 128 <= max_chars <= 8000:
            raise ValueError("max_chars must be between 128 and 8000")
        as_of = today or datetime.now(timezone.utc).date()
        with self._connection() as conn:
            version = self._current_version(conn, guild_id)
            offices = tuple(dict(row) for row in conn.execute(
                """SELECT office, holder_user_id, holder_label, term, effective_from,
                          expires_at FROM club_offices WHERE guild_id = ?
                   AND (effective_from IS NULL OR effective_from <= ?)
                   AND (expires_at IS NULL OR expires_at >= ?)
                   ORDER BY office""", (guild_id, as_of.isoformat(), as_of.isoformat())))
            roster_recorded = conn.execute(
                "SELECT 1 FROM club_offices WHERE guild_id=? LIMIT 1", (guild_id,)).fetchone() is not None
            facts = [dict(row) for row in conn.execute(
                """SELECT key, value, updated_at FROM club_facts
                   WHERE guild_id = ? AND visibility = 'public'""", (guild_id,))]
        query_tokens = set(tokenize_relevance(query))
        # Stable two-pass sort: newest committed first, then relevance bucket
        # (overlapping the query wins), so a fresh relevant fact is always on top.
        facts.sort(key=lambda f: f["updated_at"], reverse=True)
        if query_tokens:
            def bucket(fact: dict) -> int:
                return 0 if query_tokens.intersection(tokenize_relevance(
                    f"{fact['key'].replace('_', ' ')} {fact['value']}")) else 1
            facts.sort(key=bucket)
        # Supersession uses every committed public fact, not just the shown top-N.
        live_keys = {fact["key"] for fact in facts}
        facts = tuple(facts[:max_items])
        lines: list[str] = []
        if offices:
            terms = sorted({o["term"] for o in offices})
            term_note = ", ".join(terms) if len(terms) == 1 else "mixed terms"
            roster = "; ".join(
                f"{o['office'].replace('_', ' ')}: {o['holder_label']}" for o in offices)
            lines.append(f"Current officer roster ({term_note}): {roster}.")
        for fact in facts:
            lines.append(f"- {fact['key'].replace('_', ' ')}: {fact['value']}")
        if roster_recorded:
            live_keys.add(OFFICER_ROSTER_SUPERSESSION_KEY)
        text = ""
        budget = max_chars
        if lines:
            text = ("Current authoritative club facts (latest officer-confirmed edits "
                    "win over any older text):\n" + "\n".join(lines))
        if len(text) > max_chars:
            text = text[:max_chars - 1].rstrip() + "…"
        budget = max_chars - len(text)
        if static_chunks and budget > 200:
            candidates = [
                chunk for chunk in static_chunks
                if not chunk_is_superseded(chunk, live_keys, SUPERSESSION_ALIASES)
                and not chunk_is_expired(chunk, as_of)]
            ranked = rank_knowledge_chunks(query, candidates, max_chunks=6) if query else []
            excerpt = build_knowledge_excerpt(ranked, max_chars=budget) if ranked else None
            if excerpt:
                text += ("\n\nBackground reference (static; older than the facts above "
                         "and never authoritative over them):\n" + excerpt)
        return ClubSnapshot(version=version, offices=offices, facts=facts, text=text)

    def chat_context(self, guild_id: int, query: str, *,
                     static_chunks: Sequence = (), max_chars: int = 2200) -> tuple[str, int]:
        """Adapter for the gateway: returns (context text, committed version)."""
        snap = self.snapshot(guild_id, query=query, static_chunks=static_chunks,
                             max_chars=max_chars)
        return snap.text, snap.version

    def role_advice(self, guild_id: int, role_requirements: Mapping[str, int],
                    member_roles: Mapping[int, Sequence[int]]) -> tuple[str, ...]:
        """Advisory-only mismatch notes for a private operator receipt.

        Both mappings must come from the gateway's fresh Discord data. This
        never authorizes or blocks anything: Discord roles decide permissions,
        the roster only states facts.
        """
        if not isinstance(role_requirements, Mapping) or not role_requirements:
            return ()
        notes: list[str] = []
        for office in self.public_officers(guild_id):
            required = role_requirements.get(office["office"])
            if required is None or not _valid_id(required):
                continue
            held = member_roles.get(office["holder_user_id"], ())
            if not isinstance(held, (tuple, list, set, frozenset)) \
                    or required not in held:
                notes.append(
                    f"{office['office'].replace('_', ' ')} {office['holder_label']} "
                    "does not currently hold the configured Discord role; the "
                    "published roster is factual only and granted no access")
        return tuple(notes)
