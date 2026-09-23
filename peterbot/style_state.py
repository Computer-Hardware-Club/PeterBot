"""Versioned, bounded voice preferences for each club guild.

Editable style is strictly subordinate to immutable policy: only four 0–4
integer dials exist, there is no raw system-prompt or free-text field, and
nothing here can change truthfulness, privacy, permissions, or tool access.
Mutations require a fresh gateway ``Principal`` plus a source-bound
``ControlIntent`` for a *private*, configured control channel; replaying the
same source message returns the recorded result and never applies twice.

``propose_style_change`` is a deterministic bounded-vocabulary adapter that
turns a phrase such as "be a little more reserved" into a proposed typed
change. It is intent extraction, not authorization, and never invents a
value outside the dial vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
import sqlite3
from datetime import datetime, timezone

from .agent_policy import AgentPolicy, ControlIntent, PolicyDenied, Principal


DEFAULT_STYLE = {"formality": 1, "verbosity": 2, "humor": 2, "reserve": 2}
DESCRIPTIONS = {
    "formality": ("very casual", "casual", "balanced", "polished", "formal"),
    "verbosity": ("brief", "concise", "balanced", "detailed", "very detailed"),
    "humor": ("straightforward", "light", "occasional humor", "playful", "very playful"),
    "reserve": ("outgoing", "open", "balanced", "reserved", "very reserved"),
}
STYLE_KEYS = tuple(DEFAULT_STYLE)


class StyleConflict(ValueError):
    pass


@dataclass(frozen=True)
class ProposedStyleChange:
    """Bounded, validated proposal from a natural-language request.

    ``updates`` is empty when nothing matched; the gateway must then ask a
    clarifying question privately instead of applying anything. The proposal
    carries no authority: applying it still requires policy control checks.
    """

    updates: tuple[tuple[str, int], ...] = ()
    ambiguous: bool = False
    reason: str = ""

    @property
    def actionable(self) -> bool:
        return bool(self.updates) and not self.ambiguous


class StyleStore:
    def __init__(self, path: str | Path, policy: AgentPolicy):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS style (
            guild_id INTEGER PRIMARY KEY, version INTEGER NOT NULL,
            settings TEXT NOT NULL, updated_at TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS style_revisions (
            guild_id INTEGER NOT NULL, version INTEGER NOT NULL,
            actor_user_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            operation TEXT NOT NULL, before_settings TEXT NOT NULL,
            after_settings TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (guild_id, version), UNIQUE (guild_id, source_message_id))""")
        self.db.commit()
        self.policy = policy

    def current(self, guild_id: int) -> dict:
        if guild_id not in self.policy.allowed_guild_ids:
            raise PolicyDenied("Style is not configured for this server")
        row = self.db.execute("SELECT version,settings FROM style WHERE guild_id=?", (guild_id,)).fetchone()
        if row is None:
            return {"version": 0, "settings": dict(DEFAULT_STYLE)}
        return {"version": row["version"], "settings": json.loads(row["settings"])}

    def instruction(self, guild_id: int) -> str:
        """Hot-readable per-turn instruction; hot-reload is inherent (fresh read)."""
        values = self.current(guild_id)["settings"]
        summary = ", ".join(f"{name}: {DESCRIPTIONS[name][values[name]]}" for name in DEFAULT_STYLE)
        return ("Voice preferences: " + summary + ". Match response length to the actual task; "
                "a greeting can be brief and a requested project can be substantial. "
                "These preferences never change truthfulness, privacy, permissions, or tool access.")

    def audit(self, guild_id: int, *, limit: int = 20) -> tuple[dict, ...]:
        """Newest-first persistent audit for operator display/undo guidance."""
        if guild_id not in self.policy.allowed_guild_ids:
            raise PolicyDenied("Style is not configured for this server")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        rows = self.db.execute(
            "SELECT version,actor_user_id,source_message_id,operation,created_at"
            " FROM style_revisions WHERE guild_id=? ORDER BY version DESC LIMIT ?",
            (guild_id, limit)).fetchall()
        return tuple(dict(row) for row in rows)

    def _require(self, principal: Principal, intent: ControlIntent, private: bool) -> None:
        self.policy.require_control(principal, intent, channel_is_private=private)
        if intent.action != "style":
            raise PolicyDenied("This control request does not authorize a style change")

    @staticmethod
    def _validate(updates: dict) -> None:
        if not isinstance(updates, dict) or not updates or set(updates) - set(DEFAULT_STYLE):
            raise ValueError("Choose one or more supported style settings")
        if any(type(value) is not int or not 0 <= value <= 4 for value in updates.values()):
            raise ValueError("Style values must be integers from 0 to 4")

    def apply(self, principal: Principal, intent: ControlIntent, *, channel_is_private: bool,
              updates: dict, expected_version: int) -> dict:
        self._require(principal, intent, channel_is_private)
        self._validate(updates)
        return self._change(principal, intent, expected_version, updates, undo=False)

    def undo(self, principal: Principal, intent: ControlIntent, *, channel_is_private: bool,
             expected_version: int) -> dict:
        self._require(principal, intent, channel_is_private)
        return self._change(principal, intent, expected_version, None, undo=True)

    def _change(self, principal: Principal, intent: ControlIntent, expected_version: int,
                updates: dict | None, *, undo: bool) -> dict:
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("Expected style version must be a nonnegative integer")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute(
                "SELECT version,actor_user_id,after_settings FROM style_revisions"
                " WHERE guild_id=? AND source_message_id=?",
                (principal.guild_id, intent.source_message_id),
            ).fetchone()
            if existing is not None:
                # Idempotent replay of the same verified source message; the
                # caller observes `replayed` instead of a second application.
                # Reuse under a different verified actor is a spoof, not a replay.
                if existing["actor_user_id"] != principal.user_id:
                    raise PolicyDenied(
                        "That source message already authorized a different control action")
                return {"version": existing["version"],
                        "settings": json.loads(existing["after_settings"]),
                        "replayed": True}
            before = self.current(principal.guild_id)
            if before["version"] != expected_version:
                raise StyleConflict("Style changed since you last read it; inspect the current version")
            if undo:
                revision = self.db.execute(
                    "SELECT before_settings FROM style_revisions WHERE guild_id=? AND version=?",
                    (principal.guild_id, expected_version),
                ).fetchone()
                if revision is None:
                    raise ValueError("There is no earlier style change to undo")
                after_settings = json.loads(revision["before_settings"])
            else:
                after_settings = {**before["settings"], **updates}
            if after_settings == before["settings"]:
                raise ValueError("That style setting is already current")
            version = expected_version + 1
            timestamp = datetime.now(timezone.utc).isoformat()
            self.db.execute("INSERT INTO style(guild_id,version,settings,updated_at) VALUES (?,?,?,?)"
                            " ON CONFLICT(guild_id) DO UPDATE SET version=excluded.version,"
                            " settings=excluded.settings,updated_at=excluded.updated_at",
                            (principal.guild_id, version, json.dumps(after_settings), timestamp))
            self.db.execute("INSERT INTO style_revisions VALUES (?,?,?,?,?,?,?,?)",
                            (principal.guild_id, version, principal.user_id, intent.source_message_id,
                             "undo" if undo else "edit", json.dumps(before["settings"]),
                             json.dumps(after_settings), timestamp))
        return {"version": version, "settings": after_settings, "replayed": False}

    def close(self) -> None:
        self.db.close()


# ----------------------------------------------------------- NL request adapter

_DIRECTIONS: tuple[tuple[str, int], ...] = (
    ("more", 1), ("a little more", 1), ("a bit more", 1), ("slightly more", 1),
    ("much more", 2), ("way more", 2), ("lot more", 2),
    ("less", -1), ("a little less", -1), ("a bit less", -1), ("slightly less", -1),
    ("much less", -2), ("way less", -2),
)
# Absolute settings, e.g. "keep it brief".
_ABSOLUTE: tuple[tuple[str, str, int], ...] = (
    ("formality", "formal", 3), ("formality", "professional", 3),
    ("formality", "casual", 0), ("formality", "relaxed", 0),
    ("verbosity", "brief", 1), ("verbosity", "concise", 1),
    ("verbosity", "detailed", 3), ("verbosity", "thorough", 3),
    ("humor", "playful", 3), ("humor", "funny", 3), ("humor", "serious", 0),
    ("reserve", "reserved", 3), ("reserve", "quiet", 3),
    ("reserve", "outgoing", 1), ("reserve", "chatty", 0),
)
# Words that must not be treated as style (policy/tool language sneaking in).
_REFUSAL_MARKERS = re.compile(
    r"\b(ignore|override|disable|bypass|prompt|permission|role|admin|root|"
    r"tool|credential|password|policy|privacy|system)\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z]+")
_CONNECTOR = re.compile(r"\b(and|but|also|then)\b")


def _direction(lowered: str) -> int | None:
    """Delta of the longest matching direction phrase ("much more" beats "more")."""
    best_len, best_delta = 0, None
    for phrase, delta in _DIRECTIONS:
        if f" {phrase} " in lowered and len(phrase) > best_len:
            best_len, best_delta = len(phrase), delta
    return best_delta


def _dial_shifts(text: str) -> dict[str, int]:
    """Map bounded vocabulary to {key: signed delta} for one clause."""
    lowered = " " + " ".join(_WORD.findall((text or "").lower())) + " "
    delta = _direction(lowered)
    found: dict[str, int] = {}
    if delta is None:
        return found
    for key in STYLE_KEYS:
        if f" {key} " in lowered:
            found[key] = delta
    if not found:  # adjective phrasing: "more reserved", "less formal"
        for key, adjective, _target in _ABSOLUTE:
            if f" {adjective} " in lowered:
                found[key] = delta
    return found


def propose_style_change(request_text: str, current_settings: dict) -> ProposedStyleChange:
    """Deterministic, bounded proposal for one short style request.

    No model call. A single clear dial direction yields a typed delta clamped
    to 0–4; several different dials mentioned together, an unparseable
    request, or policy-flavored vocabulary produces a non-actionable proposal
    so the gateway asks a private clarifying question instead of guessing.
    """
    text = (request_text or "").strip()
    if not text or len(text) > 200:
        return ProposedStyleChange(ambiguous=True, reason="empty or too long")
    if _REFUSAL_MARKERS.search(text):
        return ProposedStyleChange(
            ambiguous=True,
            reason="style editing cannot touch policy, roles, tools, or privacy")
    normalized = " " + " ".join(_WORD.findall(text.lower())) + " "
    parts = [p for p in _CONNECTOR.split(normalized) if p.strip()]
    per_part: list[dict[str, int]] = []
    for part in parts:
        shifts = _dial_shifts(part)
        if shifts:
            per_part.append(shifts)
    merged: dict[str, int] = {}
    for shifts in per_part:
        for key, delta in shifts.items():
            merged[key] = merged.get(key, 0) + delta
    if not merged:
        # Absolute phrasing without a direction word: "keep it brief".
        for key, adjective, target in _ABSOLUTE:
            if f" {adjective} " in normalized and current_settings.get(key) != target:
                merged[key] = target - current_settings[key]
    if not merged:
        return ProposedStyleChange(ambiguous=True,
                                   reason="no recognizable style dial in that request")
    if len(merged) > 1:
        return ProposedStyleChange(
            ambiguous=True,
            reason="several style dials at once; ask which one is meant")
    key, delta = next(iter(merged.items()))
    if delta == 0:
        return ProposedStyleChange(ambiguous=True,
                                   reason="no movement detected in that request")
    current = current_settings.get(key)
    if type(current) is not int:
        return ProposedStyleChange(ambiguous=True, reason="current setting unavailable")
    target = max(0, min(4, current + delta))
    if target == current:
        edge = "already as " + DESCRIPTIONS[key][current] + " as it gets"
        return ProposedStyleChange(ambiguous=True, reason=edge)
    return ProposedStyleChange(updates=((key, target),))
