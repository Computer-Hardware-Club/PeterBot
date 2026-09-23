"""Private, aggregate-only operator diagnostics and explicit retention (PETER-16).

Invariants:

* ``diagnose()`` and ``retention_plan()`` never mutate state. Deletion happens
  only in ``retention_apply()``, which the operator invokes explicitly, and
  only after a verified snapshot (see ``deploy/state_backup.py``).
* Output contains counts, status histograms, schema versions, and ages only:
  never prompts, answers, memory text, Discord IDs, job IDs, tokens, or file
  paths. Degradation reasons are one of a fixed token set; raw exceptions
  (which can embed paths) are never surfaced.
* Stores are opened read-only wherever possible. This module never touches
  the network, the model, or Discord, and never posts anything.
* Retention never deletes: active jobs (``preparing``/``queued``/``running``),
  anything with ``unknown`` or ``exhausted`` delivery, any outbox record that
  is not a settled ``sent`` receipt, and all club/style/memory state and
  audit revisions. Project data is pruned only by
  ``ProjectStore.retention_sweep()`` under its own ``ProjectSettings`` policy
  — never by ad-hoc SQL here.
* Memory ``/forget`` is a soft-forget: the visible row is flagged ``deleted``
  and the previous content remains in the append-only ``memory_revisions``
  ledger (enforced by triggers). Nothing in this module hard-deletes memory;
  true deletion is offline snapshot destruction, documented in
  ``docs/ops-and-retention.md``.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent_jobs import PREPARING, QUEUED, RUNNING, TERMINAL_STATUSES

# Store files are discovered by these candidate names; the first present wins.
# `tasks.sqlite3` is the pre-rename gateway name and stays a recognized alias.
COMPONENTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...], int | None], ...] = (
    # name, candidate filenames, required tables, max known schema version
    ("jobs", ("jobs.sqlite3", "tasks.sqlite3"), ("jobs", "ingress"), None),
    ("memory", ("memory.sqlite3",), ("memories", "memory_revisions"), None),
    ("foreground", ("foreground.sqlite3",), ("requests", "lease"), None),
    ("club", ("club.sqlite3",), ("club_guild_state", "club_revisions"), None),
    ("style", ("style.sqlite3",), ("style", "style_revisions"), None),
    ("conversation", ("conversations.sqlite3", "conversation.sqlite3"), ("conversation_turns",), 1),
    ("outbox", ("outbox.sqlite3", "announcements.sqlite3"), ("announcements",), 1),
    ("metrics", ("metrics.sqlite3",), ("stage_metrics",), 1),
)
_PROJECTS_DIR = "projects"
_PROJECT_DB = "projects.sqlite"
_ACTIVE_STATUSES = (PREPARING, QUEUED, RUNNING)
_TERMINAL_SQL = ",".join("?" for _ in TERMINAL_STATUSES)
_TERMINAL = tuple(sorted(TERMINAL_STATUSES))

REASONS = frozenset({"schema_newer", "missing_tables", "corrupt_or_unreadable"})


class _Degraded(Exception):
    """Internal marker carrying one fixed reason token (never a raw error)."""


@dataclass(frozen=True)
class Category:
    """One retention category: identical predicate for plan and apply."""

    name: str
    component: str
    count_sql: str
    delete_sql: str
    days_attr: str

    def cutoff_args(self, cutoff: str) -> tuple:
        if self.component == "jobs":
            return (*_TERMINAL, cutoff)
        return (cutoff,)


CATEGORIES: tuple[Category, ...] = (
    Category(
        "conversations", "conversation",
        "SELECT COUNT(*) FROM conversation_turns WHERE created_at<?",
        "DELETE FROM conversation_turns WHERE created_at<?",
        "conversations_days"),
    Category(
        "metrics", "metrics",
        "SELECT COUNT(*) FROM stage_metrics WHERE at<?",
        "DELETE FROM stage_metrics WHERE at<?",
        "metrics_days"),
    Category(
        "terminal_jobs", "jobs",
        "SELECT COUNT(*) FROM jobs WHERE status IN (" + _TERMINAL_SQL + ") AND delivered=1"
        " AND delivery_status IN ('delivered','withheld') AND updated_at<?",
        "DELETE FROM jobs WHERE status IN (" + _TERMINAL_SQL + ") AND delivered=1"
        " AND delivery_status IN ('delivered','withheld') AND updated_at<?",
        "terminal_jobs_days"),
    Category(
        "settled_receipts", "outbox",
        "SELECT COUNT(*) FROM announcements WHERE status='sent' AND updated_at<?",
        "DELETE FROM announcements WHERE status='sent' AND updated_at<?",
        "settled_receipts_days"),
)


@dataclass(frozen=True)
class RetentionConfig:
    """Operator-configurable TTLs, in whole days. Nothing is deleted by default."""

    conversations_days: int = 90
    metrics_days: int = 30
    terminal_jobs_days: int = 90
    settled_receipts_days: int = 180
    include_projects: bool = False

    def __post_init__(self) -> None:
        for name in ("conversations_days", "metrics_days", "terminal_jobs_days",
                     "settled_receipts_days"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer number of days")
        if type(self.include_projects) is not bool:
            raise ValueError("include_projects must be a boolean")


def _now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


def _cutoff(now: datetime, days: int) -> str:
    return (now - timedelta(days=days)).isoformat()


def _component_names() -> dict[str, tuple[str, ...]]:
    return {name: files for name, files, _, _ in COMPONENTS}


def _find(state: Path, name: str) -> Path | None:
    for directory in (state, state / 'hermes'):
        for candidate in _component_names()[name]:
            path = directory / candidate
            if os.path.lexists(path):
                return path
    return None


def _projects_root(state: Path) -> Path:
    direct = state / _PROJECTS_DIR
    nested = state / 'hermes' / _PROJECTS_DIR
    return direct if os.path.lexists(direct) or not os.path.lexists(nested) else nested


def _projects_db(state: Path) -> Path:
    return _projects_root(state) / _PROJECT_DB


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _check_database(db: sqlite3.Connection, tables: tuple[str, ...],
                    max_schema: int | None) -> int:
    schema = db.execute("PRAGMA user_version").fetchone()[0]
    if max_schema is not None and schema > max_schema:
        raise _Degraded("schema_newer")
    present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not set(tables) <= present:
        raise _Degraded("missing_tables")
    return schema


def component_health(state_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Per-component status: ok | offline | degraded (fixed reasons only)."""
    state = Path(state_dir)
    health: dict[str, dict[str, Any]] = {}
    for name, _files, tables, max_schema in COMPONENTS:
        path = _find(state, name)
        if path is None:
            health[name] = {"status": "offline"}
            continue
        try:
            with closing(_readonly(path)) as db:
                schema = _check_database(db, tables, max_schema)
            health[name] = {"status": "ok", "schema": schema}
        except _Degraded as exc:
            health[name] = {"status": "degraded", "reason": str(exc)}
        except sqlite3.Error:
            health[name] = {"status": "degraded", "reason": "corrupt_or_unreadable"}
    db_path = _projects_db(state)
    if not os.path.lexists(db_path):
        health["projects"] = {"status": "offline"}
    else:
        try:
            with closing(_readonly(db_path)) as db:
                _check_database(db, ("projects", "versions", "files"), 1)
            health["projects"] = {"status": "ok"}
        except _Degraded as exc:
            health["projects"] = {"status": "degraded", "reason": str(exc)}
        except sqlite3.Error:
            health["projects"] = {"status": "degraded", "reason": "corrupt_or_unreadable"}
    return health


def _iso_age(created_at: str, now: datetime) -> int | None:
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0, int((now - moment).total_seconds()))


def _group_counts(db: sqlite3.Connection, sql: str) -> dict[str, int]:
    return {row[0]: row[1] for row in db.execute(sql)}


def _jobs_section(state: Path, now: datetime) -> dict | None:
    path = _find(state, "jobs")
    if path is None:
        return None
    try:
        with closing(_readonly(path)) as db:
            section = {
                "by_status": _group_counts(db, "SELECT status, COUNT(*) FROM jobs GROUP BY status"),
                "delivery": _group_counts(
                    db, "SELECT delivery_status, COUNT(*) FROM jobs GROUP BY delivery_status"),
            }
            oldest = db.execute(
                "SELECT MIN(created_at) FROM jobs WHERE status IN (?,?,?)",
                _ACTIVE_STATUSES).fetchone()[0]
            section["oldest_active_age_seconds"] = _iso_age(oldest, now) if oldest else None
        return section
    except sqlite3.Error:
        return None


def _foreground_section(state: Path) -> dict | None:
    path = _find(state, "foreground")
    if path is None:
        return None
    try:
        with closing(_readonly(path)) as db:
            return {
                "by_status": _group_counts(
                    db, "SELECT status, COUNT(*) FROM requests GROUP BY status"),
                "cleanup_unknown": db.execute(
                    "SELECT COUNT(*) FROM requests WHERE status='running'"
                    " AND cleanup='unknown'").fetchone()[0],
            }
    except sqlite3.Error:
        return None


def _outbox_section(state: Path) -> dict | None:
    path = _find(state, "outbox")
    if path is None:
        return None
    try:
        with closing(_readonly(path)) as db:
            return {"by_status": _group_counts(
                db, "SELECT status, COUNT(*) FROM announcements GROUP BY status")}
    except sqlite3.Error:
        return None


def _memory_section(state: Path) -> dict | None:
    path = _find(state, "memory")
    if path is None:
        return None
    try:
        with closing(_readonly(path)) as db:
            return {
                "active": db.execute("SELECT COUNT(*) FROM memories WHERE deleted=0").fetchone()[0],
                "soft_forgotten": db.execute(
                    "SELECT COUNT(*) FROM memories WHERE deleted=1").fetchone()[0],
                "revisions": db.execute("SELECT COUNT(*) FROM memory_revisions").fetchone()[0],
            }
    except sqlite3.Error:
        return None


def _audit_section(state: Path) -> dict:
    audit: dict[str, Any] = {}
    for name, head_table in (("club", "club_guild_state"), ("style", "style")):
        path = _find(state, name)
        if path is None:
            continue
        try:
            with closing(_readonly(path)) as db:
                heads, head = db.execute(
                    f"SELECT COUNT(*), MAX(version) FROM {head_table}").fetchone()
                revisions = db.execute(
                    f"SELECT COUNT(*) FROM {name}_revisions").fetchone()[0]
            audit[name] = {"state_rows": heads, "max_version": head, "revisions": revisions}
        except sqlite3.Error:
            continue
    return audit


def _projects_section(state: Path) -> dict | None:
    db_path = _projects_db(state)
    if not os.path.lexists(db_path):
        return None
    try:
        with closing(_readonly(db_path)) as db:
            section = {
                "live": db.execute("SELECT COUNT(*) FROM projects WHERE deleted=0").fetchone()[0],
                "versions": db.execute("SELECT COUNT(*) FROM versions").fetchone()[0],
                "files": db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "stored_bytes": db.execute(
                    "SELECT COALESCE(SUM(size),0) FROM files").fetchone()[0],
            }
        blob_count = 0
        blobs = _projects_root(state) / "blobs"
        if blobs.is_dir():
            for sub in blobs.iterdir():
                if sub.is_dir():
                    blob_count += sum(1 for entry in sub.iterdir() if entry.is_file())
        section["blob_files"] = blob_count
        return section
    except sqlite3.Error:
        return None


def diagnose(state_dir: str | Path, *, now: datetime | None = None) -> dict:
    """Read-only operator report: revision, component health, safe aggregates."""
    state = Path(state_dir)
    ts = _now(now)
    health = component_health(state)
    return {
        "revision": os.environ.get("PETERBOT_REVISION", "unknown"),
        "generated_at": ts.isoformat(),
        "components": health,
        "jobs": _jobs_section(state, ts),
        "foreground": _foreground_section(state),
        "outbox": _outbox_section(state),
        "memory": _memory_section(state),
        "audit": _audit_section(state),
        "projects": _projects_section(state),
    }


def _plan_projects(state: Path, ts: datetime) -> dict:
    """Mirror `ProjectStore.retention_sweep()` eligibility, read-only."""
    from .project_store import ProjectSettings

    db_path = _projects_db(state)
    policy_days = ProjectSettings().retention_days
    if not os.path.lexists(db_path):
        return {"status": "offline", "policy_days": policy_days}
    cutoff = int(ts.timestamp()) - policy_days * 86400
    try:
        with closing(_readonly(db_path)) as db:
            versions = db.execute(
                "SELECT COUNT(*) FROM versions v WHERE v.created_at<?"
                " AND (v.version < (SELECT MAX(w.version) FROM versions w"
                "     WHERE w.project_id = v.project_id)"
                "  OR v.project_id IN (SELECT id FROM projects WHERE created_at<? AND updated_at<?))",
                (cutoff, cutoff, cutoff)).fetchone()[0]
            events = db.execute("SELECT COUNT(*) FROM events WHERE ts<?", (cutoff,)).fetchone()[0]
        return {"status": "measured", "policy_days": policy_days,
                "versions_aged": versions, "events_expired": events}
    except sqlite3.Error:
        return {"status": "degraded", "policy_days": policy_days}


def retention_plan(state_dir: str | Path, config: RetentionConfig,
                   *, now: datetime | None = None) -> dict:
    """Dry-run default: per-category eligible counts, no mutation, no socket."""
    state = Path(state_dir)
    ts = _now(now)
    health = component_health(state)
    categories: dict[str, dict] = {}
    cutoffs: dict[str, str] = {}
    for category in CATEGORIES:
        cutoff = _cutoff(ts, getattr(config, category.days_attr))
        cutoffs[category.name] = cutoff
        status = health[category.component]["status"]
        if status == "offline":
            categories[category.name] = {"status": "offline", "eligible": 0}
            continue
        if status == "degraded":
            categories[category.name] = {"status": "degraded", "eligible": None}
            continue
        path = _find(state, category.component)
        try:
            with closing(_readonly(path)) as db:
                eligible = db.execute(category.count_sql,
                                      category.cutoff_args(cutoff)).fetchone()[0]
            categories[category.name] = {"status": "measured", "eligible": eligible}
        except sqlite3.Error:
            categories[category.name] = {"status": "degraded", "eligible": None}
    return {
        "dry_run": True,
        "generated_at": ts.isoformat(),
        "cutoffs": cutoffs,
        "categories": categories,
        "include_projects": config.include_projects,
        "projects": _plan_projects(state, ts) if config.include_projects else {"status": "excluded"},
    }


def retention_apply(state_dir: str | Path, config: RetentionConfig,
                    *, now: datetime | None = None) -> dict:
    """Explicit deletion. Refuses any degraded store; preserves all protected states."""
    state = Path(state_dir)
    ts = _now(now)
    health = component_health(state)
    for category in CATEGORIES:
        if health[category.component]["status"] == "degraded":
            raise ValueError(
                f"Retention refuses to write a degraded {category.component} store")
    result: dict[str, Any] = {"dry_run": False, "deleted": {}, "skipped": {}, "projects": None}
    for category in CATEGORIES:
        status = health[category.component]["status"]
        if status == "offline":
            result["skipped"][category.name] = "offline"
            continue
        path = _find(state, category.component)
        cutoff = _cutoff(ts, getattr(config, category.days_attr))
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA busy_timeout=10000")
            args = category.cutoff_args(cutoff)
            try:
                eligible = db.execute(category.count_sql, args).fetchone()[0]
                if eligible:
                    db.execute(category.delete_sql, args)
                db.commit()
            except sqlite3.Error:
                db.rollback()
                result["skipped"][category.name] = "unreadable"
                continue
        result["deleted"][category.name] = eligible
    if config.include_projects:
        root = _projects_root(state)
        if os.path.lexists(_projects_db(state)):
            from .project_store import ProjectStore

            store = ProjectStore(root)
            try:
                result["projects"] = store.retention_sweep(now=int(ts.timestamp()))
            finally:
                store.close()
        else:
            result["skipped"]["projects"] = "offline"
    return result
