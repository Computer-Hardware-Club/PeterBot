"""Private bounded timing counters; no prompts, Discord IDs, or raw errors."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


STAGES = frozenset({"ingress", "queue", "routing", "model", "tool", "worker", "artifact", "delivery"})
OUTCOMES = frozenset({"ok", "failed", "timeout", "cancelled", "unknown", "denied"})
MAX_DURATION_MS = 3_600_000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricStore:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise ValueError("Metric database is newer than this gateway")
        if version == 0:
            with self.db:
                self.db.execute("""CREATE TABLE IF NOT EXISTS stage_metrics (
                    id INTEGER PRIMARY KEY, at TEXT NOT NULL, stage TEXT NOT NULL,
                    outcome TEXT NOT NULL, duration_ms INTEGER NOT NULL,
                    input_tokens INTEGER, output_tokens INTEGER)""")
                self.db.execute("CREATE INDEX IF NOT EXISTS stage_metrics_recent ON stage_metrics(at)")
                self.db.execute("PRAGMA user_version=1")

    def record(self, stage: str, outcome: str, duration_ms: int, *,
               input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        if stage not in STAGES or outcome not in OUTCOMES:
            raise ValueError("Unknown metric stage or outcome")
        if type(duration_ms) is not int or not 0 <= duration_ms <= MAX_DURATION_MS:
            raise ValueError("Invalid duration")
        for count in (input_tokens, output_tokens):
            if count is not None and (type(count) is not int or not 0 <= count <= 10_000_000):
                raise ValueError("Invalid token count")
        with self.db:
            self.db.execute("""INSERT INTO stage_metrics
                (at,stage,outcome,duration_ms,input_tokens,output_tokens)
                VALUES (?,?,?,?,?,?)""",
                (_utcnow(),stage,outcome,duration_ms,input_tokens,output_tokens))

    def summary(self, *, hours: int = 24) -> dict[str, dict]:
        if type(hours) is not int or not 1 <= hours <= 720:
            raise ValueError("Invalid metric window")
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.db.execute("""SELECT stage,outcome,duration_ms,input_tokens,output_tokens
            FROM stage_metrics WHERE at>=? ORDER BY stage,duration_ms""", (cutoff,))
        grouped: dict[str, list] = {}
        for row in rows:
            grouped.setdefault(row["stage"], []).append(row)
        result = {}
        for stage, items in grouped.items():
            durations = [row["duration_ms"] for row in items]
            count = len(items)
            result[stage] = {
                "count": count,
                "p50_ms": durations[(count - 1) // 2],
                "p95_ms": durations[min(count - 1, (95 * count + 99) // 100 - 1)],
                "outcomes": {outcome: sum(row["outcome"] == outcome for row in items)
                             for outcome in sorted(OUTCOMES) if any(row["outcome"] == outcome for row in items)},
                "input_tokens": sum(row["input_tokens"] or 0 for row in items),
                "output_tokens": sum(row["output_tokens"] or 0 for row in items),
            }
        return result

    def delete_before(self, cutoff: datetime) -> int:
        if cutoff.tzinfo is None:
            raise ValueError("Cutoff must have a timezone")
        with self.db:
            changed = self.db.execute("DELETE FROM stage_metrics WHERE at<?",
                                      (cutoff.astimezone(timezone.utc).isoformat(),))
        return changed.rowcount
