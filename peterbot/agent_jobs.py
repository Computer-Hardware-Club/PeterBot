"""Durable task queue. Only the trusted gateway opens this database.

Execution state machine:

    preparing -> queued -> running -> completed | failed | timeout | cancelled | interrupted

`preparing` means the submission may not have reached the user yet; the queue
never claims or delivers such a job. Terminal states are frozen: every status
change goes through `transition()`, which conditions on the legal predecessor
set inside a single UPDATE, so a late completion cannot overwrite a terminal
cancellation and two claimers cannot both start the same queued job.

Delivery state is separate from execution state (`delivery_status`):

    pending -> delivering -> delivered | withheld | exhausted
                   \\-> unknown (crash-ambiguous; operator-reconciled)

`undelivered()` selects only an explicit set of terminal statuses, never a
negative list, so an empty in-flight `preparing` job can never be "delivered"
ahead of its answer. Per-chunk cursors and Discord message receipts persist
which parts were acknowledged. A restart cannot tell whether the in-flight
chunk landed, so it becomes `unknown` and is never auto-replayed;
`reconcile_unknown_delivery()` records the operator decision. Known Discord
client rejects and rate limits retry from the durable cursor; server errors
remain unknown. Exactly-once
delivery across an unknown outcome is NOT promised.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

log = logging.getLogger(__name__)

PREPARING = 'preparing'
QUEUED = 'queued'
RUNNING = 'running'
ACTIVE_STATUSES = frozenset({PREPARING, QUEUED, RUNNING})
# Only these terminal states are eligible for delivery. Never widen by exclusion.
TERMINAL_STATUSES = frozenset({'completed', 'failed', 'timeout', 'cancelled', 'interrupted'})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES

# Legal predecessors per target status; `transition()` enforces this table.
LEGAL_PREDECESSORS: dict[str, frozenset[str]] = {
    PREPARING: frozenset(),
    QUEUED: frozenset({PREPARING}),
    RUNNING: frozenset({QUEUED}),
    'completed': frozenset({RUNNING}),
    'failed': ACTIVE_STATUSES,
    'timeout': frozenset({RUNNING}),
    'cancelled': frozenset({QUEUED, RUNNING}),
    'interrupted': frozenset({RUNNING}),
}

# After this many failed Discord attempts a result stops polling the gateway and
# is marked `exhausted` for operator review instead of retrying forever.
DELIVERY_ATTEMPT_LIMIT = 12


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
        for name, definition in {'input_files':"TEXT NOT NULL DEFAULT '[]'", 'delivery_cursor':'INTEGER NOT NULL DEFAULT 0',
                                 'delivery_mode':"TEXT NOT NULL DEFAULT 'private'", 'context':"TEXT NOT NULL DEFAULT '[]'",
                                 'delivery_status':"TEXT NOT NULL DEFAULT 'pending'", 'delivery_attempts':'INTEGER NOT NULL DEFAULT 0',
                                 'delivery_receipts':"TEXT NOT NULL DEFAULT '[]'"}.items():
            if name not in columns:
                self.db.execute(f'ALTER TABLE jobs ADD COLUMN {name} {definition}')
        # Ingress reservations map one Discord source message to one job so a
        # replayed MESSAGE_CREATE cannot spawn a second thread or task.
        self.db.execute('''CREATE TABLE IF NOT EXISTS ingress (
            guild_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
            job_id TEXT, claimed_at TEXT NOT NULL,
            PRIMARY KEY (guild_id, source_message_id))''')
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',answer='Task submission was interrupted.',updated_at=? WHERE status='preparing'",(now(),))
            self.db.execute("UPDATE jobs SET status='interrupted', answer='The gateway restarted during this task. Use /continue_task to resume from the saved objective.', updated_at=? WHERE status='running'", (now(),))
            # A crash mid-delivery leaves an ambiguous send: the in-flight
            # chunk may or may not have reached Discord. Freeze it as `unknown`
            # for reconciliation instead of resending blindly; the cursor and
            # answer stay intact and acknowledged parts are never replayed.
            stranded = [r['id'] for r in self.db.execute("SELECT id FROM jobs WHERE delivery_status='delivering'")]
            self.db.execute("UPDATE jobs SET delivery_status='unknown', updated_at=? WHERE delivery_status='delivering'", (now(),))
            if stranded:
                log.error('Delivery outcome unknown after restart; reconcile before resending: %s', stranded)
            # Unbound reservations came from a submission that died before its
            # job row committed. `create()` binds the reservation inside the
            # insert transaction, so a job can never outlive its reservation.
            self.db.execute("DELETE FROM ingress WHERE job_id IS NULL")

    def create(self, *, guild_id: int, user_id: int, channel_id: int,
               source_message_id: int, prompt: str, parent_id: str | None = None, input_files: list | None = None, ready: bool = True, delivery_mode: str = 'private', context: list | None = None,
               ingress: tuple[int, int] | None = None) -> dict:
        if not prompt.strip() or len(prompt) > 16000:
            raise ValueError('Please use a task description between 1 and 16,000 characters.')
        if delivery_mode not in {'private','channel'}:
            raise ValueError('Invalid delivery mode')
        job_id = uuid.uuid4().hex
        with self.db:
            # Acquire the SQLite writer lock before counting. A context manager
            # alone starts a deferred transaction only at the first write.
            self.db.execute('BEGIN IMMEDIATE')
            self.check_capacity(user_id)
            self.db.execute('INSERT INTO jobs (id,guild_id,user_id,channel_id,source_message_id,prompt,parent_id,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)',
                            (job_id,guild_id,user_id,channel_id,source_message_id,prompt,parent_id,QUEUED if ready else PREPARING,now(),now()))
            self.db.execute('UPDATE jobs SET input_files=?,delivery_mode=?,context=? WHERE id=?',(json.dumps(input_files or []),delivery_mode,json.dumps(context or []),job_id))
            if ingress is not None:
                # Job insertion and ingress binding share one transaction: no
                # crash window can leave a committed job whose reservation is
                # still unbound (and therefore deletable at restart).
                cur = self.db.execute("UPDATE ingress SET job_id=? WHERE guild_id=? AND source_message_id=? AND job_id IS NULL",
                                      (job_id, ingress[0], ingress[1]))
                if not cur.rowcount:
                    self.db.execute('INSERT INTO ingress (guild_id,source_message_id,job_id,claimed_at) VALUES (?,?,?,?)',
                                    (ingress[0], ingress[1], job_id, now()))
        return self.get(job_id)

    def check_capacity(self, user_id: int) -> None:
        count = self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','running')").fetchone()[0]
        own = self.db.execute("SELECT COUNT(*) FROM jobs WHERE user_id=? AND status IN ('preparing','queued','running')", (user_id,)).fetchone()[0]
        if count >= 20 or own >= 2:
            raise ValueError('The task queue is full for now. Finish or cancel an existing task first.')

    def latest_for_thread(self, guild_id: int, user_id: int, channel_id: int) -> dict | None:
        row=self.db.execute("SELECT * FROM jobs WHERE guild_id=? AND user_id=? AND channel_id=? AND delivery_mode='private' ORDER BY created_at DESC LIMIT 1",(guild_id,user_id,channel_id)).fetchone()
        return dict(row) if row else None

    def conversation_context(self, guild_id: int, user_id: int, channel_id: int) -> list[dict]:
        rows=self.db.execute("SELECT prompt,answer FROM jobs WHERE guild_id=? AND user_id=? AND channel_id=? AND delivery_mode='channel' AND status='completed' AND julianday(created_at)>=julianday('now','-1 hour') ORDER BY created_at DESC LIMIT 3",(guild_id,user_id,channel_id)).fetchall()
        result=[]
        for row in reversed(rows):
            result.extend([{'role':'user','content':row['prompt'][:2000]}, {'role':'assistant','content':row['answer'][:2000]}])
        return result

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

    def claim(self, job_id: str) -> bool:
        """Atomically admit one queued job; False if anyone else claimed it first."""
        with self.db:
            cur = self.db.execute("UPDATE jobs SET status=?,updated_at=? WHERE id=? AND status=?", (RUNNING, now(), job_id, QUEUED))
        return cur.rowcount == 1

    def transition(self, job_id: str, to: str, *, answer: str | None = None,
                   artifacts: list | None = None) -> bool:
        """Conditionally move to `to`; False when the job is not in a legal predecessor state."""
        if to not in LEGAL_PREDECESSORS:
            raise ValueError('Invalid task status')
        froms = sorted(LEGAL_PREDECESSORS[to])
        if not froms:
            return False
        fields = {'status': to, 'updated_at': now()}
        if answer is not None:
            fields['answer'] = answer[:24000]
        if artifacts is not None:
            fields['artifacts'] = json.dumps(artifacts)
        sql = ('UPDATE jobs SET ' + ','.join(f'{key}=?' for key in fields)
               + ' WHERE id=? AND status IN (' + ','.join('?' * len(froms)) + ')')
        with self.db:
            cur = self.db.execute(sql, (*fields.values(), job_id, *froms))
        return cur.rowcount == 1

    def abandon_submission(self, job_id: str, answer: str) -> None:
        """Fail a never-admitted submission; acknowledged without any delivery."""
        with self.db:
            self.db.execute("UPDATE jobs SET status='failed',answer=?,delivered=1,delivery_status='withheld',updated_at=?"
                            + " WHERE id=? AND status IN ('preparing','queued')", (answer[:24000], now(), job_id))

    def undelivered(self) -> list[dict]:
        placeholders = ','.join('?' * len(TERMINAL_STATUSES))
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM jobs WHERE delivered=0 AND delivery_status='pending' AND status IN ("
            + placeholders + ") ORDER BY created_at LIMIT 20", sorted(TERMINAL_STATUSES))]

    def begin_delivery(self, job_id: str) -> bool:
        """Claim the delivery side of a job; False if it is claimed or resolved."""
        with self.db:
            cur = self.db.execute("UPDATE jobs SET delivery_status='delivering',updated_at=?"
                                  " WHERE id=? AND delivered=0 AND delivery_status='pending'", (now(), job_id))
        return cur.rowcount == 1

    def advance_delivery(self, job_id: str, *, cursor: int, receipts: list) -> bool:
        """Persist a part receipt; False if the durable cursor already covers it."""
        with self.db:
            changed = self.db.execute(
                "UPDATE jobs SET delivery_cursor=?,delivery_receipts=?,updated_at=?"
                " WHERE id=? AND delivery_status='delivering' AND delivery_cursor<?",
                (cursor, json.dumps(receipts), now(), job_id, cursor),
            )
        return changed.rowcount == 1

    def complete_delivery(self, job_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE jobs SET delivery_status='delivered',delivered=1,updated_at=? WHERE id=?", (now(), job_id))

    def withhold_delivery(self, job_id: str) -> None:
        """Resolve delivery as deliberately withheld; the stored answer is kept."""
        with self.db:
            self.db.execute("UPDATE jobs SET delivery_status='withheld',delivered=1,updated_at=? WHERE id=?", (now(), job_id))

    def note_delivery_failure(self, job_id: str) -> str:
        """Count one failed Discord attempt; returns the resulting delivery_status."""
        with self.db:
            self.db.execute("UPDATE jobs SET delivery_attempts=delivery_attempts+1,updated_at=?,"
                            + " delivery_status=CASE WHEN delivery_attempts+1>=? THEN 'exhausted' ELSE 'pending' END"
                            + " WHERE id=? AND delivered=0 AND delivery_status='delivering'",
                            (now(), DELIVERY_ATTEMPT_LIMIT, job_id))
        row = self.db.execute('SELECT delivery_status FROM jobs WHERE id=?', (job_id,)).fetchone()
        return row['delivery_status'] if row else 'missing'

    def mark_delivery_unknown(self, job_id: str) -> None:
        """Freeze a crash-ambiguous or unclassifiable send for reconciliation."""
        with self.db:
            self.db.execute("UPDATE jobs SET delivery_status='unknown',updated_at=? WHERE id=? AND delivery_status='delivering'", (now(), job_id))

    def reconcile_unknown_delivery(self, job_id: str, *, retry: bool,
                                   confirmed_message_id: int | None = None) -> bool:
        """Resume after an operator checks Discord for the uncertain chunk.

        ``retry=True`` means the chunk was not found. Confirmation requires
        the actual Discord message ID; it advances only that one chunk and
        leaves any remaining chunks pending for normal delivery.
        """
        if retry and confirmed_message_id is not None:
            raise ValueError('A retry cannot also confirm a receipt')
        if not retry and (type(confirmed_message_id) is not int or confirmed_message_id <= 0):
            raise ValueError('A confirmed Discord message ID is required')
        with self.db:
            if retry:
                cur = self.db.execute("UPDATE jobs SET delivery_status='pending',updated_at=? WHERE id=? AND delivery_status='unknown'", (now(), job_id))
            else:
                row = self.db.execute("SELECT delivery_cursor,delivery_receipts FROM jobs WHERE id=? AND delivery_status='unknown'", (job_id,)).fetchone()
                if row is None:
                    return False
                receipts = json.loads(row['delivery_receipts'])
                receipts.append(str(confirmed_message_id))
                cur = self.db.execute(
                    "UPDATE jobs SET delivery_cursor=?,delivery_receipts=?,delivery_status='pending',updated_at=?"
                    " WHERE id=? AND delivery_status='unknown'",
                    (row['delivery_cursor'] + 1, json.dumps(receipts), now(), job_id),
                )
        return cur.rowcount == 1

    def claim_ingress(self, guild_id: int, source_message_id: int) -> str | None:
        """Reserve the ingress slot for one Discord message.

        None: this caller owns the reservation. Otherwise the slot is taken:
        the bound job ID, or '' while a concurrent submission is still creating
        its job row.
        """
        with self.db:
            cur = self.db.execute('INSERT OR IGNORE INTO ingress (guild_id,source_message_id,job_id,claimed_at) VALUES (?,?,NULL,?)',
                                  (guild_id, source_message_id, now()))
            if cur.rowcount:
                return None
            row = self.db.execute('SELECT job_id FROM ingress WHERE guild_id=? AND source_message_id=?',
                                  (guild_id, source_message_id)).fetchone()
        return row['job_id'] or ''

    def release_ingress(self, guild_id: int, source_message_id: int) -> None:
        """Drop only an unbound reservation so a corrected retry can proceed."""
        with self.db:
            self.db.execute("DELETE FROM ingress WHERE guild_id=? AND source_message_id=? AND job_id IS NULL",
                            (guild_id, source_message_id))

    def find_by_ingress(self, guild_id: int, source_message_id: int) -> dict | None:
        row = self.db.execute('SELECT jobs.* FROM ingress JOIN jobs ON jobs.id=ingress.job_id'
                              ' WHERE ingress.guild_id=? AND ingress.source_message_id=?',
                              (guild_id, source_message_id)).fetchone()
        return dict(row) if row else None

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
