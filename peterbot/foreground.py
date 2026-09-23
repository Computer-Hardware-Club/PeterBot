"""One foreground cognitive chain, a durable FIFO queue, and a process lease.

PETER-04: every inference and agent entry point — name/mention/reply chat,
ordinary conversation, ``/ask``, ``/recap``, ``/task`` submissions, and task
continuations — funnels through this scheduler. At most one request holds the
slot; a handoff from a conversational turn inherits the turn's queue position
so the accepted objective is never released to another requester mid-chain.

Durability model (deliberate split):

* A *task* request is bound to a durable job row (``agent_jobs``); it survives
  restart and resumes honestly from the queue.
* A *chat/ask/recap* request persists only its identity-bound envelope (kind,
  guild, user, channel, source message). The prompt itself stays in process
  memory. On restart an acknowledged-but-unstarted envelope becomes an honest
  ``interrupted`` event and is dropped; an unacknowledged one is dropped
  silently. Either way: no model call, nothing private stored or leaked.

Slot safety:

* ``claim_next()`` refuses to start anything while any row is ``running`` or
  while an unresolved ``cleanup-unknown`` row exists, so a slot is never
  released while a prior worker could still be live. A task execution is only
  live once the runner confirms the worker is gone (or the job completed); a
  hard kill mid-execution leaves ``cleanup-unknown`` and the queue stays held
  until ``resolve_cleanup()`` records the operator reconciliation.
* Duplicate Discord events collapse on the UNIQUE (guild, source_message_id,
  kind) envelope key: a replay is never a second cognitive chain.
* ``acquire_lease()`` rejects a second live gateway against the same state
  directory; accidental double processes cannot run two foreground chains.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sqlite3
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

QUEUED = 'queued'
RUNNING = 'running'
DONE = 'done'
DROPPED = 'dropped'

WORKER_KINDS = frozenset({'task'})

# A conversational request waits at most this long before giving up, so a held
# cleanup-unknown never strands a live Discord handler forever.
DEFAULT_TOTAL_TIMEOUT = 900.0

# Set by the scheduler while one request's work runs; a handoff (chat -> task)
# inherits this request's queue position instead of cutting the line.
current_request_id: ContextVar[Optional[str]] = ContextVar('peter_foreground_request', default=None)


class QuotaExceeded(ValueError):
    """Explicitly rejected admission; the request is never silently dropped."""


class ForegroundCancelled(Exception):
    """The owner (or shutdown) cancelled the queued request before it started."""


class DuplicateEvent(ValueError):
    """A replayed Discord/interaction event; the original chain owns it."""


class AlreadyRunning(RuntimeError):
    """Another live gateway process holds the singleton lease."""


class ForegroundScheduler:
    """Durable FIFO admission with exactly one live slot.

    All SQLite claims are single ``BEGIN IMMEDIATE`` transactions, so two
    pumps on the same database (two threads, or two processes sharing the
    file) cannot both start the same envelope.
    """

    def __init__(self, path: str, *, instance: str | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 per_user_cap: int = 3, global_cap: int = 12,
                 ack_after: float = 1.5, poll: float = 0.05,
                 lease_seconds: float = 30.0,
                 total_timeout: float = DEFAULT_TOTAL_TIMEOUT) -> None:
        if per_user_cap < 1 or global_cap < 1 or per_user_cap > global_cap:
            raise ValueError('foreground caps must satisfy 1 <= per_user <= global')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('''CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL, channel_id INTEGER NOT NULL,
            source_message_id INTEGER NOT NULL, job_id TEXT, parent_id TEXT,
            seq INTEGER NOT NULL, status TEXT NOT NULL,
            acked INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
            instance TEXT, cleanup TEXT NOT NULL DEFAULT 'none',
            created_at REAL NOT NULL, started_at REAL, finished_at REAL)''')
        columns = {r[1] for r in self.db.execute('PRAGMA table_info(requests)')}
        if 'parent_id' not in columns:
            self.db.execute('ALTER TABLE requests ADD COLUMN parent_id TEXT')
        self.db.execute('''CREATE UNIQUE INDEX IF NOT EXISTS request_ingress
            ON requests (guild_id, source_message_id, kind)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
            event TEXT NOT NULL, at REAL NOT NULL)''')
        self.db.execute('''CREATE TABLE IF NOT EXISTS lease (
            id INTEGER PRIMARY KEY CHECK (id = 1), instance TEXT NOT NULL,
            pid INTEGER NOT NULL, host TEXT NOT NULL, heartbeat REAL NOT NULL)''')
        self.clock = clock
        self.instance = instance or f'{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}'
        self.per_user_cap = per_user_cap
        self.global_cap = global_cap
        self.ack_after = ack_after
        self.poll = poll
        self.lease_seconds = lease_seconds
        self.total_timeout = total_timeout
        # In-process execution state. Closures are chat turns owned by a
        # waiting handler; executors are durable kinds (task) run by the
        # gateway, which releases the slot only when worker cleanup is known.
        self._closures: dict[str, Callable[[], Any]] = {}
        self._futures: dict[str, asyncio.Future] = {}
        self._executors: dict[str, Callable[[dict], Optional[asyncio.Task]]] = {}
        self._running: dict[str, asyncio.Task] = {}
        self._closed = False

    # ------------------------------------------------------------------ state

    def get(self, request_id: str) -> dict | None:
        row = self.db.execute('SELECT * FROM requests WHERE id=?', (request_id,)).fetchone()
        return dict(row) if row else None

    def find(self, guild_id: int, source_message_id: int, kind: str) -> dict | None:
        row = self.db.execute('SELECT * FROM requests WHERE guild_id=? AND source_message_id=?'
                              ' AND kind=?', (guild_id, source_message_id, kind)).fetchone()
        return dict(row) if row else None

    def reopen(self, request_id: str) -> bool:
        """Return an abnormally-terminated envelope for the SAME job to the queue.

        Only ever used by startup reconciliation for a queued job whose bound
        envelope lost its claim path (crash/reconcile races). FIFO position is
        preserved; admission caps are not re-checked (the row was already live).
        """
        with self.db:
            cur = self.db.execute('UPDATE requests SET status=?, instance=NULL, started_at=NULL,'
                                  " finished_at=NULL, cleanup='none', cancel_requested=0"
                                  ' WHERE id=? AND status IN (?,?)',
                                  (QUEUED, request_id, DONE, DROPPED))
            if cur.rowcount:
                self._event(request_id, 'reopened', self.clock())
            return cur.rowcount == 1

    def position(self, request_id: str) -> int:
        """Deterministic queue position: live work plus envelopes ahead."""
        row = self.get(request_id)
        if row is None:
            return 0
        ahead = self.db.execute(
            'SELECT COUNT(*) FROM requests WHERE status=? AND (seq, id) < (?, ?)',
            (QUEUED, row['seq'], row['id'])).fetchone()[0]
        live = self.db.execute('SELECT COUNT(*) FROM requests WHERE status=?', (RUNNING,)).fetchone()[0]
        return ahead + live

    def counts(self) -> dict:
        rows = self.db.execute('SELECT status, COUNT(*) n FROM requests WHERE status IN (?,?)'
                               ' GROUP BY status', (QUEUED, RUNNING)).fetchall()
        out = {r['status']: r['n'] for r in rows}
        return {'queued': out.get(QUEUED, 0), 'running': out.get(RUNNING, 0),
                'cleanup_unknown': self.has_cleanup_unknown()}

    def has_cleanup_unknown(self) -> bool:
        return self.db.execute("SELECT 1 FROM requests WHERE status=? AND cleanup='unknown'"
                               ' LIMIT 1', (RUNNING,)).fetchone() is not None

    # --------------------------------------------------------------- admission

    def enqueue(self, *, kind: str, guild_id: int, user_id: int, channel_id: int,
                source_message_id: int, job_id: str | None = None,
                inherit_from: str | None = None) -> tuple[dict, bool]:
        """Reserve an envelope. Returns (row, created); created=False is a duplicate.

        Raises QuotaExceeded when bounded admission is full; the rejection is
        explicit and the caller must tell the requester.
        """
        if not kind:
            raise ValueError('foreground kind is required')
        if kind in WORKER_KINDS and job_id is None:
            # A worker envelope is the claim path for durable job execution:
            # without the binding, a chat-style closure could resolve it and
            # the actual worker would run with no foreground ownership.
            raise ValueError('worker-kind envelopes require a job_id binding')
        now = self.clock()
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            existing = self.db.execute(
                'SELECT * FROM requests WHERE guild_id=? AND source_message_id=? AND kind=?',
                (guild_id, source_message_id, kind)).fetchone()
            if existing is not None:
                # A replayed Discord event never becomes a second chain, even
                # after the original finished.
                return dict(existing), False
            live = self.db.execute(
                "SELECT COUNT(*) FROM requests WHERE status IN ('queued','running')").fetchone()[0]
            if live >= self.global_cap:
                raise QuotaExceeded('My queue is full right now. Please try again shortly.')
            own = self.db.execute(
                "SELECT COUNT(*) FROM requests WHERE user_id=? AND status IN ('queued','running')",
                (user_id,)).fetchone()[0]
            if own >= self.per_user_cap:
                raise QuotaExceeded("You already have a few requests waiting. Let me catch up first.")
            if inherit_from is not None:
                # A handoff keeps the parent's line position: nothing queued
                # after the accepted turn can overtake the work it started.
                parent = self.db.execute('SELECT seq, status FROM requests WHERE id=?',
                                         (inherit_from,)).fetchone()
                if parent is None:
                    raise ValueError('Unknown foreground request to inherit')
                seq = parent['seq']
            else:
                seq = self.db.execute('SELECT COALESCE(MAX(seq),0)+1 FROM requests').fetchone()[0]
            request_id = uuid.uuid4().hex
            self.db.execute(
                'INSERT INTO requests (id,kind,guild_id,user_id,channel_id,source_message_id,'
                ' job_id,parent_id,seq,status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (request_id, kind, guild_id, user_id, channel_id, source_message_id,
                 job_id, inherit_from, seq, QUEUED, now))
            self._event(request_id, 'queued', now)
        return self.get(request_id), True

    def bind_job(self, request_id: str, job_id: str) -> None:
        """Attach the durable job a conversational handoff created."""
        with self.db:
            self.db.execute('UPDATE requests SET job_id=? WHERE id=?', (job_id, request_id))

    def mark_acknowledged(self, request_id: str) -> None:
        """Record that the deterministic busy ack was attempted; never retried."""
        with self.db:
            changed = self.db.execute(
                "UPDATE requests SET acked=1 WHERE id=? AND status IN (?,?) AND acked=0",
                (request_id, QUEUED, RUNNING))
            if changed.rowcount:
                self._event(request_id, 'acknowledged', self.clock())

    def was_acknowledged(self, request_id: str) -> bool:
        row = self.get(request_id)
        return bool(row and row['acked'])

    # ------------------------------------------------------------------- slot

    def claim_next(self) -> dict | None:
        """Atomically start the next FIFO envelope; None when the slot is held.

        A cleanup-unknown row blocks the slot entirely: the prior execution may
        still be live, and starting another chain would break the one-mind
        invariant.

        Ownership continuity: a handoff child inherits the parent's queue
        position (``seq``), so the moment the routing turn resolves, the child
        is the front of the FIFO and no other requester can slip into the slot
        between the turn and the work it started.
        """
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.db.execute("SELECT 1 FROM requests WHERE status=? AND cleanup='unknown'"
                               ' LIMIT 1', (RUNNING,)).fetchone():
                return None
            if self.db.execute("SELECT 1 FROM requests WHERE status=? LIMIT 1", (RUNNING,)).fetchone():
                return None
            for candidate in self.db.execute(
                    'SELECT * FROM requests WHERE status=? ORDER BY seq, id', (QUEUED,)):
                if candidate['parent_id'] is None:
                    row = candidate
                    break
                parent = self.db.execute('SELECT status FROM requests WHERE id=?',
                                         (candidate['parent_id'],)).fetchone()
                # A child can inherit the position only after its parent has
                # resolved. While that parent is still queued, random ID
                # ordering must never start the child first; while running,
                # it owns the slot. A missing/terminal parent is safe to pass.
                if parent is None or parent['status'] not in (QUEUED, RUNNING):
                    row = candidate
                    break
            else:
                return None
            now = self.clock()
            cur = self.db.execute(
                'UPDATE requests SET status=?, instance=?, started_at=? WHERE id=? AND status=?',
                (RUNNING, self.instance, now, row['id'], QUEUED))
            if not cur.rowcount:
                return None
            self._event(row['id'], 'started', now)
            return self.get(row['id'])

    def request_cancel(self, request_id: str) -> str | None:
        """Owner-facing cancellation. Queued -> dropped now; running -> flagged.

        A running conversational closure is interrupted in-process (its HTTP
        request dies with it, nothing is brokered). A running *worker* row is
        only flagged: the gateway drives the sandbox cancel and confirms
        cleanup before releasing the slot.
        """
        with self.db:
            row = self.db.execute('SELECT status, kind FROM requests WHERE id=?',
                                  (request_id,)).fetchone()
            if row is None:
                return None
            if row['status'] == QUEUED:
                self.db.execute('UPDATE requests SET status=?, finished_at=? WHERE id=? AND status=?',
                                (DROPPED, self.clock(), request_id, QUEUED))
                self._event(request_id, 'cancelled', self.clock())
                return DROPPED
            self.db.execute('UPDATE requests SET cancel_requested=1 WHERE id=?', (request_id,))
            if row['kind'] not in WORKER_KINDS:
                live = self._running.get(request_id)
                if live is not None and not live.done():
                    live.cancel()
            return row['status']

    def cancel_for_job(self, job_id: str) -> str | None:
        """Cancel the foreground envelope bound to a job (queued drop or running flag)."""
        row = self.db.execute("SELECT id FROM requests WHERE job_id=? AND status IN ('queued','running')",
                              (job_id,)).fetchone()
        return self.request_cancel(row['id']) if row else None

    def finish(self, request_id: str, *, event: str = 'finished',
               cleanup: str = 'none') -> bool:
        """Resolve a running row as complete and free the slot."""
        with self.db:
            cur = self.db.execute('UPDATE requests SET status=?, finished_at=?, cleanup=?'
                                  ' WHERE id=? AND status=?',
                                  (DONE, self.clock(), cleanup, request_id, RUNNING))
            if cur.rowcount:
                self._event(request_id, event, self.clock())
            return cur.rowcount == 1

    def fail(self, request_id: str, reason: str) -> bool:
        """Resolve a running row as failed and free the slot (chat turns only)."""
        with self.db:
            cur = self.db.execute("UPDATE requests SET status=?, finished_at=?, cleanup='none'"
                                  ' WHERE id=? AND status=?',
                                  (DROPPED, self.clock(), request_id, RUNNING))
            if cur.rowcount:
                self._event(request_id, f'failed:{reason}'[:200], self.clock())
            return cur.rowcount == 1

    def release_worker(self, request_id: str, *, confirmed: bool, event: str) -> None:
        """A worker execution ended. Free the slot only with cleanup proof.

        ``confirmed=False`` (runner unreachable / cancel unacknowledged) records
        the explicit cleanup-unknown state and holds the queue; the operator
        resolves it with ``resolve_cleanup()`` after verifying the worker.
        """
        with self.db:
            if confirmed:
                cur = self.db.execute('UPDATE requests SET status=?, finished_at=?, cleanup=?'
                                      ' WHERE id=? AND status=?',
                                      (DONE, self.clock(), 'none', request_id, RUNNING))
            else:
                cur = self.db.execute("UPDATE requests SET cleanup='unknown' WHERE id=? AND status=?",
                                      (request_id, RUNNING))
            if cur.rowcount:
                self._event(request_id, event if confirmed else 'cleanup-unknown', self.clock())
        if not confirmed:
            log.error('Foreground worker cleanup unconfirmed; queue held until reconciliation: %s',
                      request_id)

    def resolve_cleanup(self, request_id: str) -> bool:
        """Operator recorded that the prior worker is confirmed gone."""
        with self.db:
            cur = self.db.execute("UPDATE requests SET status=?, cleanup='resolved', finished_at=?"
                                  ' WHERE id=? AND status=? AND cleanup=?',
                                  (DROPPED, self.clock(), request_id, RUNNING, 'unknown'))
            if cur.rowcount:
                self._event(request_id, 'cleanup-resolved', self.clock())
            return cur.rowcount == 1

    # --------------------------------------------------------------- recovery

    def recover(self) -> dict:
        """Deterministic restart reconciliation. Call once at process startup.

        Durability split by kind:
        - A running *worker* row cannot be released blindly: the previous
          process died without cleanup proof, so the slot stays held as
          cleanup-unknown until an operator resolves it.
        - Queued *task* envelopes are durable work: they stay queued and resume
          in FIFO order (the job row behind them is re-claimable).
        - Running chat/ask/recap envelopes brokered an in-process turn that is
          gone; they are dropped. An acknowledged one gets an honest
          interrupted event; an unacknowledged one drops silently — no reply
          to a request whose interaction may already have expired.
        - Queued chat envelopes are likewise gone.
        """
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            now = self.clock()
            workers = [r['id'] for r in self.db.execute(
                "SELECT id FROM requests WHERE status=? AND kind IN ({})"
                .format(','.join('?' * len(WORKER_KINDS))), (RUNNING, *WORKER_KINDS))]
            self.db.execute("UPDATE requests SET cleanup='unknown' WHERE status=? AND kind IN ({})"
                            .format(','.join('?' * len(WORKER_KINDS))), (RUNNING, *WORKER_KINDS))
            for request_id in workers:
                self._event(request_id, 'cleanup-unknown', now)
            dead_turns = [r['id'] for r in self.db.execute(
                "SELECT id FROM requests WHERE status=? AND kind NOT IN ({})"
                .format(','.join('?' * len(WORKER_KINDS))), (RUNNING, *WORKER_KINDS))]
            self.db.execute("UPDATE requests SET status=?, cleanup='none', finished_at=?"
                            " WHERE status=? AND kind NOT IN ({})"
                            .format(','.join('?' * len(WORKER_KINDS))),
                            (DROPPED, now, RUNNING, *WORKER_KINDS))
            for request_id in dead_turns:
                self._event(request_id, 'interrupted', now)
            stale_turns = [r['id'] for r in self.db.execute(
                "SELECT id FROM requests WHERE status=? AND kind NOT IN ({}) AND acked=1"
                .format(','.join('?' * len(WORKER_KINDS))), (QUEUED, *WORKER_KINDS))]
            silent = [r['id'] for r in self.db.execute(
                "SELECT id FROM requests WHERE status=? AND kind NOT IN ({}) AND acked=0"
                .format(','.join('?' * len(WORKER_KINDS))), (QUEUED, *WORKER_KINDS))]
            self.db.execute("UPDATE requests SET status=?, finished_at=? WHERE status=?"
                            " AND kind NOT IN ({})".format(','.join('?' * len(WORKER_KINDS))),
                            (DROPPED, now, QUEUED, *WORKER_KINDS))
            for request_id in stale_turns:
                self._event(request_id, 'interrupted-after-restart', now)
            for request_id in silent:
                self._event(request_id, 'dropped-after-restart', now)
        if workers:
            log.error('Foreground restart: slot held as cleanup-unknown for %s', workers)
        return {'cleanup_unknown': workers, 'interrupted': dead_turns + stale_turns,
                'dropped': silent}

    # ------------------------------------------------------------------ lease

    def acquire_lease(self) -> None:
        """Singleton guard: one live gateway per state directory.

        The heartbeat is monotonic, so it is comparable only within one boot.
        A heartbeat ahead of the current clock can only come from an earlier
        boot — a reboot reset the clock — and that holder is gone by
        definition, so it is reclaimed rather than wedging startup with a
        phantom AlreadyRunning. Within one boot monotonic never goes
        backwards, so two genuinely live gateways still cannot both hold it.
        """
        now = self.clock()
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT * FROM lease WHERE id=1').fetchone()
            if row is not None and row['instance'] != self.instance:
                age = now - row['heartbeat']
                if 0 <= age < self.lease_seconds:
                    raise AlreadyRunning(
                        f"another Peter gateway ({row['instance']} on {row['host']}) is still active")
                if age < 0:
                    log.warning('Foreground lease heartbeat %ss ahead of this clock; '
                                'treating %s as reboot-stale', age, row['instance'])
            self.db.execute('INSERT OR REPLACE INTO lease (id,instance,pid,host,heartbeat)'
                            ' VALUES (1,?,?,?,?)',
                            (self.instance, os.getpid(), socket.gethostname(), now))

    def renew_lease(self) -> None:
        with self.db:
            self.db.execute('UPDATE lease SET heartbeat=? WHERE id=1 AND instance=?',
                            (self.clock(), self.instance))

    def release_lease(self) -> None:
        with self.db:
            self.db.execute('DELETE FROM lease WHERE id=1 AND instance=?', (self.instance,))

    def lease_holder(self) -> str | None:
        row = self.db.execute('SELECT instance FROM lease WHERE id=1').fetchone()
        return row['instance'] if row else None

    # ------------------------------------------------------------------- pump

    def register(self, kind: str, executor: Callable[[dict], Optional[asyncio.Task]]) -> None:
        """Durable kinds (task) run through the gateway executor, which owns
        releasing the slot once worker cleanup is confirmed."""
        self._executors[kind] = executor

    async def pump_once(self) -> dict | None:
        """Claim at most one envelope and dispatch it; returns what was claimed."""
        if self._closed:
            return None
        claimed = self.claim_next()
        if claimed is None:
            return None
        self._dispatch(claimed)
        return claimed

    def _dispatch(self, row: dict) -> None:
        request_id = row['id']
        token = None
        try:
            if row['kind'] in self._executors:
                task = self._executors[row['kind']](row)
            elif request_id in self._closures:
                task = asyncio.create_task(self._run_closure(row))
            else:
                # No live waiter in this process (e.g. a chat envelope whose
                # handler already timed out): resolve honestly, free the slot.
                self.fail(request_id, 'no live handler')
                return
            if task is not None:
                self._running[request_id] = task
                task.add_done_callback(lambda _t, key=request_id: self._running.pop(key, None))
        except Exception:  # noqa: BLE001 - a dispatch bug must not strand the slot
            log.exception('Foreground dispatch failed: %s', request_id)
            self.fail(request_id, 'dispatch error')
        finally:
            if token is not None:
                current_request_id.reset(token)

    async def _run_closure(self, row: dict) -> None:
        """Run a chat envelope's work under the slot, then release it.

        A cancelled chat turn closes its own HTTP request; unlike a sandbox
        worker it leaves nothing brokered that could act afterwards, so the
        slot is released with an honest failure rather than held.
        """
        request_id = row['id']
        work = self._closures.pop(request_id, None)
        future = self._futures.get(request_id)
        token = current_request_id.set(request_id)
        try:
            if self._closed or work is None:
                self._settle(request_id, future, exc=ForegroundCancelled('request lost before execution'))
                self.fail(request_id, 'lost before execution')
                return
            fresh = self.get(request_id)
            if fresh is None or fresh['status'] != RUNNING:
                self._settle(request_id, future, exc=ForegroundCancelled('request no longer live'))
                return
            if fresh['cancel_requested']:
                self._settle(request_id, future, exc=ForegroundCancelled('cancellation requested'))
                self.fail(request_id, 'cancelled before execution')
                return
            try:
                value = await work()
            except asyncio.CancelledError:
                self.fail(request_id, 'interrupted')
                self._settle(request_id, future, exc=ForegroundCancelled('interrupted by shutdown'))
                raise
            except Exception as exc:  # surfaced to the waiter verbatim
                self.fail(request_id, type(exc).__name__)
                self._settle(request_id, future, exc=exc)
                return
            self.finish(request_id)
            self._settle(request_id, future, value=value)
        finally:
            current_request_id.reset(token)

    def _settle(self, request_id: str, future: Optional[asyncio.Future], *,
                value: Any = None, exc: Optional[BaseException] = None) -> None:
        if future is not None and not future.done():
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(value)

    async def run_one(self, *, kind: str, guild_id: int, user_id: int, channel_id: int,
                      source_message_id: int, work: Callable[[], Any],
                      acknowledge: Optional[Callable[[int], Any]] = None,
                      inherit_from: str | None = None,
                      total_timeout: float | None = None) -> tuple[dict, Any]:
        """Admit one conversational envelope and run ``work`` under the slot.

        Returns (row, value). While waiting, ``acknowledge(position)`` is
        awaited exactly once after ``ack_after`` seconds: it is a transport
        message, never a model call, and carries only the queue position —
        never another requester's prompt, identity, or channel.
        """
        row, created = self.enqueue(kind=kind, guild_id=guild_id, user_id=user_id,
                                    channel_id=channel_id, source_message_id=source_message_id,
                                    inherit_from=inherit_from)
        if not created:
            raise DuplicateEvent('that message was already handled')
        request_id = row['id']
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._futures[request_id] = future
        self._closures[request_id] = work
        budget = self.total_timeout if total_timeout is None else total_timeout
        started_waiting = self.clock()
        acked = False
        try:
            while True:
                # Pump until our envelope is claimed; a peer pump (queue loop
                # or another handler) may claim and run it first — the future
                # observes either outcome.
                await self.pump_once()
                try:
                    value = await asyncio.wait_for(asyncio.shield(future), timeout=self.poll)
                    return self.get(request_id), value
                except asyncio.TimeoutError:
                    pass
                state = self.get(request_id)
                # Completion can race the short polling timeout: the row may
                # already be DONE even though wait_for timed out on its shield.
                # Read the original future before treating that state as a
                # lost result, so a successful turn is not falsely cancelled.
                if future.done():
                    return state, future.result()
                if state is None or state['status'] == DROPPED:
                    raise ForegroundCancelled('your queued request was cancelled')
                if state['status'] == DONE:
                    raise ForegroundCancelled('request ended without a result')
                if state['cancel_requested']:
                    self.request_cancel(request_id)
                    raise ForegroundCancelled('cancellation requested')
                if state['status'] == QUEUED and self.clock() - started_waiting > budget:
                    # The budget bounds queue waiting; an already-running turn
                    # carries its own handler deadline.
                    self.request_cancel(request_id)
                    raise asyncio.TimeoutError('the foreground queue did not reach your request in time')
                if acknowledge is not None and not acked and state['status'] == QUEUED \
                        and self.clock() - started_waiting >= self.ack_after:
                    acked = True
                    # Attempted-once is recorded BEFORE delivery: a queue ack is
                    # best-effort transport (like a typing indicator). If it fails
                    # — an expired interaction is the common cause — we neither
                    # retry it (no duplicate ack) nor let it abort a turn that is
                    # already running, which would orphan an in-flight model call
                    # and force the requester to ask again.
                    self.mark_acknowledged(request_id)
                    try:
                        await acknowledge(self.position(request_id))
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.warning('Foreground queue ack undeliverable: %s', request_id)
                await asyncio.sleep(self.poll)
        finally:
            self._futures.pop(request_id, None)
            self._closures.pop(request_id, None)
            state = self.get(request_id)
            if state is not None and state['kind'] not in WORKER_KINDS \
                    and state['status'] in (QUEUED, RUNNING):
                # The waiter is gone. A chat/ask/recap turn brokers nothing
                # outside this process: cancel the queued promise (or the live
                # closure — its HTTP request dies with it) rather than keep a
                # response nobody is waiting for, or hold the slot for it.
                self.request_cancel(request_id)
                live = self._running.get(request_id)
                if live is not None:
                    # Let the closure finish unwinding (its work releases the
                    # per-user guard) before the handler exits.
                    try:
                        await asyncio.wait((live,), timeout=5.0)
                    except asyncio.CancelledError:
                        pass

    async def close(self) -> None:
        """Stop dispatching. Cancelled chat turns resolve; worker rows keep
        their gateway-assigned cleanup state (release_worker ran or will run).
        Anything still ``running`` without a cleanup verdict is held unknown.
        """
        self._closed = True
        tasks = [t for t in self._running.values() if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with self.db:
            rows = self.db.execute("SELECT id, kind FROM requests WHERE status=? AND cleanup=?",
                                   (RUNNING, 'none')).fetchall()
            for row in rows:
                if row['kind'] in WORKER_KINDS:
                    self.db.execute("UPDATE requests SET cleanup='unknown' WHERE id=?", (row['id'],))
                    self._event(row['id'], 'cleanup-unknown', self.clock())
                    log.error('Foreground shutdown: worker cleanup unconfirmed: %s', row['id'])
                else:
                    # A chat closure without a verdict: nothing was brokered.
                    self.db.execute("UPDATE requests SET status=?, cleanup='none', finished_at=?"
                                    ' WHERE id=?', (DROPPED, self.clock(), row['id']))
                    self._event(row['id'], 'interrupted', self.clock())
        self.release_lease()

    def close_sync(self) -> None:
        try:
            self.db.close()
        except sqlite3.Error:
            pass

    def _event(self, request_id: str, event: str, at: float) -> None:
        self.db.execute('INSERT INTO events (request_id,event,at) VALUES (?,?,?)',
                        (request_id, event, at))

    def events(self, request_id: str) -> list[str]:
        return [r['event'] for r in self.db.execute(
            'SELECT event FROM events WHERE request_id=? ORDER BY id', (request_id,))]
