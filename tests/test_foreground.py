"""PETER-04: one foreground cognitive chain, durable FIFO, lease, recovery.

Every test uses real concurrency (two requesters, two schedulers on one
database, threads racing claims) with bounded wall-clock windows so the suite
stays deterministic.
"""

import asyncio
import threading
import uuid
from unittest.mock import patch

import pytest

from peterbot.foreground import (
    AlreadyRunning,
    DuplicateEvent,
    ForegroundCancelled,
    ForegroundScheduler,
    QuotaExceeded,
    current_request_id,
)


def make(tmp_path, **kwargs):
    kwargs.setdefault('poll', 0.01)
    kwargs.setdefault('ack_after', 0.03)
    return ForegroundScheduler(str(tmp_path / 'foreground.sqlite3'), **kwargs)


def chat(sched, user, source, work, **kwargs):
    return sched.run_one(kind='chat', guild_id=10, user_id=user, channel_id=20,
                         source_message_id=source, work=work, **kwargs)


async def value_work(value):
    return value


# ------------------------------------------------------------------- slot

def test_two_users_share_exactly_one_active_chain(tmp_path):
    """Two users asking at once never overlap: peak concurrency is 1 and the
    queue drains in FIFO order."""
    sched = make(tmp_path)
    gate = asyncio.Event()
    active = 0
    peak = 0
    order = []

    def work(user):
        async def run():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if user == 1:
                await gate.wait()
            order.append(user)
            active -= 1
            return f"answer-{user}"
        return run

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 101, work(1), total_timeout=5))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(chat(sched, 2, 102, work(2), total_timeout=5))
        await asyncio.sleep(0.05)
        assert active == 1  # second user is queued, not running
        gate.set()
        row1, value1 = await asyncio.wait_for(first, timeout=5)
        row2, value2 = await asyncio.wait_for(second, timeout=5)
        return (row1, value1), (row2, value2)

    (row1, value1), (row2, value2) = asyncio.run(scenario())
    assert peak == 1
    assert order == [1, 2]
    assert (value1, value2) == ('answer-1', 'answer-2')
    assert row1['status'] == row2['status'] == 'done'


def test_queued_request_gets_one_deterministic_ack_and_no_work(tmp_path):
    """The queue ack is transport-only: exactly one, with the FIFO position,
    and the queued request's work never starts while it waits."""
    sched = make(tmp_path)
    gate = asyncio.Event()
    acks = []
    started = []

    async def hold():
        await gate.wait()
        return 'first'

    def second_work():
        async def run():
            started.append('second')
            return 'second'
        return run

    async def acknowledge(position):
        acks.append(position)

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 201, hold, total_timeout=5))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(chat(sched, 2, 202, second_work(),
                                          acknowledge=acknowledge, total_timeout=5))
        await asyncio.sleep(0.1)
        assert acks == [1]           # exactly one ack, position includes live work
        assert started == []          # queued work performed nothing yet
        assert sched.was_acknowledged(sched.find(10, 202, 'chat')['id'])
        gate.set()
        await asyncio.wait_for(first, timeout=5)
        return await asyncio.wait_for(second, timeout=5)

    row, value = asyncio.run(scenario())
    assert value == 'second'
    assert started == ['second']
    assert row['acked'] == 1
    assert len(acks) == 1             # never retried


def test_ack_delivery_failure_does_not_abort_or_duplicate_the_request(tmp_path):
    """A queue ack is best-effort transport. If the Discord send raises (e.g.
    the interaction expired) the turn already dispatched under the slot must
    still complete exactly once and return its answer; the ack is never
    retried, and the requester is never forced to ask again."""
    sched = make(tmp_path, ack_after=0.01, poll=0.01)
    ack_attempts = []
    ran = []

    async def slow_answer():
        ran.append(True)
        await asyncio.sleep(0.08)
        return 'the answer'

    async def broken_ack(position):
        ack_attempts.append(position)
        raise RuntimeError('interaction expired')

    async def scenario():
        return await asyncio.wait_for(
            sched.run_one(kind='chat', guild_id=10, user_id=1, channel_id=20,
                          source_message_id=241, work=slow_answer,
                          acknowledge=broken_ack, total_timeout=5), timeout=5)

    row, value = asyncio.run(scenario())
    assert value == 'the answer'
    assert ran == [True]                      # ran exactly once, never aborted
    assert len(ack_attempts) == 1             # attempted once, never retried
    assert row['status'] == 'done'
    assert sched.was_acknowledged(row['id'])  # attempt recorded honestly


def test_running_chat_cancellation_interrupts_the_turn_and_frees_slot(tmp_path):
    """The owner cancelling a running chat turn interrupts the model call in
    this process and releases the slot for the next requester."""
    sched = make(tmp_path)
    gate = asyncio.Event()
    unwound = []

    def stalled():
        async def run():
            try:
                await gate.wait()
                return 'never'
            finally:
                unwound.append(True)
        return run

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 211, stalled(), total_timeout=5))
        await asyncio.sleep(0.05)
        request_id = sched.find(10, 211, 'chat')['id']
        assert sched.request_cancel(request_id) == 'running'
        with pytest.raises(ForegroundCancelled):
            await asyncio.wait_for(first, timeout=5)
        assert unwound == [True]
        return await asyncio.wait_for(
            chat(sched, 2, 212, lambda: value_work('next'), total_timeout=5), timeout=5)

    row, value = asyncio.run(scenario())
    assert value == 'next'
    assert sched.get(sched.find(10, 211, 'chat')['id'])['status'] == 'dropped'


def test_queued_cancellation_drops_without_ever_running(tmp_path):
    sched = make(tmp_path)
    gate = asyncio.Event()

    async def hold():
        await gate.wait()
        return 'first'

    def queued_work():
        async def run():
            raise AssertionError('cancelled queue entry must never run')
        return run

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 221, hold, total_timeout=5))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(chat(sched, 2, 222, queued_work(), total_timeout=5))
        await asyncio.sleep(0.05)
        queued_id = sched.find(10, 222, 'chat')['id']
        assert sched.request_cancel(queued_id) == 'dropped'
        with pytest.raises(ForegroundCancelled):
            await asyncio.wait_for(second, timeout=5)
        gate.set()
        await asyncio.wait_for(first, timeout=5)

    asyncio.run(scenario())
    assert 'cancelled' in sched.events(sched.find(10, 222, 'chat')['id'])


def test_waiter_timeout_never_leaves_a_stale_queue_promise(tmp_path):
    """A handler that gives up waiting cancels its own queued envelope; the
    queue stays honest for everyone else."""
    sched = make(tmp_path, total_timeout=0.08)
    gate = asyncio.Event()

    async def hold():
        await gate.wait()
        return 'first'

    def never_work():
        async def run():
            raise AssertionError('timed-out queue entry must never run')
        return run

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 231, hold, total_timeout=5))
        await asyncio.sleep(0.05)
        with pytest.raises(asyncio.TimeoutError):
            await chat(sched, 2, 232, never_work, total_timeout=0.08)
        gate.set()
        await asyncio.wait_for(first, timeout=5)

    asyncio.run(scenario())
    assert sched.get(sched.find(10, 232, 'chat')['id'])['status'] == 'dropped'


# ------------------------------------------------------------- duplicates

def test_duplicate_discord_event_never_becomes_a_second_chain(tmp_path):
    sched = make(tmp_path)
    gate = asyncio.Event()

    async def hold():
        await gate.wait()
        return 'only'

    async def scenario():
        first = asyncio.create_task(chat(sched, 1, 301, hold, total_timeout=5))
        await asyncio.sleep(0.05)
        with pytest.raises(DuplicateEvent):
            await chat(sched, 1, 301, hold, total_timeout=5)
        gate.set()
        return await asyncio.wait_for(first, timeout=5)

    row, value = asyncio.run(scenario())
    assert value == 'only'
    # And after the original finished, the replayed event is still suppressed.
    again, created = sched.enqueue(kind='chat', guild_id=10, user_id=1,
                                   channel_id=20, source_message_id=301)
    assert not created and again['id'] == row['id']


# ------------------------------------------------------------------ FIFO

def test_handoff_keeps_the_accepted_objective_at_the_front(tmp_path):
    """A routing turn that starts a task hands its queue position to the job:
    nobody queued in between can slip into the slot after the turn ends."""
    sched = make(tmp_path)
    # Force the child ID to sort before its parent within the inherited seq.
    # Random IDs made this bug appear only on some suite runs.
    with patch('peterbot.foreground.uuid.uuid4', side_effect=[
        uuid.UUID(int=2), uuid.UUID(int=3), uuid.UUID(int=1),
    ]):
        parent, _ = sched.enqueue(kind='ask', guild_id=10, user_id=1,
                                  channel_id=20, source_message_id=401)
        interloper, _ = sched.enqueue(kind='ask', guild_id=10, user_id=2,
                                      channel_id=20, source_message_id=402)
        child, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                                 source_message_id=403, job_id='job-1',
                                 inherit_from=parent['id'])
    assert sched.claim_next()['id'] == parent['id']
    assert sched.claim_next() is None      # slot held by the routing turn
    sched.finish(parent['id'])
    assert sched.claim_next()['id'] == child['id']   # handoff before interloper
    sched.finish(child['id'])
    assert sched.claim_next()['id'] == interloper['id']


def test_handoff_context_var_is_scoped_to_the_running_turn(tmp_path):
    """`current_request_id` is set only while that turn's work runs, so a
    handoff binds to the right parent and later requests cannot forge it."""
    sched = make(tmp_path)
    seen = {}

    def work():
        async def run():
            seen['inside'] = current_request_id.get()
            row, _ = sched.enqueue(kind='task', guild_id=10, user_id=1,
                                   channel_id=20, source_message_id=412,
                                   job_id='job-ctx',
                                   inherit_from=current_request_id.get())
            seen['child_parent'] = row['parent_id']
            return 'routed'
        return run

    async def scenario():
        assert current_request_id.get() is None
        row, value = await asyncio.wait_for(
            chat(sched, 1, 411, work(), total_timeout=5), timeout=5)
        assert current_request_id.get() is None
        return row, value

    row, value = asyncio.run(scenario())
    assert value == 'routed'
    assert seen['inside'] == row['id']
    assert seen['child_parent'] == row['id']


def test_orphaned_handoff_stands_on_its_own_fifo_position(tmp_path):
    """If the routing turn died before resolving, its handoff child is still
    claimable — durable work is never stranded by a dead parent."""
    sched = make(tmp_path)
    parent, _ = sched.enqueue(kind='ask', guild_id=10, user_id=1,
                              channel_id=20, source_message_id=421)
    child, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                             source_message_id=422, job_id='job-2',
                             inherit_from=parent['id'])
    sched.request_cancel(parent['id'])     # parent dropped while queued
    assert sched.claim_next()['id'] == child['id']


# ---------------------------------------------------------------- quotas

def test_per_user_and_global_admission_are_bounded(tmp_path):
    sched = make(tmp_path, per_user_cap=2, global_cap=3)
    sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                  source_message_id=501, job_id='j1')
    sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                  source_message_id=502, job_id='j2')
    with pytest.raises(QuotaExceeded, match="You already have"):
        sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                      source_message_id=503, job_id='j3')
    sched.enqueue(kind='task', guild_id=10, user_id=2, channel_id=20,
                  source_message_id=504, job_id='j4')
    with pytest.raises(QuotaExceeded, match="queue is full"):
        sched.enqueue(kind='task', guild_id=10, user_id=3, channel_id=20,
                      source_message_id=505, job_id='j5')


# ------------------------------------------------- duplicate admission

def test_two_gateway_processes_cannot_both_claim_one_request(tmp_path):
    """Two schedulers (separate connections, same state directory) racing a
    claim: exactly one wins, because every claim is one IMMEDIATE
    transaction."""
    path = str(tmp_path / 'foreground.sqlite3')
    seed = ForegroundScheduler(path)
    seed.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                 source_message_id=601, job_id='job-race')
    seed.close_sync()
    results = []
    lock = threading.Lock()

    def claim():
        # Each thread owns its connection: sqlite forbids sharing a handle.
        sched = ForegroundScheduler(path)
        try:
            got = sched.claim_next()
        finally:
            sched.close_sync()
        with lock:
            results.append(got)

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive()
    claimed = [row for row in results if row is not None]
    assert len(claimed) == 1
    assert claimed[0]['job_id'] == 'job-race'


def test_reboot_stale_lease_is_reclaimed_not_wedged(tmp_path):
    """monotonic resets on reboot. A heartbeat from a previous boot lies in
    this boot's future; holding the lease forever on it would wedge startup
    with a phantom AlreadyRunning. A negative age is proof of reboot, so the
    lease is reclaimed — and within one boot (non-negative age) two live
    gateways are still refused."""
    now = [1000.0]
    path = str(tmp_path / 'foreground.sqlite3')
    before_boot = ForegroundScheduler(path, clock=lambda: now[0], lease_seconds=30)
    before_boot.acquire_lease()
    holder = before_boot.lease_holder()
    now[0] = 5.0                        # reboot: monotonic restarts lower
    fresh = ForegroundScheduler(path, clock=lambda: now[0], lease_seconds=30)
    fresh.acquire_lease()               # reboot-stale lease is reclaimed
    assert fresh.lease_holder() != holder
    # And the reclaimed lease is a real lease again: a second live process in
    # this boot is still refused.
    rival = ForegroundScheduler(path, clock=lambda: now[0], lease_seconds=30)
    with pytest.raises(AlreadyRunning):
        rival.acquire_lease()


def test_second_live_gateway_cannot_take_the_singleton_lease(tmp_path):
    """A duplicate-process start is refused while the lease heartbeat is
    fresh; a dead process's expired lease is recoverable."""
    now = [1000.0]
    path = str(tmp_path / 'foreground.sqlite3')
    first = ForegroundScheduler(path, clock=lambda: now[0], lease_seconds=30)
    second = ForegroundScheduler(path, clock=lambda: now[0], lease_seconds=30)
    first.acquire_lease()
    with pytest.raises(AlreadyRunning):
        second.acquire_lease()
    first.renew_lease()
    now[0] += 20                      # renewed lease is still fresh
    with pytest.raises(AlreadyRunning):
        second.acquire_lease()
    now[0] += 31                      # heartbeat expired: stale lease is stealable
    second.acquire_lease()
    assert second.lease_holder() == second.instance


# --------------------------------------------------------------- restart

def test_restart_holds_the_slot_until_worker_cleanup_is_resolved(tmp_path):
    """A gateway crash mid-worker never silently releases the slot: the row
    comes back as cleanup-unknown and blocks the queue until reconciled."""
    path = str(tmp_path / 'foreground.sqlite3')
    old = ForegroundScheduler(path)
    worker, _ = old.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                            source_message_id=701, job_id='job-crash')
    old.claim_next()                                   # running when it crashed
    silent, _ = old.enqueue(kind='chat', guild_id=10, user_id=2,
                            channel_id=20, source_message_id=702)
    promised, _ = old.enqueue(kind='chat', guild_id=10, user_id=3,
                              channel_id=20, source_message_id=703)
    old.mark_acknowledged(promised['id'])
    old.close_sync()                                   # simulate crash

    fresh = ForegroundScheduler(path)
    summary = fresh.recover()
    assert summary['cleanup_unknown'] == [worker['id']]
    assert set(summary['interrupted']) == {promised['id']}
    assert set(summary['dropped']) == {silent['id']}
    # The acknowledged promise gets an honest interrupted event; the silent
    # one is dropped without any reply path — no private data leaks on restart.
    assert 'interrupted-after-restart' in fresh.events(promised['id'])
    assert 'dropped-after-restart' in fresh.events(silent['id'])
    assert fresh.claim_next() is None                  # queue held on cleanup-unknown
    assert fresh.has_cleanup_unknown()
    assert fresh.resolve_cleanup(worker['id'])
    assert not fresh.has_cleanup_unknown()
    # A new request can finally start; the crashed worker's envelope closed.
    later, _ = fresh.enqueue(kind='chat', guild_id=10, user_id=4,
                             channel_id=20, source_message_id=704)
    assert fresh.claim_next()['id'] == later['id']


def test_unconfirmed_worker_cleanup_holds_the_queue_by_design(tmp_path):
    """No cleanup proof from the runner => no new chain starts, even though
    the executor task itself ended."""
    sched = make(tmp_path)
    first, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                             source_message_id=711, job_id='job-unknown')
    sched.claim_next()
    second, _ = sched.enqueue(kind='task', guild_id=10, user_id=2, channel_id=20,
                              source_message_id=712, job_id='job-wait')
    sched.release_worker(first['id'], confirmed=False, event='task-cleanup-unknown')
    assert sched.claim_next() is None
    assert sched.counts() == {'queued': 1, 'running': 1, 'cleanup_unknown': True}
    assert sched.cancel_for_job('job-unknown') == 'running'  # flagged, not released
    # Operator reconciliation is the only sanctioned way out:
    assert sched.resolve_cleanup(first['id'])
    assert sched.claim_next()['id'] == second['id']


# ---------------------------------------------------------------- pump

def test_registered_worker_executor_runs_and_releases_with_proof(tmp_path):
    sched = make(tmp_path)
    seen = []

    def executor(envelope):
        seen.append(envelope['job_id'])

        async def run():
            sched.release_worker(envelope['id'], confirmed=True,
                                 event='task-cleanup-confirmed')
        return asyncio.create_task(run())

    sched.register('task', executor)
    row, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                           source_message_id=721, job_id='job-run')

    async def scenario():
        claimed = await sched.pump_once()
        assert claimed['id'] == row['id']
        for _ in range(100):
            await asyncio.sleep(0.01)
            if sched.get(row['id'])['status'] == 'done':
                break
        return sched.get(row['id'])

    done = asyncio.run(scenario())
    assert seen == ['job-run']
    assert done['status'] == 'done'
    assert 'task-cleanup-confirmed' in sched.events(row['id'])


def test_claim_without_a_handler_resolves_honestly(tmp_path):
    """A durable envelope whose executor vanished (no registration) fails
    closed: the slot is freed and the failure is recorded, never replayed
    into a second execution."""
    sched = make(tmp_path)
    row, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                           source_message_id=731, job_id='job-lost')

    async def scenario():
        claimed = await sched.pump_once()
        assert claimed['id'] == row['id']
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    done = sched.get(row['id'])
    assert done['status'] == 'dropped'
    assert any(event.startswith('failed:') for event in sched.events(row['id']))
    # The unique ingress key keeps the dead source message suppressed.
    again, created = sched.enqueue(kind='task', guild_id=10, user_id=1,
                                   channel_id=20, source_message_id=731,
                                   job_id='job-lost-2')
    assert not created and again['id'] == row['id']


def test_close_holds_worker_rows_without_verdict_as_unknown(tmp_path):
    """Shutdown must not pretend a live worker was cleaned up: rows without a
    cleanup verdict restart as cleanup-unknown, chat rows resolve honestly,
    and the lease is released for the next process."""
    sched = make(tmp_path)
    worker, _ = sched.enqueue(kind='task', guild_id=10, user_id=1, channel_id=20,
                              source_message_id=741, job_id='job-live')
    turn, _ = sched.enqueue(kind='chat', guild_id=10, user_id=2, channel_id=20,
                            source_message_id=742)
    sched.acquire_lease()
    assert sched.claim_next()['id'] == worker['id']   # worker running, no verdict

    async def scenario():
        # Model shutdown catching a second running row: dispatch moved the
        # chat envelope into running and its closure has no verdict yet.
        sched.db.execute("UPDATE requests SET status='running' WHERE id=?", (turn['id'],))
        sched.db.commit()
        await sched.close()

    asyncio.run(scenario())
    assert sched.get(worker['id'])['cleanup'] == 'unknown'
    assert sched.get(worker['id'])['status'] == 'running'   # held, not released
    assert sched.get(turn['id'])['status'] == 'dropped'
    assert 'interrupted' in sched.events(turn['id'])
    assert sched.lease_holder() is None                    # lease released on close
    fresh = ForegroundScheduler(str(tmp_path / 'foreground.sqlite3'))
    assert fresh.recover()['cleanup_unknown'] == [worker['id']]
    # Restart reconciliation does not touch a queued task envelope's job —
    # durable work resumes in FIFO order after cleanup resolves.
    durable, _ = fresh.enqueue(kind='task', guild_id=10, user_id=3, channel_id=20,
                               source_message_id=743, job_id='job-durable')
    assert fresh.get(durable['id'])['status'] == 'queued'
