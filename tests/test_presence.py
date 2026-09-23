"""The member-facing presence layer: one message that shows work and becomes the answer.

These tests care about what the member sees, so the fakes record sends and edits rather
than merely returning values.
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from peterbot.agent_jobs import JobStore
from peterbot.agent_policy import Principal
from peterbot.hermes_gateway import HermesGateway
from peterbot.hermes_settings import HermesSettings
from peterbot.presence import Presence, elapsed_label, watch_task


def http_error(status=403):
    return discord.HTTPException(SimpleNamespace(status=status, reason='Forbidden'), 'nope')


class FakeMessage:
    def __init__(self, message_id, content=''):
        self.id = message_id
        self.content = content
        self.edits = []
        self.fail_edits = False

    async def edit(self, *, content=None, **kwargs):
        if self.fail_edits:
            raise http_error()
        self.content = content
        self.edits.append(content)
        return self


class FakeChannel:
    def __init__(self, *, fail_sends=False):
        self.sent = []
        self.messages = {}
        self.next_id = 1000
        self.fail_sends = fail_sends

    async def send(self, content=None, **kwargs):
        if self.fail_sends:
            raise http_error()
        message = FakeMessage(self.next_id, content or '')
        self.next_id += 1
        self.sent.append((content, kwargs))
        self.messages[message.id] = message
        return message

    async def fetch_message(self, message_id):
        if message_id not in self.messages:
            raise http_error(404)
        return self.messages[message_id]


def run(coro):
    return asyncio.run(coro)


def test_quick_turn_never_posts_a_placeholder():
    """Banter must not flicker a status message that is immediately replaced."""

    async def scenario():
        channel = FakeChannel()
        presence = Presence(channel, status_after=3600)
        async with presence:
            await asyncio.sleep(0)
            delivered = await presence.finish('Only on Tuesdays.')
        return channel, delivered

    channel, delivered = run(scenario())
    assert channel.sent == []
    assert delivered is False


def test_slow_turn_shows_a_status_line_and_then_becomes_the_answer():
    async def scenario():
        channel = FakeChannel()
        presence = Presence(channel, status_after=0)
        async with presence:
            await asyncio.sleep(0.01)
            assert presence.message_id is not None
            delivered = await presence.finish('KEC 1005, Fridays at 6.')
        return channel, delivered

    channel, delivered = run(scenario())
    assert delivered is True
    assert len(channel.sent) == 1  # the status message
    message = channel.messages[channel.next_id - 1]
    # The same message now holds the answer, with no second message above it.
    assert message.edits == ['KEC 1005, Fridays at 6.']
    assert message.content == 'KEC 1005, Fridays at 6.'


def test_long_answer_edits_the_first_chunk_and_sends_the_rest():
    async def scenario():
        channel = FakeChannel()
        presence = Presence(channel, status_after=0, max_chars=100)
        async with presence:
            await asyncio.sleep(0.01)
            delivered = await presence.finish('x' * 250)
        return channel, presence, delivered

    channel, presence, delivered = run(scenario())
    assert delivered is True
    assert len(channel.sent) == 3  # status message + two follow-on chunks
    assert len(presence.message.edits[0]) == 100
    assert channel.sent[1][0] == 'x' * 100


def test_status_updates_are_throttled_and_can_be_forced():
    async def scenario():
        channel = FakeChannel()
        clock = [100.0]
        presence = Presence(channel, status_after=0, now=lambda: clock[0])
        async with presence:
            await asyncio.sleep(0.01)
            clock[0] += 1.0
            await presence.show('still working')          # too soon after the post
            clock[0] += 10.0
            await presence.show('still working harder')   # allowed
            await presence.show('still working hardest')  # too soon again
            await presence.show('forced', force=True)
        return channel, presence

    channel, presence = run(scenario())
    assert presence.message.edits == ['still working harder', 'forced']
    assert len(channel.sent) == 1


def test_a_deleted_or_uneditable_message_never_breaks_the_turn():
    async def scenario():
        channel = FakeChannel()
        presence = Presence(channel, status_after=0)
        async with presence:
            await asyncio.sleep(0.01)
            presence.message.fail_edits = True
            await presence.show('update')          # does not raise
            delivered = await presence.finish('answer')  # falls back to the caller
        return delivered

    assert run(scenario()) is False


def test_a_failed_status_post_never_breaks_the_turn():
    async def scenario():
        channel = FakeChannel(fail_sends=True)
        presence = Presence(channel, status_after=0)
        async with presence:
            await asyncio.sleep(0.01)
            delivered = await presence.finish('answer')
        return presence, delivered

    presence, delivered = run(scenario())
    assert presence.message_id is None
    assert delivered is False


def test_progress_reports_elapsed_time_only():
    async def scenario():
        channel = FakeChannel()
        clock = [100.0]
        presence = Presence(channel, status_after=0, max_chars=200, now=lambda: clock[0])
        async with presence:
            await asyncio.sleep(0.01)
            clock[0] += 10.0  # past the edit throttle, as a real 20s tick would be
            task = asyncio.create_task(watch_task(presence, 'job', interval=0.01, status='running'))
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return channel, presence

    channel, presence = run(scenario())
    assert len(channel.sent) == 1, 'the status line must not spawn extra messages'
    edits = presence.message.edits
    assert edits, 'the status line should have been updated'
    assert all(edit.startswith('still working — 0s in (running)') for edit in edits)
    # No invented percentage, and no claim about a stage we cannot see.
    assert not any('%' in edit for edit in edits)


def test_elapsed_label_is_readable():
    assert elapsed_label(9) == '9s'
    assert elapsed_label(59.9) == '59s'
    assert elapsed_label(60) == '1m 00s'
    assert elapsed_label(605) == '10m 05s'


def test_adopt_wraps_an_already_posted_message():
    async def scenario():
        channel = FakeChannel()
        posted = await channel.send('on it — this needs real work')
        presence = Presence.adopt(channel, posted)
        assert presence.message_id == posted.id
        await presence.show('on it — this needs real work')  # unchanged text: no edit
        await presence.show('on it — still working', force=True)
        return posted

    posted = run(scenario())
    assert posted.edits == ['on it — still working']


@asynccontextmanager
async def gateway_with_channel(tmp_path, channel):
    guild = SimpleNamespace(id=10, fetch_member=AsyncMock(return_value=None))
    channel.guild = guild
    bot = SimpleNamespace(get_guild=lambda guild_id: guild, fetch_channel=AsyncMock(return_value=channel))
    config = SimpleNamespace(agent=SimpleNamespace(search_base_url=''),
                             inference=SimpleNamespace(model='trusted-qwen', base_url='http://trusted-model:8000/v1'),
                             llama_cpp_api_key='k', max_discord_message_chars=1800, peter_name='Peter')
    settings = HermesSettings(allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
                              owner_user_ids=frozenset({1}), runner_url='http://runner:8080',
                              tool_service_url='http://gateway:8770', runner_token='r' * 40,
                              state_dir=str(tmp_path), officer_only=True)
    gateway = HermesGateway(bot, config, settings)
    # Authorization itself is covered elsewhere; these tests are about what the member sees.
    async def principal(guild_id, user_id, channel_id, *args, **kwargs):
        return Principal(guild_id, user_id, channel_id, (100,))
    gateway.principal = principal
    try:
        yield gateway
    finally:
        gateway.jobs.close()


def channel_job(gateway, **extra):
    job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                              prompt='write a rust program', delivery_mode='channel',
                              status_message_id=extra.get('status_message_id'))
    gateway.jobs.update(job['id'], status='completed', answer=extra.get('answer', 'Here is the program.'),
                        artifacts=extra.get('artifacts', []))
    return gateway.jobs.get(job['id'])


def test_delivery_turns_the_status_message_into_the_answer(tmp_path):
    async def scenario():
        channel = FakeChannel()
        status = await channel.send('on it — this needs real work')
        async with gateway_with_channel(tmp_path, channel) as gateway:
            job = channel_job(gateway, status_message_id=status.id)
            await gateway.deliver(job)
            return status, channel, gateway.jobs.get(job['id'])

    status, channel, job = run(scenario())
    # One message in the channel: the one the member was already watching.
    assert channel.sent == [('on it — this needs real work', {})]
    assert status.content == 'Here is the program.'
    assert job['delivered'] == 1
    assert job['delivery_cursor'] == 1


def test_status_edit_without_saved_receipt_is_held_for_reconciliation(tmp_path):
    async def scenario():
        channel = FakeChannel()
        status = await channel.send('on it — this needs real work')
        async with gateway_with_channel(tmp_path, channel) as gateway:
            job = channel_job(gateway, status_message_id=status.id)
            def fail_receipt(*args, **kwargs):
                raise RuntimeError('disk write failed')
            gateway.jobs.advance_delivery = fail_receipt
            await gateway.deliver(job)
            return status, channel, gateway.jobs.get(job['id']), gateway.jobs.undelivered()

    status, channel, job, pending = run(scenario())
    assert status.content == 'Here is the program.'
    assert job['delivery_status'] == 'unknown'
    assert job['delivery_cursor'] == 0
    assert pending == []
    assert len(channel.sent) == 1  # no duplicate answer is posted


def test_delivery_falls_back_to_a_new_message_when_the_status_message_is_gone(tmp_path):
    async def scenario():
        channel = FakeChannel()
        async with gateway_with_channel(tmp_path, channel) as gateway:
            job = channel_job(gateway, status_message_id=999999)  # never existed
            await gateway.deliver(job)
            return channel, gateway.jobs.get(job['id'])

    channel, job = run(scenario())
    assert [sent[0] for sent in channel.sent] == ['Here is the program.']
    assert job['delivered'] == 1


def test_delivery_sends_artifacts_after_the_status_message_becomes_the_answer(tmp_path):
    import base64

    async def scenario():
        channel = FakeChannel()
        status = await channel.send('on it — this needs real work')
        payload = base64.b64encode(b'sum_amount=60\n').decode()
        async with gateway_with_channel(tmp_path, channel) as gateway:
            job = channel_job(gateway, status_message_id=status.id, answer='Done.',
                              artifacts=[{'name': 'result.txt', 'data_base64': payload}])
            await gateway.deliver(job)
            return status, channel

    status, channel = run(scenario())
    assert status.content == 'Done.'
    assert len(channel.sent) == 2  # status message + one file, no duplicate text
    assert channel.sent[1][1].get('file') is not None


def test_progress_reporter_edits_the_message_the_member_is_watching(tmp_path):
    async def scenario():
        channel = FakeChannel()
        status = await channel.send('on it')
        async with gateway_with_channel(tmp_path, channel) as gateway:
            job = {'id': 'x', 'channel_id': 20, 'status_message_id': status.id}
            task = asyncio.create_task(gateway.report_progress(job, interval=0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return status, channel

    status, channel = run(scenario())
    assert len(channel.sent) == 1                 # never posts a second message
    assert status.edits and status.edits[0].startswith('still working — ')


def test_progress_reporter_gives_up_quietly_on_a_missing_message(tmp_path):
    async def scenario():
        channel = FakeChannel()
        async with gateway_with_channel(tmp_path, channel) as gateway:
            await gateway.report_progress({'id': 'x', 'channel_id': 20, 'status_message_id': 999999},
                                          interval=0.01)
        return channel

    assert run(scenario()).sent == []
