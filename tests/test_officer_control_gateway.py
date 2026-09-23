"""Trusted Discord-source behavior for officer facts, voice and announcements."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from peterbot.agent_policy import PolicyDenied, Principal
from peterbot.hermes_gateway import HermesGateway
from peterbot.hermes_settings import HermesSettings


class Guild:
    id = 10
    member_count = 5

    def __init__(self):
        self.default_role = SimpleNamespace(id=0)
        self.officer = True
        self.bot_member = SimpleNamespace(id=99, bot=True, roles=[])
        self.members = {
            1: SimpleNamespace(id=1, bot=False, roles=[SimpleNamespace(id=100)],
                               display_name='Officer', name='officer', global_name=None),
            2: SimpleNamespace(id=2, bot=False, roles=[], display_name='Member',
                               name='member', global_name=None),
            1001: SimpleNamespace(id=1001, bot=False, roles=[], display_name='Alex',
                                  name='alex', global_name=None),
            1002: SimpleNamespace(id=1002, bot=False, roles=[], display_name='Sam',
                                  name='sam', global_name=None),
            99: self.bot_member,
        }

    async def fetch_member(self, user_id):
        member = self.members[user_id]
        if user_id == 1 and not self.officer:
            return SimpleNamespace(**{**vars(member), 'roles': []})
        return member

    async def fetch_members(self, *, limit):
        for member in list(self.members.values())[:limit]:
            yield member


class Channel:
    def __init__(self, guild, channel_id, *, private):
        self.guild, self.id, self.private = guild, channel_id, private

    def permissions_for(self, subject):
        return SimpleNamespace(view_channel=(not self.private if subject is self.guild.default_role else True),
                               send_messages=True)


class HTTP:
    def __init__(self):
        self.calls = []

    async def request(self, route, **kwargs):
        self.calls.append((route, kwargs))
        return {'id': '44', 'channel_id': '30'}


def make(tmp_path):
    guild = Guild()
    control = Channel(guild, 20, private=True)
    general = Channel(guild, 21, private=False)
    testing = Channel(guild, 22, private=True)
    destination = Channel(guild, 30, private=False)
    channels = {20: control, 21: general, 22: testing, 30: destination}
    bot = SimpleNamespace(user=SimpleNamespace(id=99), http=HTTP(),
        get_guild=lambda guild_id: guild if guild_id == 10 else None,
        fetch_channel=AsyncMock(side_effect=lambda channel_id: channels[channel_id]))
    config = SimpleNamespace(agent=SimpleNamespace(search_base_url=''),
                             peter_system_prompt='You are Peter.',
                             inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    settings = HermesSettings(allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
        owner_user_ids=frozenset({1}), runner_url='http://runner:8780',
        tool_service_url='http://gateway:8770', runner_token='r' * 40,
        state_dir=str(tmp_path), control_channel_ids=frozenset({20, 22}),
        announcement_destination_ids=frozenset({30}))
    return HermesGateway(bot, config, settings), guild, channels, bot


def message(guild, channel, text, *, user_id=1, source=500):
    sent = []
    async def reply(content, **kwargs):
        sent.append(content)
        return SimpleNamespace(id=900)
    return SimpleNamespace(guild=guild, channel=channel, id=source, content=text,
                           author=SimpleNamespace(id=user_id), attachments=[],
                           reply=reply, sent=sent)


def test_officer_fact_style_roster_and_undo_apply_immediately(tmp_path):
    gateway, guild, channels, _bot = make(tmp_path)

    async def scenario():
        fact = message(guild, channels[20], 'set public club fact meeting_room to KEC 1005')
        assert await gateway.handle_control_message(fact, fact.content)
        assert 'v1' in fact.sent[0]
        context, _ = gateway.club.chat_context(10, 'meeting room')
        assert 'KEC 1005' in context
        style = message(guild, channels[20], 'be a little more reserved', source=501)
        assert await gateway.handle_control_message(style, style.content)
        assert gateway.style.current(10)['version'] == 1
        from unittest.mock import patch
        with patch('peterbot.conversation.reply_or_use_tools', new=AsyncMock(return_value='KEC 1005')) as model:
            assert await gateway.conversational_reply(
                Principal(10, 1, 21, (100,)), 'Where is our meeting room?', [], audience='public') == 'KEC 1005'
            assert 'KEC 1005' in model.await_args.kwargs['club_context']
            assert 'reserved' in model.await_args.kwargs['style_instruction']
        assert 'KEC 1005' in gateway.club_persona(10, 'meeting room')
        roster = message(guild, channels[20], 'Alex is president for Fall 2026; Sam is vice president', source=502)
        assert await gateway.handle_control_message(roster, roster.content)
        assert {(row['office'], row['holder_user_id']) for row in gateway.club.public_officers(10)} == {
            ('president', 1001), ('vice_president', 1002)}
        undo = message(guild, channels[20], 'undo last roster', source=503)
        assert await gateway.handle_control_message(undo, undo.content)
        assert gateway.club.public_officers(10) == ()
        await gateway.close()

    asyncio.run(scenario())


def test_public_or_nonofficer_source_cannot_mutate_or_fall_through(tmp_path):
    gateway, guild, channels, _bot = make(tmp_path)

    async def scenario():
        for source in (message(guild, channels[21], 'set public club fact room to 214'),
                       message(guild, channels[20], 'set public club fact room to 214',
                               user_id=2, source=501)):
            with pytest.raises(PolicyDenied):
                await gateway.handle_control_message(source, source.content)
        assert gateway.club.current(10)['version'] == 0
        assert gateway.outbox.pending() == []
        await gateway.close()

    asyncio.run(scenario())


def test_officer_can_change_and_undo_a_fact_in_private_testing(tmp_path):
    gateway, guild, channels, _bot = make(tmp_path)

    async def scenario():
        request = message(guild, channels[22],
                          'set public club fact release_test_room to TEST ONLY 123', source=540)
        assert await gateway.handle_control_message(request, request.content)
        assert gateway.club.public_facts(10)[0]['value'] == 'TEST ONLY 123'
        denied = message(guild, channels[22],
                         'set public club fact release_test_room to WRONG', user_id=2, source=541)
        with pytest.raises(PolicyDenied):
            await gateway.handle_control_message(denied, denied.content)
        undo = message(guild, channels[22], 'undo last club fact', source=542)
        assert await gateway.handle_control_message(undo, undo.content)
        assert gateway.club.public_facts(10) == ()
        await gateway.close()

    asyncio.run(scenario())


def test_announcement_uses_bound_receipt_and_rechecks_revoked_role(tmp_path):
    gateway, guild, channels, bot = make(tmp_path)

    async def scenario():
        first = message(guild, channels[20], 'announce in <#30>: Meeting Friday.', source=510)
        assert await gateway.handle_control_message(first, first.content)
        assert len(bot.http.calls) == 1
        assert bot.http.calls[0][1]['json']['enforce_nonce'] is True
        action_id = gateway.outbox.db.execute(
            'SELECT id FROM announcements WHERE source_message_id=510').fetchone()['id']
        assert gateway.outbox.receipt_url(action_id).endswith('/44')
        # A replay of the same Discord source gets the durable receipt, never a second send.
        assert await gateway.handle_control_message(first, first.content)
        assert len(bot.http.calls) == 1
        guild.officer = False
        revoked = message(guild, channels[20], 'announce in <#30>: Never post this.', source=511)
        with pytest.raises(PolicyDenied):
            await gateway.handle_control_message(revoked, revoked.content)
        assert len(bot.http.calls) == 1
        await gateway.close()

    asyncio.run(scenario())


def test_role_revoked_between_proposal_and_send_blocks_dispatch(tmp_path):
    gateway, guild, channels, bot = make(tmp_path)
    original_fetch = guild.fetch_member

    async def revoke_at_destination_check(user_id):
        if user_id == bot.user.id:
            guild.officer = False
        return await original_fetch(user_id)

    guild.fetch_member = revoke_at_destination_check

    async def scenario():
        request = message(guild, channels[20], 'announce in <#30>: Do not send.', source=520)
        with pytest.raises(PolicyDenied):
            await gateway.handle_control_message(request, request.content)
        assert bot.http.calls == []
        assert gateway.outbox.pending()[0]['status'] == 'pending'
        await gateway.close()

    asyncio.run(scenario())


def test_uncertain_discord_send_is_frozen_for_review(tmp_path):
    gateway, guild, channels, bot = make(tmp_path)
    bot.http.request = AsyncMock(side_effect=asyncio.TimeoutError())

    async def scenario():
        request = message(guild, channels[20], 'announce in <#30>: Hold if uncertain.', source=530)
        assert await gateway.handle_control_message(request, request.content)
        assert 'uncertain' in request.sent[0]
        record = gateway.outbox.db.execute(
            'SELECT * FROM announcements WHERE source_message_id=530').fetchone()
        assert record['status'] == 'unknown'
        assert await gateway.handle_control_message(request, request.content)
        assert bot.http.request.await_count == 1
        await gateway.close()

    asyncio.run(scenario())


def test_saved_turn_survives_restart_without_cross_audience_recall(tmp_path):
    from unittest.mock import patch
    gateway, guild, channels, _bot = make(tmp_path)
    gateway.conversations.append_turn(guild_id=10, user_id=1, channel_id=21,
        source_message_id=600, audience='public',
        prompt='Build a controller. Do not publish it yet.', answer='I will keep it private.')

    async def inspect(service, principal, audience):
        with patch('peterbot.conversation.reply_or_use_tools', new=AsyncMock(return_value='okay')) as model:
            await service.conversational_reply(principal, 'continue', [], audience=audience)
            return model.await_args.args[4]

    async def scenario():
        assert 'Do not publish it yet' in str(await inspect(gateway, Principal(10, 1, 21, (100,)), 'public'))
        assert 'Do not publish it yet' not in str(await inspect(gateway, Principal(10, 2, 21), 'public'))
        assert 'Do not publish it yet' not in str(await inspect(gateway, Principal(10, 1, 20, (100,)), 'officer'))
        await gateway.close()
        restarted, _, _, _ = make(tmp_path)
        assert 'Do not publish it yet' in str(await inspect(restarted, Principal(10, 1, 21, (100,)), 'public'))
        await restarted.close()

    asyncio.run(scenario())
