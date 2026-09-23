"""Exercise registered Discord handlers with the real admission guard."""

import asyncio
import itertools
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import discord

import peterbot.commands as handlers
import peterbot.context as delivery
from peterbot.foreground import ForegroundScheduler
from peterbot.guardrails import GuardLimits, RequestGuard
from peterbot.agent_policy import PolicyDenied
from peterbot.agent_policy import Principal
from test_llama_cpp_client import build_config

_interaction_ids = itertools.count(500)


class FakeTree:
    def __init__(self):
        self.callbacks = {}

    def command(self, *, name, description):
        def register(callback):
            self.callbacks[name] = callback
            return callback
        return register

    def error(self, callback):
        return callback


class FakeBot:
    def __init__(self):
        self.tree = FakeTree()
        self.user = SimpleNamespace(id=999)
        self.events = {}
        self.process_commands = AsyncMock()

    def event(self, callback):
        self.events[callback.__name__] = callback
        return callback


@pytest.fixture
def setup_handlers(tmp_path, monkeypatch):
    bot = FakeBot()
    config = build_config(tmp_path)
    runtime = SimpleNamespace(
        config=config,
        request_guard=RequestGuard(GuardLimits()),
        llm_client=SimpleNamespace(call_chat=AsyncMock(return_value="Here is the answer.")),
        knowledge_index=SimpleNamespace(chunks=[], channel_profiles={}),
        foreground=ForegroundScheduler(str(tmp_path / "foreground.sqlite3"),
                                       ack_after=0.05, poll=0.01, total_timeout=0.3),
    )
    monkeypatch.setattr(handlers, "get_channel_context_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "get_recent_channel_entries", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "safe_send_interaction_message", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "send_chunked_followup", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "send_chunked_reply", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "build_prompt_artifacts", lambda **kwargs: ("You are Peter.", []))
    handlers.register_handlers(bot, runtime)
    return bot, runtime


def interaction(user_id=1, interaction_id=None):
    return SimpleNamespace(
        id=interaction_id or next(_interaction_ids),
        user=SimpleNamespace(id=user_id, display_name="Student"),
        guild=SimpleNamespace(id=10, name="Club"),
        channel=SimpleNamespace(id=20, name="hardware"),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
        created_at=datetime.now(timezone.utc),
    )




def assert_another_user_can_start(runtime):
    assert runtime.request_guard.acquire(user_id=22, guild_id=10, prompt="next") == (True, None)
    runtime.request_guard.release(user_id=22)


@pytest.mark.parametrize("failure", ["defer", "history", "model", "delivery", "fallback"])
def test_ask_releases_slot_after_failures(setup_handlers, monkeypatch, failure):
    bot, runtime = setup_handlers
    request = interaction()
    error = RuntimeError("simulated downstream failure")
    if failure == "defer":
        request.response.defer.side_effect = error
    elif failure == "history":
        handlers.get_channel_context_messages.side_effect = error
    elif failure == "model":
        runtime.llm_client.call_chat.side_effect = error
    elif failure == "delivery":
        handlers.send_chunked_followup.side_effect = error
    else:
        runtime.llm_client.call_chat.side_effect = error
        handlers.safe_send_interaction_message.side_effect = error
    if failure == "fallback":
        with pytest.raises(RuntimeError, match="downstream"):
            asyncio.run(bot.tree.callbacks["ask"](request, "help with hardware"))
    else:
        asyncio.run(bot.tree.callbacks["ask"](request, "help with hardware"))
    assert_another_user_can_start(runtime)


def test_recap_empty_history_early_return_releases_slot(setup_handlers):
    bot, runtime = setup_handlers
    asyncio.run(bot.tree.callbacks["recap"](interaction(), 10))
    runtime.llm_client.call_chat.assert_not_awaited()
    assert "enough recent messages" in handlers.safe_send_interaction_message.await_args.args[1]
    assert_another_user_can_start(runtime)


def test_recap_respects_officer_pilot_before_reading_history(setup_handlers):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(allowed_guild_ids=frozenset({10}),
                                 listen_channel_ids=frozenset()),
        principal=AsyncMock(side_effect=PolicyDenied('The agent pilot is currently available to officers only')),
        style=SimpleNamespace(instruction=Mock()),
    )
    asyncio.run(bot.tree.callbacks['recap'](interaction(), 10))
    handlers.get_recent_channel_entries.assert_not_awaited()
    runtime.llm_client.call_chat.assert_not_awaited()
    assert 'officers only' in handlers.safe_send_interaction_message.await_args.args[1]


def test_recap_uses_current_style_instruction(setup_handlers, monkeypatch):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(allowed_guild_ids=frozenset({10}),
                                 listen_channel_ids=frozenset()),
        principal=AsyncMock(return_value=Principal(10, 1, 20, (100,))),
        style=SimpleNamespace(instruction=Mock(return_value='Keep the recap relaxed.')),
    )
    handlers.get_recent_channel_entries.return_value = [object()]
    monkeypatch.setattr(handlers, 'build_recap_history', lambda *args: [])
    asyncio.run(bot.tree.callbacks['recap'](interaction(), 10))
    runtime.hermes.principal.assert_awaited_once()
    assert 'Keep the recap relaxed.' in runtime.llm_client.call_chat.await_args.kwargs['system_prompt']


@pytest.mark.parametrize("failure", ["defer", "history"])
def test_recap_releases_slot_after_failure(setup_handlers, failure):
    bot, runtime = setup_handlers
    request = interaction()
    if failure == "defer":
        request.response.defer.side_effect = RuntimeError("defer failed")
    else:
        handlers.get_recent_channel_entries.side_effect = RuntimeError("history failed")
    asyncio.run(bot.tree.callbacks["recap"](request, 10))
    assert_another_user_can_start(runtime)


def test_queued_ask_reads_no_history_and_makes_no_model_call(setup_handlers):
    """PETER-04: a second request waits in the FIFO queue. While queued it is
    a deterministic ack only — no Discord reads, no model call — and its
    rejection never touches the running request's guard state."""
    bot, runtime = setup_handlers
    entered, never = asyncio.Event(), asyncio.Event()

    async def stalled_model(*args, **kwargs):
        entered.set()
        await never.wait()

    async def scenario():
        runtime.llm_client.call_chat.side_effect = stalled_model
        first = asyncio.create_task(bot.tree.callbacks["ask"](interaction(1, 501), "first"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(bot.tree.callbacks["ask"](interaction(2, 502), "queued"))
        await asyncio.sleep(0.15)
        # Only the running request ever read history; the queued one produced
        # just its ephemeral ack.
        assert handlers.get_channel_context_messages.await_count == 1
        assert handlers.safe_send_interaction_message.await_args.kwargs["ephemeral"] is True
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        never.set()
        await asyncio.wait_for(first, timeout=1)

    asyncio.run(scenario())
    assert_another_user_can_start(runtime)


@pytest.mark.parametrize("command", ["ask", "recap"])
def test_handler_cancellation_propagates_and_releases_slot(setup_handlers, command):
    bot, runtime = setup_handlers

    async def scenario():
        entered = asyncio.Event()
        never = asyncio.Event()

        async def stalled_history(*args, **kwargs):
            entered.set()
            await never.wait()

        history = (handlers.get_channel_context_messages if command == "ask"
                   else handlers.get_recent_channel_entries)
        history.side_effect = stalled_history
        argument = "question" if command == "ask" else 10
        task = asyncio.create_task(bot.tree.callbacks[command](interaction(), argument))
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert not runtime.request_guard.acquire(user_id=2, guild_id=10, prompt="busy")[0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_another_user_can_start(runtime)

    asyncio.run(scenario())


def test_handler_deadline_includes_discord_history_fetch(setup_handlers):
    bot, runtime = setup_handlers
    runtime.config = replace(runtime.config, agent=replace(runtime.config.agent, request_timeout_seconds=0.01))
    # The handlers capture config when registered, as they do during application startup.
    handlers.register_handlers(bot, runtime)

    async def scenario():
        stopped = asyncio.Event()

        async def stalled_history(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        handlers.get_channel_context_messages.side_effect = stalled_history
        await asyncio.wait_for(bot.tree.callbacks["ask"](interaction(), "question"), timeout=1)
        assert stopped.is_set()
        runtime.llm_client.call_chat.assert_not_awaited()
        assert_another_user_can_start(runtime)

    asyncio.run(scenario())


def mention(bot):
    return SimpleNamespace(
        id=42, author=SimpleNamespace(id=1, display_name="Student", bot=False),
        mentions=[bot.user], content="<@999> what is this?", attachments=[],
        guild=SimpleNamespace(id=10, name="Club"),
        channel=SimpleNamespace(id=20, name="hardware"),
        created_at=datetime.now(timezone.utc),
    )


def test_private_control_request_without_mention_uses_trusted_gateway(setup_handlers):
    bot, runtime = setup_handlers
    handle = AsyncMock(return_value=True)
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(control_channel_ids=frozenset({20}),
                                 listen_channel_ids=frozenset(), allowed_guild_ids=frozenset({10})),
        handle_control_message=handle,
    )
    request = mention(bot)
    request.mentions = []
    request.content = 'set public club fact meeting_room to KEC 1005'
    asyncio.run(bot.events['on_message'](request))
    handle.assert_awaited_once()
    runtime.llm_client.call_chat.assert_not_awaited()
    bot.process_commands.assert_awaited_once()


def test_denied_control_request_never_falls_to_legacy_model(setup_handlers):
    bot, runtime = setup_handlers
    handle = AsyncMock(side_effect=PolicyDenied('Use the configured private officer control channel'))
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(control_channel_ids=frozenset({20}),
                                 listen_channel_ids=frozenset(), allowed_guild_ids=frozenset({10})),
        handle_control_message=handle,
    )
    request = mention(bot)
    request.mentions = []
    request.content = 'set public club fact meeting_room to KEC 1005'
    asyncio.run(bot.events['on_message'](request))
    handle.assert_awaited_once()
    runtime.llm_client.call_chat.assert_not_awaited()
    assert 'private officer' in handlers.send_chunked_reply.await_args.args[1]


def test_denied_club_mention_never_falls_to_legacy_model(setup_handlers):
    bot, runtime = setup_handlers
    reply = AsyncMock(side_effect=PolicyDenied('The agent pilot is currently available to officers only'))
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(control_channel_ids=frozenset(),
                                 listen_channel_ids=frozenset(), allowed_guild_ids=frozenset({10})),
        eligible=AsyncMock(return_value=False), respond_to_message=reply,
    )
    asyncio.run(bot.events['on_message'](mention(bot)))
    reply.assert_awaited_once()
    runtime.hermes.eligible.assert_not_awaited()
    runtime.llm_client.call_chat.assert_not_awaited()
    assert 'officers only' in handlers.send_chunked_reply.await_args.args[1]


def test_ask_uses_fresh_hermes_facts_and_private_context_when_enabled(setup_handlers):
    bot, runtime = setup_handlers
    saved = Mock()
    hermes = SimpleNamespace(
        principal=AsyncMock(return_value=Principal(10, 1, 20, (100,))),
        conversational_reply=AsyncMock(return_value='KEC 1005 on Fridays.'),
        conversations=SimpleNamespace(append_turn=saved),
    )
    runtime.hermes = hermes
    asyncio.run(bot.tree.callbacks['ask'](interaction(1, 801), 'Where do we meet?'))
    hermes.conversational_reply.assert_awaited_once()
    assert hermes.conversational_reply.await_args.kwargs['audience'] == 'private'
    assert 0 < hermes.conversational_reply.await_args.kwargs['budget_seconds'] <= runtime.config.agent.request_timeout_seconds
    assert handlers.send_chunked_followup.await_args.args[1] == 'KEC 1005 on Fridays.'
    assert saved.call_args.kwargs['audience'] == 'private'
    runtime.llm_client.call_chat.assert_not_awaited()


def test_ask_handoff_keeps_foreground_slot_and_never_uses_legacy_fallback(setup_handlers):
    bot, runtime = setup_handlers
    hermes = SimpleNamespace(
        principal=AsyncMock(return_value=Principal(10, 1, 20, (100,))),
        conversational_reply=AsyncMock(return_value=None),
        jobs=SimpleNamespace(latest_for_thread=lambda *args: None),
        submit=AsyncMock(return_value={'guild_id': 10, 'channel_id': 21}),
        conversations=SimpleNamespace(append_turn=Mock()),
    )
    runtime.hermes = hermes
    asyncio.run(bot.tree.callbacks['ask'](interaction(1, 802), 'Research the latest Rust release'))
    hermes.submit.assert_awaited_once()
    assert 'https://discord.com/channels/10/21' in handlers.send_chunked_followup.await_args.args[1]
    hermes.conversations.append_turn.assert_not_called()
    runtime.llm_client.call_chat.assert_not_awaited()


def test_ask_privileged_denial_has_no_legacy_fallback(setup_handlers):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(principal=AsyncMock(side_effect=PolicyDenied('Officer pilot')))
    asyncio.run(bot.tree.callbacks['ask'](interaction(1, 803), 'Run code'))
    runtime.llm_client.call_chat.assert_not_awaited()
    handlers.send_chunked_followup.assert_not_awaited()
    assert 'Officer pilot' in handlers.safe_send_interaction_message.await_args.args[1]


def test_unusable_mention_image_early_return_releases_slot(setup_handlers, monkeypatch):
    bot, runtime = setup_handlers
    monkeypatch.setattr(handlers, "resolve_mention_images", AsyncMock(return_value=([], "Image unavailable.")))
    asyncio.run(bot.events["on_message"](mention(bot)))
    handlers.get_recent_channel_entries.assert_not_awaited()
    runtime.llm_client.call_chat.assert_not_awaited()
    bot.process_commands.assert_awaited_once()
    assert_another_user_can_start(runtime)


def test_mention_history_failure_releases_slot(setup_handlers, monkeypatch):
    bot, runtime = setup_handlers
    monkeypatch.setattr(handlers, "resolve_mention_images", AsyncMock(return_value=([], None)))
    handlers.get_recent_channel_entries.side_effect = RuntimeError("history unavailable")
    asyncio.run(bot.events["on_message"](mention(bot)))
    runtime.llm_client.call_chat.assert_not_awaited()
    assert_another_user_can_start(runtime)


def test_mention_clarification_early_return_releases_slot(setup_handlers, monkeypatch):
    bot, runtime = setup_handlers
    monkeypatch.setattr(handlers, "resolve_mention_images", AsyncMock(return_value=([], None)))
    monkeypatch.setattr(handlers, "resolve_reply_target_entry", AsyncMock(return_value=None))
    monkeypatch.setattr(handlers, "build_mention_context_bundle", lambda *args, **kwargs: {
        "clarification_text": "Which message?", "selection_reason": "ambiguous",
        "target_message_id": None, "target_age_text": None, "selected_count": 0,
        "needs_strong_target": True,
    })
    asyncio.run(bot.events["on_message"](mention(bot)))
    runtime.llm_client.call_chat.assert_not_awaited()
    assert handlers.send_chunked_reply.await_args.args[1] == "Which message?"
    assert_another_user_can_start(runtime)


def assert_silent_delivery(call):
    assert call.kwargs["suppress_embeds"] is True
    assert call.kwargs["allowed_mentions"].to_dict() == {"parse": []}


def test_every_reply_chunk_suppresses_embeds_and_mentions():
    message = SimpleNamespace(reply=AsyncMock(), channel=SimpleNamespace(send=AsyncMock()))
    text = "@everyone https://www.kernel.org/\n\n<@123> https://docs.python.org/3/"
    assert asyncio.run(delivery.send_chunked_reply(message, text, max_len=35))
    assert message.reply.await_count == 1
    assert message.channel.send.await_count >= 1
    for call in [*message.reply.await_args_list, *message.channel.send.await_args_list]:
        assert_silent_delivery(call)


def test_every_followup_chunk_suppresses_embeds_and_mentions():
    request = SimpleNamespace(followup=SimpleNamespace(send=AsyncMock()))
    text = "@everyone https://www.kernel.org/\n\n<@123> https://docs.python.org/3/"
    assert asyncio.run(delivery.send_chunked_followup(request, text, max_len=35))
    assert request.followup.send.await_count >= 2
    for call in request.followup.send.await_args_list:
        assert_silent_delivery(call)
        assert call.kwargs["ephemeral"] is True


@pytest.mark.parametrize("response_state", ["new", "done", "raced"])
def test_interaction_delivery_suppresses_embeds_and_mentions_in_each_response_path(response_state):
    request = SimpleNamespace(
        response=SimpleNamespace(is_done=lambda: response_state == "done", send_message=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    if response_state == "raced":
        request.response.send_message.side_effect = discord.InteractionResponded(request)
    assert asyncio.run(delivery.safe_send_interaction_message(
        request, "@everyone see https://www.kernel.org/",
    ))
    sent = request.response.send_message if response_state == "new" else request.followup.send
    sent.assert_awaited_once()
    assert_silent_delivery(sent.await_args)
    assert sent.await_args.kwargs["ephemeral"] is True
