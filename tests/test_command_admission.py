"""Exercise registered Discord handlers with the real admission guard."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import discord

import peterbot.commands as handlers
import peterbot.context as delivery
from peterbot.guardrails import GuardLimits, RequestGuard
from test_llama_cpp_client import build_config


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
    )
    monkeypatch.setattr(handlers, "get_channel_context_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "get_recent_channel_entries", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "safe_send_interaction_message", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "send_chunked_followup", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "send_chunked_reply", AsyncMock(return_value=True))
    monkeypatch.setattr(handlers, "build_prompt_artifacts", lambda **kwargs: ("You are Peter.", []))
    handlers.register_handlers(bot, runtime)
    return bot, runtime


def interaction(user_id=1):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="Student"),
        guild=SimpleNamespace(id=10, name="Club"),
        channel=SimpleNamespace(id=20, name="hardware"),
        response=SimpleNamespace(defer=AsyncMock()),
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


def test_rejected_ask_cannot_release_active_request_or_read_history(setup_handlers):
    bot, runtime = setup_handlers
    assert runtime.request_guard.acquire(user_id=1, guild_id=10, prompt="first")[0]
    asyncio.run(bot.tree.callbacks["ask"](interaction(1), "duplicate"))
    asyncio.run(bot.tree.callbacks["ask"](interaction(2), "busy"))
    handlers.get_channel_context_messages.assert_not_awaited()
    runtime.llm_client.call_chat.assert_not_awaited()
    assert not runtime.request_guard.acquire(user_id=3, guild_id=10, prompt="still busy")[0]
    runtime.request_guard.release(user_id=1)
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
