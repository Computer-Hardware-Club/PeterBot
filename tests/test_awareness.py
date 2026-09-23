import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord

from peterbot.awareness import AwarenessRouter
from test_command_admission import setup_handlers


def message(content, *, author_id=1, message_id=10, channel_id=20, reference=None):
    guild = SimpleNamespace(id=10, name="Club")
    channel = SimpleNamespace(id=channel_id, guild=guild)
    return SimpleNamespace(id=message_id, content=content, guild=guild, channel=channel,
                           author=SimpleNamespace(id=author_id, bot=False, display_name="Student"),
                           mentions=[], reference=reference, webhook_id=None,
                           type=discord.MessageType.default, attachments=[])


def router(clock=lambda: 0):
    return AwarenessRouter(guild_ids=frozenset({10}), channel_ids=frozenset({20}),
                           bot_user_id=999, clock=clock)


def test_name_reply_and_same_user_lease_without_ambient_model_calls():
    async def scenario():
        current = [0]
        detector = router(clock=lambda: current[0])
        first = message("Hey Peter, can you help?", message_id=1)
        assert await detector.addressed(first) == "name"
        detector.remember(first, "name")
        assert await detector.addressed(first) is None
        assert await detector.addressed(message("can you write some Rust?", message_id=2)) == "followup"
        assert await detector.addressed(message("can you write some Rust?", author_id=2, message_id=3)) is None
        reply = message("Can you check this?", author_id=2, message_id=4,
                        reference=SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=999))))
        assert await detector.addressed(reply) == "reply"
        detector.remember(reply, "reply")
        assert await detector.addressed(message("one more thing", author_id=2, channel_id=21, message_id=5)) is None
        current[0] = 121
        assert await detector.addressed(message("can you write some Rust?", message_id=6)) is None
    asyncio.run(scenario())


def test_discussion_code_urls_webhooks_and_system_messages_are_ignored():
    async def scenario():
        detector = router()
        for text in ("Peter said that yesterday", "```python\nPeter, help me\n```",
                     "> Peter, help me", "https://example.org/Peter", "!peter help"):
            assert await detector.addressed(message(text)) is None
        webhook = message("Hey Peter", message_id=2)
        webhook.webhook_id = 77
        assert await detector.addressed(webhook) is None
        system = message("Hey Peter", message_id=3)
        system.type = discord.MessageType.channel_name_change
        assert await detector.addressed(system) is None
        other_guild = message("Hey Peter", message_id=4)
        other_guild.guild.id = 11
        assert await detector.addressed(other_guild) is None
    asyncio.run(scenario())


def test_name_and_followup_enter_existing_conversation_handler(setup_handlers):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(
        settings=SimpleNamespace(allowed_guild_ids=frozenset({10}),
                                 listen_channel_ids=frozenset({20}), conversation_lease_seconds=120),
        eligible=AsyncMock(return_value=True), respond_to_message=AsyncMock())
    first = message("Hey Peter", message_id=1)
    followup = message("can you help me write this?", message_id=2)
    unrelated = message("I was talking to Scott", author_id=2, message_id=3)

    async def scenario():
        await bot.events["on_message"](first)
        await bot.events["on_message"](followup)
        await bot.events["on_message"](unrelated)

    asyncio.run(scenario())
    assert runtime.hermes.respond_to_message.await_count == 2
    assert runtime.hermes.respond_to_message.await_args_list[0].args[0] is first
    assert runtime.hermes.respond_to_message.await_args_list[1].args[0] is followup
    runtime.llm_client.call_chat.assert_not_awaited()
