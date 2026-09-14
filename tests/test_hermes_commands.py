import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import discord
import pytest

from peterbot.agent_policy import PolicyDenied, Principal
from peterbot.hermes_commands import register_agent_commands, submit_interaction
from test_command_admission import FakeBot, setup_handlers
from test_hermes_gateway import capability, gateway_client


def interaction(*, guild_id=10, user_id=1, channel_id=20):
    response = SimpleNamespace(done=False)

    async def defer(**kwargs):
        assert not response.done, "Interaction was deferred twice"
        response.done = True

    response.defer = AsyncMock(side_effect=defer)
    response.is_done = lambda: response.done
    response.send_message = AsyncMock()
    return SimpleNamespace(
        id=30, guild=SimpleNamespace(id=guild_id) if guild_id else None,
        user=SimpleNamespace(id=user_id), channel=SimpleNamespace(id=channel_id),
        channel_id=channel_id, response=response,
        followup=SimpleNamespace(send=AsyncMock()),
    )


def http_error():
    return discord.HTTPException(SimpleNamespace(status=500, reason="failure"), "Discord unavailable")


def test_task_defers_privately_before_network_and_binds_interaction_identity():
    async def scenario():
        request = interaction()

        async def submit(**kwargs):
            assert request.response.done
            assert kwargs == dict(guild_id=10, user_id=1, channel=request.channel,
                                  source_message_id=30, prompt="do work", parent_id=None, attachments=[])
            return {"id": "job", "guild_id": 10, "channel_id": 99}

        await submit_interaction(SimpleNamespace(submit=submit), request, "do work")
        request.response.defer.assert_awaited_once_with(ephemeral=True)
        args, kwargs = request.followup.send.await_args
        assert "https://discord.com/channels/10/99" in args[0]
        assert kwargs["ephemeral"] is True
        assert kwargs["allowed_mentions"].everyone is False

    asyncio.run(scenario())


def test_task_rejects_dms_after_deferring_without_calling_service():
    request = interaction(guild_id=None)
    service = SimpleNamespace(submit=AsyncMock())
    asyncio.run(submit_interaction(service, request, "do work"))
    service.submit.assert_not_awaited()
    assert "DMs are disabled" in request.followup.send.await_args.args[0]


def test_continue_command_preserves_actor_and_parent_task():
    bot, request = FakeBot(), interaction(user_id=2)
    service = SimpleNamespace(submit=AsyncMock(return_value={"id": "new", "guild_id": 10, "channel_id": 99}))
    register_agent_commands(bot, service)
    asyncio.run(bot.tree.callbacks["continue_task"](request, "parent", "more work"))
    assert service.submit.await_args.kwargs["parent_id"] == "parent"
    assert service.submit.await_args.kwargs["user_id"] == 2
    request.response.defer.assert_awaited_once_with(ephemeral=True)


def test_ask_defers_before_normal_chat_and_never_creates_a_task(setup_handlers):
    bot, runtime = setup_handlers
    from test_command_admission import interaction as chat_interaction
    request = chat_interaction()
    runtime.hermes = SimpleNamespace(eligible=AsyncMock(), submit=AsyncMock())

    async def chat(**kwargs):
        request.response.defer.assert_awaited_once_with(ephemeral=True)
        return "Just chatting."

    runtime.llm_client.call_chat.side_effect = chat
    asyncio.run(bot.tree.callbacks["ask"](request, "Tell me a joke"))
    runtime.hermes.eligible.assert_not_awaited()
    runtime.hermes.submit.assert_not_awaited()
    runtime.llm_client.call_chat.assert_awaited_once()


def task_channels(gateway):
    text = Mock(spec=discord.TextChannel)
    text.id = 20
    text.guild = gateway.bot.get_guild(10)
    thread = Mock(spec=discord.Thread)
    thread.id = 99
    thread.guild = text.guild
    thread.add_user = AsyncMock()
    thread.send = AsyncMock(return_value=SimpleNamespace(id=123))
    thread.delete = AsyncMock()
    thread.edit = AsyncMock()
    thread.is_private.return_value = True
    text.create_thread = AsyncMock(return_value=thread)
    gateway.principal = AsyncMock(return_value=Principal(10, 1, 20, (100,)))
    gateway.bot.fetch_channel = AsyncMock(side_effect=lambda channel_id: thread if channel_id == 99 else text)
    return text, thread


def test_new_tasks_create_noninvitable_private_threads_and_add_only_requester(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            text, thread = task_channels(gateway)
            job = await gateway.submit(guild_id=10, user_id=1, channel=text,
                                       source_message_id=30, prompt="do work")
            kwargs = text.create_thread.await_args.kwargs
            assert kwargs["type"] is discord.ChannelType.private_thread
            assert kwargs["invitable"] is False
            thread.add_user.assert_awaited_once_with(gateway.test_members[1])
            assert job["channel_id"] == 99
            assert job["user_id"] == 1
            assert thread.send.await_args.kwargs["allowed_mentions"].everyone is False

    asyncio.run(scenario())


def test_followup_requires_same_actor_and_reuses_original_thread(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path, officer_only=False) as (gateway, _):
            text, thread = task_channels(gateway)
            old = gateway.jobs.create(guild_id=10, user_id=1, channel_id=99,
                                      source_message_id=30, prompt="old objective")
            gateway.jobs.update(old["id"], status="completed", answer="old answer")
            with pytest.raises(ValueError, match="not found"):
                await gateway.submit(guild_id=10, user_id=2, channel=text,
                                     source_message_id=31, prompt="hijack", parent_id=old["id"])
            updated = await gateway.submit(guild_id=10, user_id=1, channel=text,
                                           source_message_id=32, prompt="continue", parent_id=old["id"])
            assert updated["parent_id"] == old["id"]
            assert updated["channel_id"] == 99
            text.create_thread.assert_not_awaited()

    asyncio.run(scenario())


def test_failure_to_add_requester_cleans_thread_and_never_queues_job(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            text, thread = task_channels(gateway)
            thread.add_user.side_effect = http_error()
            with pytest.raises(discord.HTTPException):
                await gateway.submit(guild_id=10, user_id=1, channel=text,
                                     source_message_id=30, prompt="do work")
            assert gateway.jobs.pending() == []
            assert thread.delete.await_count or thread.edit.await_count

    asyncio.run(scenario())


def test_queue_full_does_not_leave_an_empty_task_thread(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            text, thread = task_channels(gateway)
            for _ in range(2):
                gateway.jobs.create(guild_id=10, user_id=1, channel_id=99,
                                    source_message_id=30, prompt="existing")
            with pytest.raises(ValueError, match="queue is full"):
                await gateway.submit(guild_id=10, user_id=1, channel=text,
                                     source_message_id=30, prompt="extra")
            assert not text.create_thread.await_count or thread.delete.await_count or thread.edit.await_count

    asyncio.run(scenario())


def test_failed_queue_announcement_does_not_silently_run_job(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            text, thread = task_channels(gateway)
            thread.send.side_effect = http_error()
            try:
                await gateway.submit(guild_id=10, user_id=1, channel=text,
                                     source_message_id=30, prompt="do work")
            except discord.HTTPException:
                # If the user receives a failure, the task must not run secretly.
                assert gateway.jobs.pending() == []
            else:
                # Alternatively the caller receives a concrete saved task link.
                assert len(gateway.jobs.pending()) == 1

    asyncio.run(scenario())


def test_cancel_confirms_local_revocation_even_when_runner_is_unreachable(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            cap = capability(gateway)
            gateway.session.post = Mock(side_effect=aiohttp.ClientConnectionError("runner down"))
            bot, request = FakeBot(), interaction()
            register_agent_commands(bot, gateway)
            await bot.tree.callbacks["cancel_task"](request, cap.job["id"])
            assert gateway.jobs.get(cap.job["id"])["status"] == "cancelled"
            assert not gateway.capabilities
            request.response.defer.assert_awaited_once_with(ephemeral=True)
            assert "cancel" in request.followup.send.await_args.args[0].lower()

    asyncio.run(scenario())


def test_result_delivery_withholds_after_authority_is_revoked(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            _, thread = task_channels(gateway)
            cap = capability(gateway)
            gateway.jobs.update(cap.job["id"], status="completed", answer="private result")
            gateway.principal.side_effect = PolicyDenied("role revoked")
            await gateway.deliver(gateway.jobs.get(cap.job["id"]))
            thread.send.assert_not_awaited()
            assert gateway.jobs.undelivered() == []

    asyncio.run(scenario())


def test_delivery_retry_does_not_repeat_successfully_sent_chunks(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            _, thread = task_channels(gateway)
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=99, source_message_id=30, prompt="task")
            gateway.jobs.update(job["id"], status="completed", answer=("first paragraph " * 150) + "\n" + ("second paragraph " * 150))
            sent, calls = [], 0

            async def send(text=None, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise http_error()
                sent.append(text)
                return SimpleNamespace(id=100 + calls)

            thread.send.side_effect = send
            for _ in range(2):
                for pending in gateway.jobs.undelivered():
                    await gateway.deliver(pending)
            assert len(sent) > 1
            assert len(sent) == len(set(sent)), "Successful output chunks were delivered twice"
            assert gateway.jobs.undelivered() == []

    asyncio.run(scenario())


def test_malformed_artifact_cannot_poison_queue_or_repeat_answer(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _):
            _, thread = task_channels(gateway)
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=99, source_message_id=30, prompt="task")
            gateway.jobs.update(job["id"], status="completed", answer="the answer",
                                artifacts=[{"name": "bad.bin", "data_base64": "not base64!!!"},
                                           {"name": "good.txt", "data_base64": base64.b64encode(b"valid").decode()}])
            for _ in range(2):
                for pending in gateway.jobs.undelivered():
                    await gateway.deliver(pending)
            sent_answers = [call.args[0] for call in thread.send.await_args_list
                            if call.args and call.args[0] == "the answer"]
            assert len(sent_answers) == 1
            assert gateway.jobs.undelivered() == []

    asyncio.run(scenario())
