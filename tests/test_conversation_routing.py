"""Conversation stays in place; public work never inherits private task context."""
import asyncio
import json
import time
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from peterbot.agent_policy import PolicyDenied, Principal
from peterbot.hermes_gateway import Capability, HermesGateway
from peterbot.hermes_settings import HermesSettings
from test_hermes_gateway import UpstreamSession
from test_command_admission import interaction, mention, setup_handlers
import peterbot.commands as handlers


@asynccontextmanager
async def conversation_gateway(tmp_path):
    member = SimpleNamespace(id=1, bot=False, display_name="Officer", roles=[SimpleNamespace(id=100)])
    guild = SimpleNamespace(id=10, name="Club", fetch_member=AsyncMock(return_value=member))
    channel = SimpleNamespace(id=20, name="hardware", guild=guild, send=AsyncMock(),
                              create_thread=AsyncMock(), typing=lambda: nullcontext(), permissions_for=lambda actor: SimpleNamespace(view_channel=True))
    bot = SimpleNamespace(user=SimpleNamespace(id=999), get_guild=lambda guild_id: guild,
                          fetch_channel=AsyncMock(return_value=channel))
    config = SimpleNamespace(agent=SimpleNamespace(search_base_url=""),
                             inference=SimpleNamespace(model="qwen", base_url="http://model/v1"),
                             llama_cpp_api_key="", peter_system_prompt="Be Peter.",
                             peter_name="Peter", channel_context_limit=10, max_context_message_chars=1000,
                             max_discord_message_chars=1800)
    settings = HermesSettings(allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
                             owner_user_ids=frozenset({1}), runner_url="http://runner", tool_service_url="http://gateway",
                             runner_token="r" * 40, state_dir=str(tmp_path))
    gateway = HermesGateway(bot, config, settings)
    gateway.session = UpstreamSession()
    try:
        yield gateway, channel
    finally:
        await gateway.close()
        gateway.jobs.close()


def public_job(gateway, **kwargs):
    return gateway.jobs.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                               prompt="Look this up", delivery_mode="channel", **kwargs)


def test_channel_submission_does_not_create_thread_or_emit_status(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            job = await gateway.submit(guild_id=10, user_id=1, channel=channel, source_message_id=30,
                                       prompt="Look this up", in_channel=True)
            assert job["channel_id"] == 20
            assert job["delivery_mode"] == "channel"
            assert job["source_message_id"] == 30
            assert job["status"] == "queued"
            channel.create_thread.assert_not_awaited()
            channel.send.assert_not_awaited()
    asyncio.run(scenario())


def test_public_jobs_are_not_implicit_private_thread_followups(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            job = public_job(gateway)
            assert gateway.jobs.latest_for_thread(10, 1, 20) is None
            gateway.jobs.update(job["id"], status="completed")
            private = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21, source_message_id=31, prompt="Private task")
            assert private["delivery_mode"] == "private"
            assert gateway.jobs.latest_for_thread(10, 1, 21)["id"] == private["id"]
    asyncio.run(scenario())


def test_public_submission_cannot_continue_private_task(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            private = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21, source_message_id=31, prompt="Private details")
            gateway.jobs.update(private["id"], status="completed", answer="Private answer")
            with pytest.raises((ValueError, PolicyDenied)):
                await gateway.submit(guild_id=10, user_id=1, channel=channel, source_message_id=30,
                                     prompt="Tell everyone", parent_id=private["id"], in_channel=True)
            channel.send.assert_not_awaited()
            channel.create_thread.assert_not_awaited()
    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["peter_memory_search", "peter_memory_add", "peter_memory_update", "peter_memory_delete"])
def test_public_tools_cannot_read_or_change_even_requesters_personal_memory(tmp_path, name):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            principal = Principal(10, 1, 20, (100,))
            personal = gateway.memory.create(principal, scope="personal", content="private secret", source_message_id=30)
            cap = Capability(public_job(gateway), time.monotonic() + 60)
            arguments = {
                "peter_memory_search": {"scope": "personal"},
                "peter_memory_add": {"scope": "personal", "content": "new fact"},
                "peter_memory_update": {"memory_id": personal["id"], "content": "changed", "expected_version": 1},
                "peter_memory_delete": {"memory_id": personal["id"], "expected_version": 1},
            }[name]
            with pytest.raises(PolicyDenied) as error:
                await gateway.dispatch_tool(cap, principal, name, arguments)
            assert "private secret" not in str(error.value)
            assert gateway.memory.get(principal, personal["id"])["content"] == "private secret"
            assert len(gateway.memory.search(principal, scope="personal")) == 1
    asyncio.run(scenario())


def test_private_task_retains_personal_memory_access(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            principal = Principal(10, 1, 20, (100,))
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30, prompt="Private task")
            cap = Capability(job, time.monotonic() + 60)
            await gateway.dispatch_tool(cap, principal, "peter_memory_add", {"scope": "personal", "content": "private secret"})
            result = await gateway.dispatch_tool(cap, principal, "peter_memory_search", {"scope": "personal"})
            assert result["memories"][0]["content"] == "private secret"
    asyncio.run(scenario())


@pytest.mark.parametrize("public", [True, False])
def test_worker_request_scopes_memory_and_suppresses_public_status(tmp_path, public):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            principal = Principal(10, 1, 20, (100,))
            gateway.memory.create(principal, scope="personal", content="private secret", source_message_id=30)
            gateway.memory.create(principal, scope="club", content="public club fact", source_message_id=30)
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=20, source_message_id=30,
                                      prompt="Help", delivery_mode="channel" if public else "private")
            gateway.jobs.update(job["id"], status="running")
            gateway.session.result = {"status": "completed", "answer": "The answer."}
            await gateway.run_job(job)
            payload = next(kwargs["json"]["request"] for url, kwargs in gateway.session.calls if url.endswith("/run"))
            assert "public club fact" in json.dumps(payload["memory_snapshots"])
            assert ("private secret" in json.dumps(payload)) is not public
            if public:
                assert payload["response_style"] == "conversation"
                channel.send.assert_not_awaited()
            else:
                assert channel.send.await_count >= 1
    asyncio.run(scenario())


def test_public_delivery_returns_answer_without_task_wrapper(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            job = public_job(gateway)
            gateway.jobs.update(job["id"], status="completed", answer="Yep, that works.")
            await gateway.deliver(gateway.jobs.get(job["id"]))
            channel.send.assert_awaited_once()
            assert channel.send.await_args.args == ("Yep, that works.",)
            assert channel.send.await_args.kwargs["allowed_mentions"].to_dict() == {"parse": []}
            assert channel.send.await_args.kwargs["reference"].message_id == 30
            assert gateway.jobs.get(job["id"])["delivered"] == 1
    asyncio.run(scenario())


def test_mention_uses_conversation_entry_point_without_task_link(setup_handlers):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(eligible=AsyncMock(return_value=True), respond_to_message=AsyncMock(),
                                     submit=AsyncMock(), jobs=SimpleNamespace(latest_for_thread=lambda *args: None))
    message = mention(bot)
    asyncio.run(bot.events["on_message"](message))
    runtime.hermes.respond_to_message.assert_awaited_once()
    assert runtime.hermes.respond_to_message.await_args.args[0] is message
    runtime.hermes.submit.assert_not_awaited()
    handlers.send_chunked_reply.assert_not_awaited()
    runtime.llm_client.call_chat.assert_not_awaited()


def test_ask_keeps_normal_reply_when_hermes_is_enabled(setup_handlers):
    bot, runtime = setup_handlers
    runtime.hermes = SimpleNamespace(eligible=AsyncMock(return_value=True), submit=AsyncMock())
    request = interaction()
    request.channel_id = request.channel.id
    asyncio.run(bot.tree.callbacks["ask"](request, "Tell me a joke"))
    runtime.hermes.submit.assert_not_awaited()
    runtime.llm_client.call_chat.assert_awaited_once()
    assert handlers.send_chunked_followup.await_args.args[1] == "Here is the answer."


@pytest.mark.parametrize("answer", ["lol, fair.", None])
def test_conversation_replies_directly_or_escalates_visibly(tmp_path, answer):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            channel.send.return_value = SimpleNamespace(id=777, edit=AsyncMock())
            gateway.conversational_reply = AsyncMock(return_value=answer)
            gateway.submit = AsyncMock()
            message = SimpleNamespace(id=30, guild=channel.guild, channel=channel,
                                      author=SimpleNamespace(id=1, display_name="Officer", bot=False),
                                      content="Hey Peter", attachments=[], reply=AsyncMock(),
                                      created_at=datetime.now(timezone.utc))
            await gateway.respond_to_message(message, "Hey Peter")
            channel.create_thread.assert_not_awaited()
            gateway.conversational_reply.assert_awaited_once()
            if answer is None:
                # Work that runs for minutes is announced, in the message that will later
                # hold the answer, instead of leaving the channel silent.
                message.reply.assert_not_awaited()
                channel.send.assert_awaited_once()
                assert "on it" in channel.send.await_args.args[0]
                gateway.submit.assert_awaited_once()
                submitted = gateway.submit.await_args.kwargs
                assert submitted["in_channel"] is True
                assert submitted["channel"] is channel
                assert submitted["source_message_id"] == 30
                assert submitted["status_message_id"] == 777
            else:
                # A quick answer never posts a placeholder first.
                channel.send.assert_not_awaited()
                gateway.submit.assert_not_awaited()
                message.reply.assert_awaited_once()
                assert message.reply.await_args.args[0] == answer
    asyncio.run(scenario())


def test_public_work_cannot_be_resumed_as_a_private_task_in_public_channel(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            job = public_job(gateway)
            gateway.jobs.update(job["id"], status="completed", answer="Public result")
            with pytest.raises((PolicyDenied, ValueError)):
                await gateway.submit(guild_id=10, user_id=1, channel=channel, source_message_id=40,
                                     prompt="Continue", parent_id=job["id"])
            channel.send.assert_not_awaited()
            assert gateway.jobs.pending() == []
    asyncio.run(scenario())


def test_worker_context_is_only_requesters_recent_completed_work_in_same_channel(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            for label, user_id, guild_id, channel_id, mode, status in [
                ("mine", 1, 10, 20, "channel", "completed"),
                ("another member", 2, 10, 20, "channel", "completed"),
                ("another guild", 1, 11, 20, "channel", "completed"),
                ("another channel", 1, 10, 21, "channel", "completed"),
                ("private conversation", 1, 10, 20, "private", "completed"),
                ("failed request", 1, 10, 20, "channel", "failed"),
            ]:
                job = gateway.jobs.create(guild_id=guild_id, user_id=user_id, channel_id=channel_id,
                                          source_message_id=30, prompt=label, delivery_mode=mode)
                gateway.jobs.update(job["id"], status=status, answer=label + " answer")
            assert gateway.jobs.conversation_context(10, 1, 20) == [
                {"role": "user", "content": "mine"}, {"role": "assistant", "content": "mine answer"}]
    asyncio.run(scenario())


@pytest.mark.parametrize("handoff", [False, True])
def test_fast_model_answers_or_hands_off_without_exposing_reasoning(tmp_path, handoff):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            message = {"content": "<think>private thoughts</think>Hey."}
            if handoff:
                message["tool_calls"] = [{"id": "call-1", "type": "function",
                                          "function": {"name": "use_tools", "arguments": '{"reason":"Need tools"}'}}]
            gateway.session.result = {"choices": [{"message": message}]}
            reply = await gateway.conversational_reply(Principal(10, 1, 20, (100,)), "Hi", [])
            assert reply == (None if handoff else "Hey.")
            url, request = gateway.session.calls[0]
            assert url == "http://model/v1/chat/completions"
            assert request["allow_redirects"] is False
            assert request["json"]["chat_template_kwargs"]["enable_thinking"] is True
            assert [tool["function"]["name"] for tool in request["json"]["tools"]] == ["use_tools"]
            assert gateway.jobs.pending() == []
    asyncio.run(scenario())


@pytest.mark.parametrize("name,arguments", [("terminal", "{}"), ("use_tools", '{"user_id":2}')])
def test_invented_fast_model_tool_decision_hands_off_instead_of_erroring(tmp_path, name, arguments):
    """A tool name or argument shape the fast model invented is a model wobble, not a
    member-facing error: hand the request to the sandbox, which re-checks authority and
    honours only its own allowlist."""
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, _):
            gateway.session.result = {"choices": [{"message": {"tool_calls": [
                {"function": {"name": name, "arguments": arguments}}]}}]}
            reply = await gateway.conversational_reply(Principal(10, 1, 20, (100,)), "Hi", [])
            assert reply is None
            assert gateway.jobs.pending() == []
    asyncio.run(scenario())
