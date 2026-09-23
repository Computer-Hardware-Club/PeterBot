import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import aiohttp
import pytest

from peterbot.agent_jobs import JobStore
from peterbot.agent_policy import PolicyDenied, Principal
from peterbot.hermes_gateway import Capability, HermesGateway
from peterbot import package_access
from peterbot.package_access import PACKAGE_TASK_BYTES, PackageBroker, PackageError
from peterbot.hermes_settings import HermesSettings


def sse_lines(result):
    """Render a completion dict the way a streaming server would: deltas, then finish.

    Tool arguments are deliberately split across two deltas because that is how real
    servers send them, so the reassembly is exercised by every test using this double.
    """
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason") or "stop"
    lines = []

    def emit(delta):
        lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}))

    if message.get("reasoning"):
        emit({"reasoning": message["reasoning"]})
    if message.get("content"):
        emit({"content": message["content"]})
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        emit({"tool_calls": [{"index": index, "id": call.get("id", f"call-{index}"), "type": "function",
                              "function": {"name": function.get("name", ""), "arguments": ""}}]})
        arguments = function.get("arguments") or ""
        if arguments:
            middle = max(1, len(arguments) // 2)
            emit({"tool_calls": [{"index": index, "function": {"arguments": arguments[:middle]}}]})
            emit({"tool_calls": [{"index": index, "function": {"arguments": arguments[middle:]}}]})
    lines.append("data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}))
    lines.append("data: [DONE]")
    lines.append("")
    return [line + "\n" for line in lines]


class UpstreamSession:
    """Record outbound calls without exposing a real inference service."""

    def __init__(self):
        self.calls = []
        self.close = AsyncMock()
        self.result = {"choices": [{"message": {"role": "assistant", "content": "result"}}]}
        self.results: list | None = None
        self.fail_times = 0

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise aiohttp.ClientConnectionError("stream dropped")
        if self.results:
            result = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        else:
            result = self.result

        class Stream:
            def __aiter__(self):
                async def generate():
                    for line in sse_lines(result):
                        yield line.encode()
                return generate()

            async def iter_chunked(self, size):
                # The sandbox model proxy reads a non-streamed body; keep that path working.
                yield json.dumps(result).encode()

            read = AsyncMock(return_value=json.dumps(result).encode())

        class Response:
            status = 200
            content = Stream()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return Response()


@asynccontextmanager
async def gateway_client(tmp_path, *, officer_only=True):
    members = {
        1: SimpleNamespace(bot=False, roles=[SimpleNamespace(id=100)]),
        2: SimpleNamespace(bot=False, roles=[]),
    }
    guild = SimpleNamespace(id=10, fetch_member=AsyncMock(side_effect=lambda user_id: members[user_id]))
    channel = SimpleNamespace(guild=guild, permissions_for=lambda member: SimpleNamespace(view_channel=True))
    bot = SimpleNamespace(get_guild=lambda guild_id: guild if guild_id == 10 else None,
                          fetch_channel=AsyncMock(return_value=channel))
    config = SimpleNamespace(agent=SimpleNamespace(search_base_url=""),
                             inference=SimpleNamespace(model="trusted-qwen", base_url="http://trusted-model:8000/v1"),
                             llama_cpp_api_key="trusted-inference-secret")
    settings = HermesSettings(
        allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
        owner_user_ids=frozenset({1}), runner_url="http://runner:8080",
        tool_service_url="http://gateway:8770", runner_token="r" * 40,
        state_dir=str(tmp_path), officer_only=officer_only,
        member_work_enabled=not officer_only,
    )
    gateway = HermesGateway(bot, config, settings)
    gateway.session = UpstreamSession()
    gateway.test_members = members
    app = web.Application()
    app.router.add_post("/tool", gateway.tool)
    app.router.add_post("/progress", gateway.progress)
    app.router.add_post("/package", gateway.package)
    app.router.add_post("/v1/chat/completions", gateway.model)
    app.router.add_get("/v1/models", gateway.models)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield gateway, client
    finally:
        await client.close()
        await gateway.close()
        gateway.jobs.close()


def capability(gateway, *, user_id=1, token="valid", status="running", deadline=None):
    job = gateway.jobs.create(guild_id=10, user_id=user_id, channel_id=20,
                              source_message_id=30, prompt="task")
    gateway.jobs.update(job["id"], status=status)
    cap = Capability(job, time.monotonic() + 60 if deadline is None else deadline)
    gateway.capabilities[token] = cap
    return cap


def headers(token="valid"):
    return {"Authorization": "Bearer " + token}


def test_capabilities_require_valid_token_running_job_and_unexpired_deadline(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            for auth in ({}, headers("unknown"), {"Authorization": "Basic valid"}):
                response = await client.get("/v1/models", headers=auth)
                assert response.status == 401
            response = await client.get("/v1/models", headers=headers())
            assert response.status == 200
            assert (await response.json())["data"][0]["id"] == "trusted-qwen"
            cap.deadline = time.monotonic() - 1
            assert (await client.get("/v1/models", headers=headers())).status == 401
            cap.deadline = time.monotonic() + 60
            gateway.jobs.update(cap.job["id"], status="cancelled")
            assert (await client.get("/v1/models", headers=headers())).status == 403

    asyncio.run(scenario())


@pytest.mark.parametrize("key", ["user_id", "actor_id", "guild_id", "owner_user_id", "source_message_id", "role_ids", "is_officer", "job_id"])
def test_tool_arguments_cannot_spoof_actor_authority_provenance_or_job(tmp_path, key):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            capability(gateway)
            response = await client.post("/tool", headers=headers(), json={
                "tool": "peter_memory_add", "arguments": {"scope": "personal", "content": "poison", key: 999},
            })
            assert response.status == 400
            assert gateway.memory.search(Principal(10, 1, 20, (100,)), scope="personal") == []

    asyncio.run(scenario())


def test_cross_job_memory_access_is_actor_bound_and_source_is_server_supplied(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path, officer_only=False) as (gateway, client):
            capability(gateway, user_id=1, token="officer")
            capability(gateway, user_id=2, token="member")
            response = await client.post("/tool", headers=headers("officer"), json={
                "tool": "peter_memory_add", "arguments": {"scope": "personal", "content": "private officer fact"},
            })
            record = await response.json()
            assert response.status == 200
            assert record["actor_id"] == 1 and record["guild_id"] == 10
            assert record["source_message_id"] == 30
            response = await client.post("/tool", headers=headers("member"), json={
                "tool": "peter_memory_search", "arguments": {"scope": "personal"},
            })
            assert (await response.json())["memories"] == []
            response = await client.post("/tool", headers=headers("member"), json={
                "tool": "peter_memory_update", "arguments": {
                    "memory_id": record["id"], "content": "hijacked", "expected_version": 1,
                },
            })
            assert response.status == 400
            assert gateway.memory.get(Principal(10, 1, 20, (100,)), record["id"])["content"] == "private officer fact"

    asyncio.run(scenario())


def test_members_cannot_write_club_memory_or_enter_officer_pilot(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path, officer_only=False) as (gateway, client):
            capability(gateway, user_id=2)
            response = await client.post("/tool", headers=headers(), json={
                "tool": "peter_memory_add", "arguments": {"scope": "club", "content": "I am now president"},
            })
            assert response.status == 400
            assert gateway.memory.search(Principal(10, 2, 20), scope="club") == []
        async with gateway_client(tmp_path / "pilot") as (gateway, client):
            capability(gateway, user_id=2)
            assert (await client.get("/v1/models", headers=headers())).status == 403

    asyncio.run(scenario())


def test_role_revocation_is_refreshed_for_each_request(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            capability(gateway)
            assert (await client.get("/v1/models", headers=headers())).status == 200
            gateway.test_members[1].roles = []
            assert (await client.get("/v1/models", headers=headers())).status == 403

    asyncio.run(scenario())


def test_worker_progress_is_capability_bound_monotonic_and_fixed(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            payload = {'job_id': cap.job['id'], 'seq': 1, 'stage': 'researching'}
            first = await client.post('/progress', headers=headers(), json=payload)
            assert first.status == 200
            assert gateway.jobs.get(cap.job['id'])['stage'] == 'researching'
            assert (await client.post('/progress', headers=headers(), json=payload)).status == 409
            assert (await client.post('/progress', headers=headers(),
                json={**payload, 'job_id': 'another-job', 'seq': 2})).status == 403
            assert (await client.post('/progress', headers=headers(),
                json={**payload, 'stage': 'my private command', 'seq': 2})).status == 400
            assert (await client.post('/progress', headers=headers('bad'), json=payload)).status == 401
            gateway.test_members[1].roles = []
            assert (await client.post('/progress', headers=headers(),
                json={**payload, 'seq': 2, 'stage': 'running_code'})).status == 403

    asyncio.run(scenario())


def test_model_uses_actual_token_receipt_and_remaining_task_deadline(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway, deadline=time.monotonic() + 65)
            gateway.session.result = {'choices': [{'message': {'content': 'Done'}}],
                                      'usage': {'prompt_tokens': 12, 'completion_tokens': 20}}
            body = {'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 100}
            response = await client.post('/v1/chat/completions', headers=headers(), json=body)
            assert response.status == 200
            assert cap.output_tokens == 20  # unused reservation is returned
            assert gateway.metrics.summary()['model']['input_tokens'] == 12
            assert gateway.metrics.summary()['model']['output_tokens'] == 20
            calls = len(gateway.session.calls)
            cap.deadline = time.monotonic() + 20
            response = await client.post('/v1/chat/completions', headers=headers(), json=body)
            assert response.status == 429
            assert len(gateway.session.calls) == calls  # no long call starts at the deadline
            cap.deadline = time.monotonic() + 4
            tool = await client.post('/tool', headers=headers(),
                json={'tool': 'calculate', 'arguments': {'expression': '2+2'}})
            assert tool.status == 429 and cap.tool_calls == 0

    asyncio.run(scenario())


def test_cancellation_revokes_only_owned_job_capability_and_calls_runner(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path, officer_only=False) as (gateway, client):
            owned = capability(gateway)
            other = capability(gateway, user_id=2, token="other")
            with pytest.raises(ValueError):
                await gateway.cancel(other.job["id"], 10, 1)
            assert "other" in gateway.capabilities
            await gateway.cancel(owned.job["id"], 10, 1)
            assert gateway.jobs.get(owned.job["id"])["status"] == "cancelled"
            assert "valid" not in gateway.capabilities
            assert "other" in gateway.capabilities
            assert (await client.get("/v1/models", headers=headers())).status == 401
            assert (await client.get("/v1/models", headers=headers("other"))).status == 200
            url, kwargs = gateway.session.calls[-1]
            assert url == "http://runner:8080/cancel"
            assert kwargs["json"] == {"job_id": owned.job["id"]}
            assert kwargs["headers"]["Authorization"] == "Bearer " + "r" * 40

    asyncio.run(scenario())


@pytest.mark.parametrize("revoke", ["cancel", "expire"])
def test_authorization_rechecks_after_discord_refresh_await(tmp_path, revoke):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            entered, release = asyncio.Event(), asyncio.Event()

            async def slow_principal(*args, **kwargs):
                entered.set()
                await release.wait()
                return Principal(10, 1, 20, (100,))

            gateway.principal = slow_principal
            pending = asyncio.create_task(client.post("/tool", headers=headers(), json={
                "tool": "peter_memory_add", "arguments": {"scope": "club", "content": "late write"},
            }))
            await entered.wait()
            if revoke == "cancel":
                await gateway.cancel(cap.job["id"], 10, 1)
            else:
                cap.deadline = time.monotonic() - 1
            release.set()
            response = await pending
            assert response.status in {401, 403}
            assert gateway.memory.search(Principal(10, 1, 20, (100,)), scope="club") == []

    asyncio.run(scenario())


def test_model_proxy_fixes_upstream_model_thinking_credentials_and_flags(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            messages = [{"role": "user", "content": "solve this"}]
            response = await client.post("/v1/chat/completions", headers=headers(), json={
                "messages": messages, "model": "attacker-model", "base_url": "http://attacker",
                "api_key": "attacker-secret", "headers": {"Authorization": "attacker-secret"},
                "stream": True, "n": 100, "parallel_tool_calls": True, "max_tokens": 999999,
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {"enable_thinking": False}, "temperature": 0.5,
            })
            assert response.status == 200
            assert (await response.json())["choices"][0]["message"]["content"] == "result"
            url, kwargs = gateway.session.calls[-1]
            assert url == "http://trusted-model:8000/v1/chat/completions"
            assert kwargs["allow_redirects"] is False
            assert kwargs["headers"] == {"Authorization": "Bearer trusted-inference-secret"}
            assert kwargs["json"] == {
                "messages": messages, "temperature": 0.5, "model": "trusted-qwen",
                "stream": False, "n": 1, "parallel_tool_calls": False,
                "max_tokens": gateway.settings.max_tokens,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            assert cap.model_calls == 1
            assert cap.output_tokens == gateway.settings.max_tokens

    asyncio.run(scenario())


def test_concurrent_model_calls_for_one_task_wait_instead_of_failing(tmp_path):
    """A client retry that races the tail of the previous call must be serialized, not
    rejected with a 429 that kills the task."""

    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            capability(gateway)
            body = {"messages": [{"role": "user", "content": "solve this"}]}
            first, second = await asyncio.gather(
                client.post("/v1/chat/completions", headers=headers(), json=body),
                client.post("/v1/chat/completions", headers=headers(), json=body),
            )
            assert (first.status, second.status) == (200, 200)
            # Both reached the upstream, one after the other.
            assert len(gateway.session.calls) == 2
            assert not gateway.capabilities["valid"].lock.locked()

    asyncio.run(scenario())


def test_a_wedged_model_lock_still_refuses_rather_than_queueing_forever(tmp_path, monkeypatch):
    import peterbot.hermes_gateway as gateway_module

    monkeypatch.setattr(gateway_module, "MODEL_LOCK_WAIT_SECONDS", 0.05)

    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            await cap.lock.acquire()  # a call that never returns
            try:
                response = await client.post("/v1/chat/completions", headers=headers(),
                                             json={"messages": [{"role": "user", "content": "hi"}]})
                assert response.status == 429
                assert gateway.session.calls == []
            finally:
                cap.lock.release()

    asyncio.run(scenario())


def test_model_wait_rechecks_cancellation_before_upstream_call(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            await cap.lock.acquire()
            pending = asyncio.create_task(client.post(
                "/v1/chat/completions", headers=headers(),
                json={"messages": [{"role": "user", "content": "hi"}]}))
            try:
                await asyncio.sleep(0.03)
                gateway.jobs.update(cap.job["id"], status="cancelled")
            finally:
                cap.lock.release()
            response = await pending
            assert response.status == 403
            assert gateway.session.calls == []
    asyncio.run(scenario())


def test_model_and_tool_budgets_stop_dispatch_before_upstream_calls(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            cap.model_calls = gateway.settings.max_model_calls
            response = await client.post("/v1/chat/completions", headers=headers(), json={"messages": []})
            assert response.status == 429
            assert gateway.session.calls == []
            cap.tool_calls = gateway.settings.max_tool_calls
            gateway.tools.execute = AsyncMock(return_value="{}")
            response = await client.post("/tool", headers=headers(), json={"tool": "calculate", "arguments": {"expression": "1+1"}})
            assert response.status == 429
            gateway.tools.execute.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("endpoint", ["/tool", "/v1/chat/completions"])
def test_cancel_during_streamed_request_body_prevents_dispatch(tmp_path, endpoint):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cap = capability(gateway)
            entered, release = asyncio.Event(), asyncio.Event()
            original_json = web.Request.json

            async def observed_json(request, *args, **kwargs):
                entered.set()
                return await original_json(request, *args, **kwargs)

            body = ({"tool": "peter_memory_add", "arguments": {"scope": "club", "content": "late write"}}
                    if endpoint == "/tool" else {"messages": [{"role": "user", "content": "late model call"}]})
            encoded = json.dumps(body).encode()

            async def delayed_body():
                yield encoded[:1]
                # Cancel once the server awaits the incomplete JSON body. This
                # works whether authentication precedes or follows parsing.
                await entered.wait()
                await gateway.cancel(cap.job["id"], 10, 1)
                release.set()
                yield encoded[1:]

            with patch.object(web.Request, "json", observed_json):
                response = await client.post(endpoint, headers={**headers(), "Content-Type": "application/json"}, data=delayed_body())
            assert release.is_set()
            assert response.status in {401, 403}
            assert gateway.memory.search(Principal(10, 1, 20, (100,)), scope="club") == []
            assert all(not url.endswith("/chat/completions") for url, _ in gateway.session.calls)

    asyncio.run(scenario())


class _TextChannelStub(discord.TextChannel):
    """Minimal stand-in for the server text channel that passes isinstance checks."""


def _discord_error(cls=discord.HTTPException, message="transient Discord failure", status=429):
    error = cls.__new__(cls)
    Exception.__init__(error, message)
    error.status = status
    return error


class TaskChannel:
    """Task thread mock: holds the first send, injects failures, keeps receipts."""

    def __init__(self, guild, channel_id=21):
        self.id = channel_id
        self.guild = guild
        self.sent = []
        self.messages = []
        self.files = []
        self.edits = []
        self.released = asyncio.Event()
        self.entered = asyncio.Event()
        self.fail_on_send = None
        self.fail_status = 429
        self.forbidden_on_send = None
        self._sends = 0

    async def send(self, content=None, *, file=None, **kwargs):
        self._sends += 1
        if self._sends == 1:
            self.entered.set()
            await self.released.wait()
        if self.fail_on_send == self._sends:
            raise _discord_error(status=self.fail_status)
        if self.forbidden_on_send == self._sends:
            raise _discord_error(discord.Forbidden, "missing access")
        if file is not None:
            self.files.append(file.filename)
        else:
            self.sent.append(content)
        receipt = SimpleNamespace(id=900 + self._sends, edit=AsyncMock())
        self.messages.append(receipt)
        return receipt

    def permissions_for(self, member):
        return SimpleNamespace(view_channel=True)

    async def edit(self, **kwargs):
        self.edits.append(kwargs)

    async def add_user(self, member):
        pass


class ControlledRunner:
    """/run responses wait for an explicit release; /cancel always succeeds."""

    def __init__(self):
        self.result = {"status": "completed", "answer": "late completion", "artifacts": []}
        self.release = asyncio.Event()
        self.close = AsyncMock()

    def post(self, url, **kwargs):
        payload = json.dumps(self.result).encode()
        release = None if url.endswith("/cancel") else self.release

        async def chunks(_size):
            yield payload

        class Response:
            status = 200
            content = SimpleNamespace(iter_chunked=chunks)

            async def __aenter__(self):
                if release is not None:
                    await release.wait()
                return self

            async def __aexit__(self, *args):
                return False

        return Response()


@asynccontextmanager
async def task_gateway(tmp_path, *, officer_only=False):
    members = {
        1: SimpleNamespace(bot=False, roles=[SimpleNamespace(id=100)]),
        2: SimpleNamespace(bot=False, roles=[]),
    }
    guild = SimpleNamespace(id=10, fetch_member=AsyncMock(side_effect=lambda user_id: members[user_id]))
    thread = TaskChannel(guild)
    source = _TextChannelStub.__new__(_TextChannelStub)
    source.id = 20
    source.guild = guild
    source.threads_created = 0
    source.permissions_for = lambda member: SimpleNamespace(view_channel=True)

    async def create_thread(**kwargs):
        source.threads_created += 1
        return thread

    source.create_thread = create_thread

    async def fetch_channel(channel_id):
        return thread if channel_id == thread.id else source

    bot = SimpleNamespace(get_guild=lambda guild_id: guild if guild_id == 10 else None,
                          fetch_channel=fetch_channel)
    config = SimpleNamespace(agent=SimpleNamespace(search_base_url=""),
                             inference=SimpleNamespace(model="trusted-qwen", base_url="http://trusted-model:8000/v1"),
                             llama_cpp_api_key="", peter_system_prompt="persona", peter_name="Peter")
    settings = HermesSettings(
        allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
        owner_user_ids=frozenset(), runner_url="http://runner:8080",
        tool_service_url="http://gateway:8770", runner_token="r" * 40,
        state_dir=str(tmp_path), officer_only=officer_only,
    )
    gateway = HermesGateway(bot, config, settings)
    gateway.session = UpstreamSession()
    gateway.test_members = members
    gateway.thread = thread
    gateway.source = source
    try:
        yield gateway
    finally:
        await gateway.close()
        gateway.jobs.close()


def test_submission_send_race_delivers_final_answer_once(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            thread = gateway.thread
            submitting = asyncio.create_task(gateway.submit(
                guild_id=10, user_id=1, channel=gateway.source,
                source_message_id=30, prompt="Research hardware options"))
            await asyncio.wait_for(thread.entered.wait(), 5)  # the acknowledgement send is held open

            for _ in range(3):
                await gateway.queue_tick()  # the queue polls during the open send
            mid = gateway.jobs.list_owned(10, 1)[0]
            assert mid["status"] == "preparing"
            assert gateway.jobs.pending() == []
            assert gateway.jobs.undelivered() == []
            assert gateway.active == {}
            assert thread.sent == [] and thread.files == []

            # A replayed Discord event for the same message cannot create a
            # second thread or job, and it returns the original even though
            # the submission cooldown is active (duplicate lookup precedes it).
            replay = await gateway.submit(guild_id=10, user_id=1, channel=gateway.source,
                                          source_message_id=30, prompt="Research hardware options")
            assert replay["id"] == mid["id"]
            assert gateway.source.threads_created == 1

            thread.released.set()
            job = await asyncio.wait_for(submitting, 5)
            assert job["status"] == "queued"

            gateway.session.result = {"status": "completed", "answer": "the final answer", "artifacts": []}
            await gateway.queue_tick()
            while gateway.active:
                await asyncio.sleep(0.01)
            await gateway.queue_tick()  # delivers the finished answer

            assert thread.sent.count("the final answer") == 1
            assert "(No response)" not in thread.sent
            stored = gateway.jobs.get(job["id"])
            assert stored["status"] == "completed"
            assert stored["delivered"] == 1
            assert stored["delivery_status"] == "delivered"

            await gateway.queue_tick()  # later polls never replay it
            assert thread.sent.count("the final answer") == 1
            assert gateway.jobs.undelivered() == []

            # A late replay of the same message returns the completed job.
            replayed = await gateway.submit(guild_id=10, user_id=1, channel=gateway.source,
                                            source_message_id=30, prompt="Research hardware options")
            assert replayed["id"] == job["id"]
            assert gateway.source.threads_created == 1

    asyncio.run(scenario())


def test_private_task_status_message_becomes_final_answer(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = await gateway.submit(guild_id=10, user_id=1, channel=gateway.source,
                source_message_id=333, prompt='Build a small CLI')
            status = gateway.thread.messages[0]
            assert gateway.jobs.get(job['id'])['status_message_id'] == status.id
            gateway.thread.fetch_message = AsyncMock(return_value=status)
            gateway.session.result = {'status': 'completed', 'answer': 'Built and tested.',
                                      'artifacts': []}
            await gateway.queue_tick()
            while gateway.active:
                await asyncio.sleep(0.01)
            await gateway.queue_tick()
            status.edit.assert_awaited()
            assert status.edit.await_args.kwargs['content'] == 'Built and tested.'
            assert 'Built and tested.' not in gateway.thread.sent
            assert gateway.jobs.get(job['id'])['delivery_status'] == 'delivered'

    asyncio.run(scenario())


def test_private_followup_edits_its_status_instead_of_leaving_a_stale_notice(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            first = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                source_message_id=330, prompt='Make counter.py')
            gateway.jobs.update(first['id'], status='completed', answer='Printed 42.',
                                delivered=True)
            followup = await gateway.submit(guild_id=10, user_id=1,
                channel=gateway.thread, source_message_id=331,
                prompt='Change it to 43', parent_id=first['id'])
            assert followup['status_message_id'] is None
            gateway.thread.fetch_message = AsyncMock(
                side_effect=lambda _id: gateway.thread.messages[0])
            gateway.session.result = {'status': 'completed', 'answer': 'Updated counter.py.',
                                      'artifacts': []}
            await gateway.queue_tick()
            while gateway.active:
                await asyncio.sleep(0.01)
            await gateway.queue_tick()
            notice = gateway.thread.messages[0]
            assert gateway.jobs.get(followup['id'])['status_message_id'] == notice.id
            assert notice.edit.await_args.kwargs['content'] == 'Updated counter.py.'
            assert 'Updated counter.py.' not in gateway.thread.sent
            assert gateway.jobs.get(followup['id'])['delivery_status'] == 'delivered'
    asyncio.run(scenario())


def test_members_can_chat_before_work_execution_rollout_but_cannot_submit(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path, officer_only=False) as gateway:
            assert await gateway.eligible(10, 2, gateway.source.id)
            with pytest.raises(PolicyDenied, match='not available to members'):
                await gateway.submit(guild_id=10, user_id=2, channel=gateway.source,
                                     source_message_id=301, prompt='Run code')
            assert gateway.source.threads_created == 0
            assert gateway.jobs.pending() == []

    asyncio.run(scenario())


def test_stale_pending_snapshot_cannot_double_start(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="solo task")
            stale = dict(job)  # the snapshot a poll would have raced against
            assert gateway.jobs.claim(job["id"]) is True
            started = []

            async def fake_run(target):
                started.append(target["id"])

            gateway.jobs.pending = lambda: [stale]
            gateway.run_job = fake_run
            await gateway.queue_tick()
            assert started == []
            assert gateway.active == {}

    asyncio.run(scenario())


def test_late_completion_cannot_overwrite_a_cancellation(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="long task")
            runner = ControlledRunner()
            gateway.session = runner
            await gateway.queue_tick()
            assert gateway.jobs.get(job["id"])["status"] == "running"
            await gateway.cancel(job["id"], 10, 1)
            assert gateway.jobs.get(job["id"])["status"] == "cancelled"
            assert gateway.capabilities == {}
            runner.release.set()
            while gateway.active:
                await asyncio.sleep(0.01)
            after = gateway.jobs.get(job["id"])
            assert after["status"] == "cancelled"
            assert after["answer"] == "Task cancelled."
            await gateway.queue_tick()
            assert gateway.thread.sent.count("Task cancelled.") == 1
            assert "late completion" not in gateway.thread.sent
            assert gateway.jobs.get(job["id"])["delivered"] == 1

    asyncio.run(scenario())


def test_shutdown_marks_live_execution_interrupted(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="running at shutdown")
            runner = ControlledRunner()
            gateway.session = runner
            await gateway.queue_tick()
            await asyncio.sleep(0)
            assert gateway.active and gateway.capabilities
            await gateway.close()
            assert gateway.active == {} and gateway.capabilities == {}
            return job["id"]

    job_id = asyncio.run(scenario())
    reopened = JobStore(str(tmp_path / "tasks.sqlite3"))
    try:
        after = reopened.get(job_id)
        assert after["status"] == "interrupted"
        assert "restarted" in after["answer"]
        assert [row["id"] for row in reopened.undelivered()] == [job_id]
    finally:
        reopened.close()


def test_revoked_user_never_receives_the_delayed_result(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path, officer_only=True) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="secret work")
            gateway.jobs.update(job["id"], status="completed", answer="private result")
            gateway.test_members[1].roles = []  # authority revoked before delivery
            await gateway.queue_tick()
            assert gateway.thread.sent == [] and gateway.thread.files == []
            after = gateway.jobs.get(job["id"])
            assert after["delivered"] == 1
            assert after["delivery_status"] == "withheld"
            assert after["answer"] == "private result"
            assert after["status"] == "completed"
            assert gateway.jobs.undelivered() == []

    asyncio.run(scenario())


def test_partial_multi_message_delivery_resumes_from_the_cursor(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            thread = gateway.thread
            thread.released.set()
            answer = "x " * 901  # splits into two chunks over 1,800 characters
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="long report")
            gateway.jobs.update(job["id"], status="completed", answer=answer)
            thread.fail_on_send = 2  # second chunk hits a transient Discord failure
            await gateway.queue_tick()
            mid = gateway.jobs.get(job["id"])
            assert mid["delivered"] == 0
            assert mid["delivery_status"] == "pending"
            assert mid["delivery_cursor"] == 1
            assert mid["delivery_attempts"] == 1
            assert len(thread.sent) == 1
            assert [row["id"] for row in gateway.jobs.undelivered()] == [job["id"]]

            thread.fail_on_send = None
            await gateway.queue_tick()
            after = gateway.jobs.get(job["id"])
            assert after["delivered"] == 1 and after["delivery_cursor"] == 2
            assert len(thread.sent) == 2  # the acknowledged chunk was not replayed
            assert " ".join(thread.sent).split() == answer.split()
            assert json.loads(after["delivery_receipts"]) == ["901", "903"]

    asyncio.run(scenario())


def test_forbidden_channel_send_withholds_the_result(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="private work")
            gateway.jobs.update(job["id"], status="completed", answer="private result")
            gateway.thread.forbidden_on_send = 1
            await gateway.queue_tick()
            after = gateway.jobs.get(job["id"])
            assert after["delivery_status"] == "withheld"
            assert after["delivered"] == 1
            assert gateway.thread.sent == []
            assert gateway.jobs.undelivered() == []

    asyncio.run(scenario())


def test_discord_server_error_has_unknown_send_outcome(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="private work")
            gateway.jobs.update(job["id"], status="completed", answer="private result")
            gateway.thread.fail_on_send = 1
            gateway.thread.fail_status = 500
            await gateway.queue_tick()
            assert gateway.jobs.get(job["id"])["delivery_status"] == "unknown"
            assert gateway.jobs.undelivered() == []
            await gateway.queue_tick()
            assert gateway.thread._sends == 1

    asyncio.run(scenario())


def test_cancellation_during_acknowledgement_send_resolves_submission(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            thread = gateway.thread
            submitting = asyncio.create_task(gateway.submit(
                guild_id=10, user_id=1, channel=gateway.source,
                source_message_id=30, prompt="Research hardware options"))
            await asyncio.wait_for(thread.entered.wait(), 5)
            submitting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submitting
            mid = gateway.jobs.get(gateway.jobs.list_owned(10, 1)[0]["id"])
            assert mid["status"] == "failed"  # never left dangling in `preparing`
            assert "interrupted" in mid["answer"]
            assert mid["delivered"] == 1
            assert thread.edits and thread.edits[-1].get("archived") is True

            # A re-delivered event cannot open a second thread on the old job.
            replay = await gateway.submit(guild_id=10, user_id=1, channel=gateway.source,
                                          source_message_id=30, prompt="Research hardware options")
            assert replay["id"] == mid["id"]
            assert gateway.source.threads_created == 1

    asyncio.run(scenario())


def test_acknowledgement_is_corrected_if_admission_fails_after_send(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            original = gateway.jobs.transition
            gateway.jobs.transition = lambda job_id, to, **kwargs: False if to == "queued" else original(job_id, to, **kwargs)
            with pytest.raises(RuntimeError, match="admission"):
                await gateway.submit(guild_id=10, user_id=1, channel=gateway.source,
                                     source_message_id=30, prompt="Research hardware options")
            gateway.thread.messages[0].edit.assert_awaited_once()
            assert "could not start" in gateway.thread.messages[0].edit.await_args.kwargs["content"]
            assert gateway.thread.edits[-1]["archived"] is True
            job = gateway.jobs.list_owned(10, 1)[0]
            assert job["status"] == "failed"

    asyncio.run(scenario())


def test_ambiguous_delivery_failure_freezes_for_reconciliation(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt="fragile result")
            gateway.jobs.update(job["id"], status="completed", answer="private result")
            working = gateway.bot.fetch_channel

            async def explode(channel_id):
                raise RuntimeError("unexpected internal failure")

            gateway.bot.fetch_channel = explode
            await gateway.queue_tick()
            frozen = gateway.jobs.get(job["id"])
            assert frozen["delivery_status"] == "unknown"
            assert frozen["delivered"] == 0
            assert frozen["answer"] == "private result"
            assert gateway.thread.sent == []
            assert gateway.jobs.undelivered() == []  # no blind resend
            await gateway.queue_tick()  # the freeze survives further polls
            assert gateway.jobs.get(job["id"])["delivery_status"] == "unknown"

            gateway.bot.fetch_channel = working
            assert gateway.jobs.reconcile_unknown_delivery(job["id"], retry=True) is True
            await gateway.queue_tick()
            assert gateway.thread.sent == ["private result"]
            assert gateway.jobs.get(job["id"])["delivery_status"] == "delivered"

    asyncio.run(scenario())


WHEEL_BODY = b"PK\x03\x04fake wheel"
WHEEL_DIGEST = hashlib.sha256(WHEEL_BODY).hexdigest()

def image_cache(tmp_path, monkeypatch):
    """Synthetic image cache holding the pinned six wheel under the task's control."""
    cache = tmp_path / "deps"
    (cache / "wheels").mkdir(parents=True)
    entry = package_access.image_lookup("pypi", "six", "1.17.0")
    (cache / "wheels" / entry["filename"]).write_bytes(WHEEL_BODY)
    monkeypatch.setattr(package_access, "PACKAGE_INVENTORY",
                        (dict(entry, sha256=WHEEL_DIGEST, size=len(WHEEL_BODY)),))
    return cache, entry

def test_package_route_serves_hash_verified_image_cache_bytes(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cache, entry = image_cache(tmp_path, monkeypatch)
            gateway.packages = PackageBroker(cache)
            cap = capability(gateway)
            response = await client.post("/package", headers=headers(), json={
                "registry": "pypi", "name": "six", "version": "1.17.0"})
            assert response.status == 200
            assert await response.read() == WHEEL_BODY
            assert response.headers["X-Peterbot-Sha256"] == WHEEL_DIGEST
            assert response.headers["X-Peterbot-Filename"] == entry["filename"]
            assert response.headers["X-Peterbot-Source"] == "image_cache"
            assert int(response.headers["X-Peterbot-Size"]) == len(WHEEL_BODY)
            # Bytes land in the task quota, not in any model-visible payload.
            assert cap.package_bytes == len(WHEEL_BODY)

    asyncio.run(scenario())


def test_package_route_accumulates_quota_and_refuses_over_limit(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cache, _entry = image_cache(tmp_path, monkeypatch)
            gateway.packages = PackageBroker(cache)
            cap = capability(gateway)
            cap.package_bytes = PACKAGE_TASK_BYTES  # prior fetches already spent the quota
            response = await client.post("/package", headers=headers(), json={
                "registry": "pypi", "name": "six", "version": "1.17.0"})
            assert response.status == 429
            assert "quota" in (await response.text()).lower()

    asyncio.run(scenario())


def test_package_route_refuses_when_deadline_is_too_close(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cache, _entry = image_cache(tmp_path, monkeypatch)
            gateway.packages = PackageBroker(cache)
            cap = capability(gateway, deadline=time.monotonic() + 5)
            started = time.monotonic()
            response = await client.post("/package", headers=headers(), json={
                "registry": "pypi", "name": "six", "version": "1.17.0"})
            assert response.status == 429
            assert time.monotonic() - started < 1  # refused without attempting a fetch

    asyncio.run(scenario())


def test_package_route_rejects_malformed_body_and_unknown_names(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            cache, _entry = image_cache(tmp_path, monkeypatch)
            gateway.packages = PackageBroker(cache)
            capability(gateway)
            for body in ({"registry": "pypi", "name": "six"},
                         {"registry": "evil", "name": "six", "version": "1.17.0"},
                         {"registry": "pypi", "name": "../../etc", "version": "1.17.0"},
                         {"registry": "pypi", "name": "six", "version": "1.0; rm -rf /"}):
                response = await client.post("/package", headers=headers(), json=body)
                assert response.status == 400, body
                assert (await response.json())["code"] in {
                    "invalid_request", "invalid_registry", "invalid_name", "invalid_version"}

    asyncio.run(scenario())


def test_package_route_maps_unavailable_provider_to_503(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            capability(gateway)

            async def down(args, quota=PACKAGE_TASK_BYTES):
                raise PackageError("provider_unavailable", "registry down", 503)

            monkeypatch.setattr(gateway.packages, "serve", down)
            response = await client.post("/package", headers=headers(), json={
                "registry": "pypi", "name": "unpinned-project", "version": "2.0.0"})
            assert response.status == 503
            assert (await response.json())["code"] == "provider_unavailable"

    asyncio.run(scenario())


def test_package_route_requires_authenticated_capability(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            for auth in ({}, headers("unknown")):
                response = await client.post("/package", headers=auth, json={
                    "registry": "pypi", "name": "six", "version": "1.17.0"})
                assert response.status == 401
            capability(gateway, status="completed")
            response = await client.post("/package", headers=headers(), json={
                "registry": "pypi", "name": "six", "version": "1.17.0"})
            assert response.status == 403  # finished jobs get no broker access

    asyncio.run(scenario())
