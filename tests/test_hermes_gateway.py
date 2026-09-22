import asyncio
from contextlib import asynccontextmanager
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import discord
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from peterbot.agent_jobs import JobStore
from peterbot.agent_policy import Principal
from peterbot.hermes_gateway import Capability, HermesGateway
from peterbot.hermes_settings import HermesSettings


class UpstreamSession:
    """Record outbound calls without exposing a real inference service."""

    def __init__(self):
        self.calls = []
        self.close = AsyncMock()
        self.result = {"choices": [{"message": {"role": "assistant", "content": "result"}}]}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = self.result

        async def chunks(size):
            yield json.dumps(result).encode()

        class Response:
            status = 200
            content = SimpleNamespace(read=AsyncMock(return_value=json.dumps(result).encode()),
                                      iter_chunked=chunks)

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
    )
    gateway = HermesGateway(bot, config, settings)
    gateway.session = UpstreamSession()
    gateway.test_members = members
    app = web.Application()
    app.router.add_post("/tool", gateway.tool)
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
                "chat_template_kwargs": {"enable_thinking": False},
            }
            assert cap.model_calls == 1
            assert cap.output_tokens == gateway.settings.max_tokens

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
