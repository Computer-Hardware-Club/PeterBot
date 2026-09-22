import asyncio
from contextlib import asynccontextmanager
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import aiohttp
import pytest

from peterbot.agent_policy import Principal
from peterbot.hermes_gateway import Capability, HermesGateway
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
        self.results = None
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
                "chat_template_kwargs": {"enable_thinking": True},
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
