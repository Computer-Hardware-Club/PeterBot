import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from peterbot.announcements import draft_announcement
from peterbot.config import AgentConfig, ModelProfile
from peterbot.llama_cpp_client import LlamaCppChatClient
from peterbot.prompts import build_system_prompt, current_time_context


@pytest.mark.parametrize("utc_time,local_time", [
    ("2026-10-06T00:05:59+00:00", "2026-10-05 17:05-07:00 (America/Los_Angeles, PDT)"),
    ("2026-01-02T07:59:59+00:00", "2026-01-01 23:59-08:00 (America/Los_Angeles, PST)"),
    ("2026-01-02T08:00:00+00:00", "2026-01-02 00:00-08:00 (America/Los_Angeles, PST)"),
    ("2026-03-08T09:59:00+00:00", "2026-03-08 01:59-08:00 (America/Los_Angeles, PST)"),
    ("2026-03-08T10:00:00+00:00", "2026-03-08 03:00-07:00 (America/Los_Angeles, PDT)"),
    ("2026-11-01T08:59:00+00:00", "2026-11-01 01:59-07:00 (America/Los_Angeles, PDT)"),
    ("2026-11-01T09:00:00+00:00", "2026-11-01 01:00-08:00 (America/Los_Angeles, PST)"),
])
def test_clock_handles_date_boundaries_and_dst(utc_time, local_time):
    now = datetime.fromisoformat(utc_time)
    context = current_time_context(now)
    assert local_time in context
    assert f"UTC: {now.isoformat(sep=' ', timespec='minutes')} (UTC)" in context
    assert "ignore stale conversation" in context
    assert "refer to when that message was sent" in context


def test_clock_rejects_naive_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        current_time_context(datetime(2026, 1, 1))


def test_clock_reads_runtime_utc_each_time(monkeypatch):
    instants = [datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc),
                datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc)]
    clock = Mock()
    clock.now.side_effect = instants
    monkeypatch.setattr("peterbot.prompts.datetime", clock)
    assert "UTC: 2026-01-01 23:59+00:00" in current_time_context()
    assert "UTC: 2026-01-02 00:00+00:00" in current_time_context()
    assert clock.now.call_count == 2
    clock.now.assert_called_with(timezone.utc)


def test_shared_system_prompt_includes_clock(monkeypatch):
    monkeypatch.setattr("peterbot.prompts.current_time_context", lambda: "TRUSTED CLOCK")
    config = SimpleNamespace(peter_system_prompt="You are Peter.", peter_name="Peter",
                             model_profile=ModelProfile.QWEN)
    assert "TRUSTED CLOCK" in build_system_prompt(config, "")


@pytest.mark.parametrize("tools_enabled", [False, True])
def test_legacy_chat_includes_clock_in_every_model_phase(monkeypatch, tools_enabled):
    monkeypatch.setattr("peterbot.llama_cpp_client.current_time_context", lambda: "TRUSTED CLOCK")
    config = SimpleNamespace(
        peter_system_prompt="You are Peter.", model_profile=ModelProfile.QWEN,
        agent=AgentConfig(enabled=tools_enabled),
        inference=SimpleNamespace(model="test-model", max_tokens=256, temperature=0.2,
                                  top_p=None, extra_request_body={}, base_url="http://model"),
    )
    payloads = []

    class Response:
        status = 200

        def __init__(self):
            self.content = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def iter_chunked(self, _size):
            yield json.dumps({"choices": [{"message": {"content": "The meeting is tomorrow."}}]}).encode()

    def post(_url, *, json, **_kwargs):
        payloads.append(json)
        return Response()

    async def scenario():
        client = LlamaCppChatClient(config)
        client.http_session = SimpleNamespace(post=post, closed=False, close=AsyncMock())
        try:
            return await client._run_chat(
                "When is the meeting?", system_prompt="You are Peter. Use the channel context.",
                author_name="Member", conversation_history=[], user_content=None,
                user_images=None, response_mode="chat", request_id="clock-test",
            )
        finally:
            await client.close()

    assert "tomorrow" in asyncio.run(scenario())
    assert len(payloads) == (2 if tools_enabled else 1)
    for payload in payloads:
        assert "TRUSTED CLOCK" in payload["messages"][0]["content"]


def test_announcement_draft_includes_clock(monkeypatch):
    monkeypatch.setattr("peterbot.announcements.current_time_context", lambda: "TRUSTED CLOCK")
    model = AsyncMock(return_value={"choices": [{"finish_reason": "stop", "message": {
        "content": "Join us tomorrow."}}]})
    monkeypatch.setattr("peterbot.announcements._post", model)
    config = SimpleNamespace(peter_system_prompt="You are Peter.", inference=SimpleNamespace(model="test-model"))
    asyncio.run(draft_announcement(object(), config, request="Invite people tomorrow", channel_name="general"))
    assert "TRUSTED CLOCK" in model.await_args.args[2]["messages"][0]["content"]
