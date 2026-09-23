"""Operator diagnostics distinguish dependency failures without leaking state."""
import asyncio
import json
from types import SimpleNamespace

import aiohttp
from aiohttp import web
import pytest

from test_hermes_gateway import gateway_client


class ProbeResponse:
    def __init__(self, status: int, body: dict | None = None):
        self.status = status
        payload = json.dumps(body or {}).encode()

        class Stream:
            async def iter_chunked(self, _size):
                yield payload

        self.content = Stream()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def test_diagnostics_require_operator_token_and_report_each_boundary(tmp_path, monkeypatch):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _client):
            with pytest.raises(web.HTTPUnauthorized):
                await gateway.diagnostics(SimpleNamespace(headers={}))
            with pytest.raises(web.HTTPUnauthorized):
                await gateway.diagnostics(SimpleNamespace(headers={
                    'Authorization': 'Bearer wrong'}))

            calls = []

            def get(url, **kwargs):
                calls.append((url, kwargs))
                if url.endswith('/health'):
                    return ProbeResponse(200)
                return ProbeResponse(200, {'data': [{'id': 'trusted-qwen'}]})

            gateway.session.get = get
            gateway.bot.is_ready = lambda: True
            gateway.foreground = SimpleNamespace(
                instance='one', lease_holder=lambda: 'one', counts=lambda: {'queued': 0})
            gateway.loop_task = asyncio.create_task(asyncio.sleep(60))
            monkeypatch.setenv('PETERBOT_REVISION', 'abcdef0123456789')
            response = await gateway.diagnostics(SimpleNamespace(headers={
                'Authorization': 'Bearer ' + gateway.settings.runner_token}))
            data = json.loads(response.text)
            assert data == {
                'status': 'ok', 'revision': 'abcdef0123456789',
                'dependencies': {'discord': 'ready', 'runner': 'ready',
                                 'model': 'ready', 'queue': 'ready'},
                'foreground': {'queued': 0},
            }
            assert len(calls) == 2
            assert all(call[1]['allow_redirects'] is False for call in calls)
            assert calls[1][1]['headers']['Authorization'] == 'Bearer trusted-inference-secret'
    asyncio.run(scenario())


def test_diagnostics_distinguish_runner_model_and_queue_failures(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, _client):
            def get(url, **_kwargs):
                if url.endswith('/health'):
                    return ProbeResponse(503)
                raise aiohttp.ClientConnectionError('model offline')

            gateway.session.get = get
            response = await gateway.diagnostics(SimpleNamespace(headers={
                'Authorization': 'Bearer ' + gateway.settings.runner_token}))
            data = json.loads(response.text)
            assert data['status'] == 'degraded'
            assert data['dependencies'] == {
                'discord': 'disconnected', 'runner': 'degraded',
                'model': 'unavailable', 'queue': 'stopped'}
            assert data['revision'] == 'unknown'
            assert 'model offline' not in response.text

            gateway.session.get = lambda url, **_kwargs: (
                ProbeResponse(200) if url.endswith('/health') else
                ProbeResponse(200, {'data': [{'id': 'another-model'}]}))
            wrong = json.loads((await gateway.diagnostics(SimpleNamespace(headers={
                'Authorization': 'Bearer ' + gateway.settings.runner_token}))).text)
            assert wrong['dependencies']['model'] == 'wrong_model'
    asyncio.run(scenario())
