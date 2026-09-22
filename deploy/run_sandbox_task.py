"""Run one real sandbox task through the runner and print the raw outcome.

Debugging tool: when a member-facing task fails, this reproduces it without Discord and
prints what the gateway actually stored, including artifacts and timing.

    docker run --rm --network peterbot_control --user 10000:10000 \
      --cap-drop ALL --security-opt no-new-privileges:true \
      --tmpfs /tmp:rw,nosuid,nodev,size=64m \
      --env-file <runner-token-env> \
      -e PETERBOT_CONFIG_FILE=/app/config.json -e PETERBOT_HERMES_CONFIG=/app/hermes.json \
      -e PETERBOT_SMOKE_STATE=/tmp/smoke-state \
      -e PETERBOT_SMOKE_PROMPT='write a hyper-optimized rust program ...' \
      -v <appdata>/config.production.json:/app/config.json:ro \
      -v <appdata>/hermes.production.json:/app/hermes.json:ro \
      --entrypoint python <gateway-image> /app/run_sandbox_task.py

Needs the worker network and the fixed gateway address, so stop the production gateway
first (it owns 192.168.240.2), exactly like deploy/smoke_hermes.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import replace

import aiohttp
from aiohttp import web

from peterbot.agent_policy import Principal
from peterbot.config import AppConfig
from peterbot.hermes_gateway import HermesGateway
from peterbot.hermes_settings import HermesSettings

DEFAULT_PROMPT = 'write a hyper-optimized rust program to calculate e to n digits'


async def main():
    os.environ.setdefault('DISCORD_TOKEN', 'sandbox-task-smoke-unused-no-discord-connection')
    prompt = os.getenv('PETERBOT_SMOKE_PROMPT', DEFAULT_PROMPT)
    conversational = os.getenv('PETERBOT_SMOKE_MODE', 'conversation')
    config = AppConfig.load()
    settings = HermesSettings.load(os.environ['PETERBOT_HERMES_CONFIG'])
    settings = replace(settings, state_dir=os.environ['PETERBOT_SMOKE_STATE'])
    guild_id = next(iter(settings.allowed_guild_ids))
    user_id, channel_id = 123456789012345678, 123456789012345679
    role_id = next(iter(settings.officer_role_ids))

    class Channel:
        async def send(self, *args, **kwargs):
            if args and isinstance(args[0], str):
                print(json.dumps({'channel_send': args[0][:200]}), flush=True)

    class Bot:
        async def fetch_channel(self, *args):
            return Channel()

    gateway = HermesGateway(Bot(), config, settings)

    async def principal(g, u, c, **kwargs):
        assert (g, u, c) == (guild_id, user_id, channel_id)
        return Principal(g, u, c, (role_id,))

    gateway.principal = principal
    gateway.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=settings.job_timeout + 60),
                                            trust_env=False)
    app = web.Application(client_max_size=2 * 1024 * 1024)
    app.router.add_post('/tool', gateway.tool)

    async def traced_model(request):
        started = time.monotonic()
        try:
            response = await gateway.model(request)
        except web.HTTPException as exc:
            print(json.dumps({'model_http_status': exc.status, 'seconds': round(time.monotonic() - started, 1)}),
                  flush=True)
            raise
        data = json.loads(response.body)
        message = (data.get('choices') or [{}])[0].get('message', {})
        print(json.dumps({'model_http_status': response.status,
                          'seconds': round(time.monotonic() - started, 1),
                          'reasoning_chars': len(message.get('reasoning') or ''),
                          'content_chars': len(message.get('content') or ''),
                          'tool_calls': [c.get('function', {}).get('name') for c in message.get('tool_calls') or []]}),
              flush=True)
        return response

    app.router.add_post('/v1/chat/completions', traced_model)
    app.router.add_get('/v1/models', gateway.models)
    server = web.AppRunner(app, access_log=None)
    await server.setup()
    await web.TCPSite(server, '0.0.0.0', 8770).start()
    started = time.monotonic()
    try:
        job = gateway.jobs.create(guild_id=guild_id, user_id=user_id, channel_id=channel_id,
                                  source_message_id=123456789012345680, prompt=prompt,
                                  delivery_mode='channel' if conversational == 'conversation' else 'private')
        gateway.jobs.update(job['id'], status='running')
        await gateway.run_job(gateway.jobs.get(job['id']))
        result = gateway.jobs.get(job['id'])
        print(json.dumps({'job_id': result['id'], 'status': result['status'],
                          'seconds': round(time.monotonic() - started, 1),
                          'answer': result['answer'],
                          'artifacts': [a.get('name') for a in json.loads(result['artifacts'])]}, indent=2),
              flush=True)
        raise SystemExit(0 if result['status'] == 'completed' else 1)
    finally:
        await server.cleanup()
        await gateway.session.close()
        await gateway.tools.close()


asyncio.run(main())
