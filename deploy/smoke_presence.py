"""Verify the presence layer end to end: a real sandbox task, a recording channel.

This drives the production code paths (``run_job`` and ``deliver``) with a real model,
a real runner and a real disposable worker, and only replaces Discord. It answers the
question the unit tests cannot: what does a member actually see while Peter works?

Expected transcript: one status message that is edited while the task runs and finally
holds the answer, with any artifacts arriving as their own messages.

Authorization is stubbed here (authority is covered by the gateway tests and the
isolation check); this smoke is about presence and delivery.
"""
import asyncio
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, '/app')

import aiohttp  # noqa: E402

from peterbot.agent_policy import Principal  # noqa: E402
from peterbot.hermes_gateway import HermesGateway  # noqa: E402
from peterbot.hermes_settings import HermesSettings  # noqa: E402
from peterbot.config import AppConfig  # noqa: E402

PROMPT = os.environ.get('PETERBOT_SMOKE_PROMPT',
                        'Create /workspace/artifacts/presence.txt containing the word alive, '
                        'then tell me in one short sentence what you wrote.')
START = time.monotonic()


def stamp():
    return round(time.monotonic() - START, 1)


class RecordingMessage:
    def __init__(self, message_id, content=''):
        self.id = message_id
        self.content = content
        self.history = []

    async def edit(self, *, content=None, **kwargs):
        self.content = content
        self.history.append((stamp(), content))
        print(json.dumps({'t': stamp(), 'event': 'edit', 'message': self.id,
                          'content': (content or '')[:400]}), flush=True)
        return self


class RecordingChannel:
    """Stands in for a Discord text channel, printing what a member would see."""

    def __init__(self):
        self.sent = []
        self.messages = {}
        self.next_id = 500

    async def send(self, content=None, **kwargs):
        message = RecordingMessage(self.next_id, content or '')
        self.next_id += 1
        self.messages[message.id] = message
        self.sent.append({'t': stamp(), 'id': message.id, 'content': content,
                          'file': getattr(kwargs.get('file'), 'filename', None)})
        print(json.dumps({'t': stamp(), 'event': 'send', 'message': message.id,
                          'file': getattr(kwargs.get('file'), 'filename', None),
                          'content': (content or '')[:400]}), flush=True)
        return message

    async def fetch_message(self, message_id):
        return self.messages[int(message_id)]


async def main():
    # AppConfig.load() validates the Discord token; this smoke never connects to Discord.
    os.environ.setdefault('DISCORD_TOKEN', 'presence-smoke-unused-no-discord-connection')
    settings = HermesSettings.load(os.environ['PETERBOT_HERMES_CONFIG'])
    settings = replace(settings, state_dir=os.environ['PETERBOT_SMOKE_STATE'])
    config = AppConfig.load()
    from aiohttp import web
    # The worker reaches the model and its tools through this process on :8770 (the smoke
    # container holds the `gateway` alias), so the task needs the real handlers served.
    # Without this the worker's calls fail and the run reports model_failed.
    # Use the real allowed guild and officer role, so policy checks behave as in production.
    guild_id = next(iter(settings.allowed_guild_ids))
    role_id = next(iter(settings.officer_role_ids))
    user_id, channel_id = 123456789012345678, 123456789012345679
    channel = RecordingChannel()
    guild = type('Guild', (), {'id': guild_id})()
    channel.guild = guild
    bot = type('Bot', (), {'user': type('U', (), {'id': 999})(),
                           'get_guild': lambda gid: guild,
                           'is_ready': lambda: True,
                           'fetch_channel': staticmethod(lambda cid: asyncio.sleep(0, result=channel))})()

    gateway = HermesGateway(bot, config, settings)

    async def principal(guild_id, user_id, channel_id, *args, **kwargs):
        return Principal(guild_id, user_id, channel_id, (role_id,))
    gateway.principal = principal
    gateway.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240), trust_env=False)

    async def traced_model(request):
        started_call = time.monotonic()
        try:
            response = await gateway.model(request)
        except web.HTTPException as exc:
            print(json.dumps({'t': stamp(), 'model_http_status': exc.status,
                              'seconds': round(time.monotonic() - started_call, 1)}), flush=True)
            raise
        data = json.loads(response.body)
        message = (data.get('choices') or [{}])[0].get('message', {})
        print(json.dumps({'t': stamp(), 'model_http_status': response.status,
                          'seconds': round(time.monotonic() - started_call, 1),
                          'reasoning_chars': len(message.get('reasoning') or ''),
                          'tool_calls': [c.get('function', {}).get('name')
                                         for c in message.get('tool_calls') or []]}), flush=True)
        return response

    app = web.Application(client_max_size=2 * 1024 * 1024)
    app.router.add_post('/tool', gateway.tool)
    app.router.add_post('/v1/chat/completions', traced_model)
    app.router.add_get('/v1/models', gateway.models)
    server = web.AppRunner(app, access_log=None)
    await server.setup()
    await web.TCPSite(server, '0.0.0.0', 8770).start()

    # The status line the member is already watching, exactly as respond_to_message posts it.
    status = await channel.send("on it — this needs real work, so give me a bit. I'll post the result here.")

    job = gateway.jobs.create(guild_id=guild_id, user_id=user_id, channel_id=channel_id, source_message_id=30,
                              prompt=PROMPT, delivery_mode='channel', status_message_id=status.id)
    gateway.jobs.update(job['id'], status='running')
    job = gateway.jobs.get(job['id'])

    result = {'job_id': job['id'], 'status': 'failed', 'seconds': None, 'answer': None,
              'artifacts': [], 'status_message': status.id,
              'posted_messages': None, 'status_edits': None}
    try:
        await gateway.run_job(job)
        job = gateway.jobs.get(job['id'])
        result['status'] = job['status']
        await gateway.deliver(gateway.jobs.get(job['id']))
        final = gateway.jobs.get(job['id'])
        result['answer'] = (final['answer'] or '')[:400]
        result['artifacts'] = [a['name'] for a in json.loads(final['artifacts'] or '[]')]
        result['delivered'] = bool(final['delivered'])
    finally:
        result['seconds'] = round(time.monotonic() - START, 1)
        result['posted_messages'] = len(channel.sent)
        # Progress edits are the visible proof of work while the task ran.
        result['status_edits'] = [text for _, text in status.history]
        result['status_final'] = status.content[:400]
        extras = [entry for entry in channel.sent if entry['id'] != status.id]
        result['extra_messages'] = extras
        result['member_sees'] = [('status', status.content[:200])] + [
            ('file' if entry['file'] else 'text', entry['file'] or (entry['content'] or '')[:200])
            for entry in extras]
        print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
        await server.cleanup()
        await gateway.session.close()
        gateway.jobs.close()


asyncio.run(main())
