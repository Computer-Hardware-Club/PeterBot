"""Exercise the fast conversational turn against the real model, with no Discord.

This is the path a member mention takes when the model answers directly, so it is
where an empty answer or an internal error string used to reach the channel. Run it
inside the gateway image with the deployment configs:

    docker run --rm --network peterbot_control \
      --user 10000:10000 --cap-drop ALL --security-opt no-new-privileges:true \
      --tmpfs /tmp:rw,nosuid,nodev,size=64m \
      -e PETERBOT_CONFIG_FILE=/app/config.json \
      -v <appdata>/config.production.json:/app/config.json:ro \
      -v deploy/smoke_conversation_turn.py:/app/smoke_conversation_turn.py:ro \
      --entrypoint python peterbot-hermes-gateway:<rev> /app/smoke_conversation_turn.py

Every case must come back as reply text or an explicit handoff. A raised exception, or
an answer that leaks an internal string, fails the run.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import aiohttp

from peterbot.agent_policy import Principal
from peterbot.config import AppConfig
from peterbot.conversation import BLANK_ANSWER_REPLY, MODEL_UNAVAILABLE_REPLY
from peterbot.hermes_settings import HermesSettings
from peterbot.knowledge import load_knowledge_index
from peterbot.hermes_gateway import HermesGateway

CASES = (
    'sup',
    'when and where do we meet, and who is the president?',
    'who are the current club officers?',
    'what is 137*29?',
    'Read https://computerhardwareclub.org/ and tell me what the club does.',
    'draw me a diagram of the club network and save it as a file',
)

INTERNAL_STRINGS = ('Traceback', 'ValueError', 'conversation', 'Empty conversation', 'Unexpected')


async def main():
    os.environ.setdefault('DISCORD_TOKEN', 'conversation-smoke-unused-no-discord-connection')
    config = AppConfig.load()
    knowledge = load_knowledge_index(knowledge_file=config.knowledge_file, channel_profiles_file=None)
    # The conversational turn touches none of this: it exists so the gateway can be
    # constructed with its real dependencies and a disposable state directory.
    settings = HermesSettings(allowed_guild_ids=frozenset({1}), officer_role_ids=frozenset({2}),
                              owner_user_ids=frozenset({3}), runner_url='http://runner:8780',
                              tool_service_url='http://gateway:8770', runner_token='x' * 40,
                              state_dir=os.environ.get('PETERBOT_SMOKE_STATE', '/tmp/smoke-state'))

    class Bot:
        user = None

        def get_guild(self, *_args):
            return None

    gateway = HermesGateway(Bot(), config, settings)
    gateway.knowledge = knowledge
    gateway.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300), trust_env=False)
    principal = Principal(1, 2, 3, ())
    failures = []
    print(json.dumps({'knowledge_chunks': len(knowledge.chunks),
                      'knowledge_file': config.knowledge_file}), flush=True)
    try:
        for prompt in CASES:
            started = time.monotonic()
            try:
                reply = await gateway.conversational_reply(principal, prompt, [])
                error = None
            except Exception as exc:  # noqa: BLE001 - the smoke reports what a member would see
                reply, error = None, f'{type(exc).__name__}: {exc}'
            elapsed = round(time.monotonic() - started, 1)
            public = {'prompt': prompt, 'seconds': elapsed,
                      'outcome': 'handoff' if reply is None and not error else ('error' if error else 'reply'),
                      'reply': (reply or '')[:600], 'error': error}
            print(json.dumps(public, ensure_ascii=False), flush=True)
            if error:
                failures.append(public)
            if reply and any(token in reply for token in INTERNAL_STRINGS):
                public['outcome'] = 'leaked_internal_string'
                failures.append(public)
            if reply == BLANK_ANSWER_REPLY:
                failures.append(public)
            if reply == MODEL_UNAVAILABLE_REPLY:
                failures.append(public)
    finally:
        await gateway.session.close()
        await gateway.tools.close()
    print(json.dumps({'failures': len(failures)}, indent=2), flush=True)
    raise SystemExit(1 if failures else 0)


asyncio.run(main())
