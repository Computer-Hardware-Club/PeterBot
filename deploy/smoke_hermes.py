"""Run a genuine Hermes/tool/artifact smoke with synthetic identity and no Discord.

Run inside the gateway image on both networks, alias gateway and worker IP .2.
Never load production memory; PETERBOT_SMOKE_STATE must be a disposable directory.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import aiohttp
from aiohttp import web
from peterbot.config import AppConfig
from peterbot.agent_policy import Principal
from peterbot.hermes_settings import HermesSettings
from peterbot.hermes_gateway import HermesGateway, Capability
import time


async def main():
    config=AppConfig.load()
    settings=HermesSettings.load(os.environ['PETERBOT_HERMES_CONFIG'])
    settings=replace(settings,state_dir=os.environ['PETERBOT_SMOKE_STATE'])
    guild_id=next(iter(settings.allowed_guild_ids)); user_id=123456789012345678
    role_id=next(iter(settings.officer_role_ids)); channel_id=123456789012345679
    class Channel:
        async def send(self,*args,**kwargs): pass
    class Bot:
        async def fetch_channel(self,*args): return Channel()
    gateway=HermesGateway(Bot(),config,settings)
    async def principal(g,u,c,**kwargs):
        assert (g,u,c)==(guild_id,user_id,channel_id)
        return Principal(g,u,c,(role_id,))
    gateway.principal=principal
    gateway.session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),trust_env=False)
    app=web.Application(client_max_size=2*1024*1024)
    app.router.add_post('/tool',gateway.tool)
    app.router.add_post('/v1/chat/completions',gateway.model)
    app.router.add_get('/v1/models',gateway.models)
    server=web.AppRunner(app,access_log=None)
    await server.setup(); await web.TCPSite(server,'0.0.0.0',8770).start()
    try:
        prompt=('Complete this deployment smoke test. First use calculate for 137*29. '
            'Read /workspace/inputs/numbers.csv and use terminal to write /workspace/artifacts/smoke-result.txt '
            'with the sum of the amount column and the calculation result. '
            'Use peter_memory_add to save a personal preference: deployment smoke prefers concise answers. '
            'Then use peter_memory_search to verify it. Return a concise final answer with both arithmetic results. '
            'Do not contact Discord or any person. No web search is needed.')
        job=gateway.jobs.create(guild_id=guild_id,user_id=user_id,channel_id=channel_id,
            source_message_id=123456789012345680,prompt=prompt,
            input_files=[{'name':'numbers.csv','data_base64':base64.b64encode(b'amount\n10\n20\n30\n').decode()}])
        gateway.jobs.update(job['id'],status='running')
        await gateway.run_job(gateway.jobs.get(job['id']))
        result=gateway.jobs.get(job['id'])
        artifacts=json.loads(result['artifacts'])
        public={'status':result['status'],'answer':result['answer'],
                'artifacts':[{'name':f['name'],'text':base64.b64decode(f['data_base64']).decode(errors='replace')[:1000]} for f in artifacts],
                'memory_count':len(gateway.memory.search(Principal(guild_id,user_id,channel_id,(role_id,)),scope='personal'))}
        print(json.dumps(public,indent=2),flush=True)
        assert result['status']=='completed',public
        assert artifacts and any('3973' in base64.b64decode(f['data_base64']).decode(errors='replace') for f in artifacts),public
        assert public['memory_count']>0,public
    finally:
        await server.cleanup(); await gateway.session.close(); await gateway.tools.close()


asyncio.run(main())
