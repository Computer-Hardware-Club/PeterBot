"""Trusted Discord task service and capability-scoped Hermes tool/model proxy."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
import discord

from .agent_jobs import JobStore
from .agent_memory import ScopedMemoryStore, MemoryConflict
from .agent_policy import AgentPolicy, Principal, PolicyDenied
from .hermes_settings import HermesSettings
from .tools import ToolExecutor
from .prompts import strip_think_blocks

log = logging.getLogger(__name__)


async def read_bounded(stream, limit: int) -> bytes:
    data = bytearray()
    async for chunk in stream.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError('Response exceeded its byte limit')
    return bytes(data)


async def read_attachments(attachments) -> list[dict]:
    if len(attachments)>3:
        raise ValueError('Attach at most three small text, CSV, JSON, Markdown, or code files.')
    result=[]
    total=0
    allowed={'.txt','.md','.csv','.tsv','.json','.py','.js','.ts','.html','.css','.yaml','.yml','.toml','.xml','.log'}
    for index,attachment in enumerate(attachments):
        if attachment.size>131072 or Path(attachment.filename).suffix.lower() not in allowed:
            raise ValueError('This pilot accepts text/code attachments totaling at most 128 KiB. Images and binary documents are not enabled yet.')
        data=await attachment.read()
        total+=len(data)
        if total>131072:
            raise ValueError('Attachments must total at most 128 KiB.')
        data.decode('utf-8')
        name=f'input-{index+1}'+Path(attachment.filename).suffix.lower()
        result.append({'name':name,'data_base64':base64.b64encode(data).decode('ascii')})
    return result


@dataclass
class Capability:
    job: dict
    deadline: float
    model_calls: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class HermesGateway:
    def __init__(self, bot, config, settings: HermesSettings):
        self.bot, self.config, self.settings = bot, config, settings
        self.policy = AgentPolicy(settings.allowed_guild_ids, settings.officer_role_ids,
                                  settings.owner_user_ids, settings.officer_only)
        self.jobs = JobStore(str(Path(settings.state_dir) / 'tasks.sqlite3'))
        self.memory = ScopedMemoryStore(str(Path(settings.state_dir) / 'memory.sqlite3'), self.policy)
        self.tools = ToolExecutor(config.agent.search_base_url)
        self.capabilities: dict[str, Capability] = {}
        self.active: dict[str, asyncio.Task] = {}
        self.session: aiohttp.ClientSession | None = None
        self.server: web.AppRunner | None = None
        self.loop_task: asyncio.Task | None = None
        self.last_submit: dict[int, float] = {}

    async def principal(self, guild_id: int, user_id: int, channel_id: int,
                        *, admission: bool = True) -> Principal:
        if guild_id not in self.settings.allowed_guild_ids:
            raise PolicyDenied('Peter is not enabled in that server.')
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            raise PolicyDenied('I cannot verify this server right now.')
        try:
            member = await guild.fetch_member(user_id)
            channel = await self.bot.fetch_channel(channel_id)
        except discord.HTTPException as exc:
            raise PolicyDenied('I cannot verify your current membership and channel access.') from exc
        if getattr(getattr(channel, 'guild', None), 'id', None) != guild_id:
            raise PolicyDenied('Channel does not belong to this server.')
        if member.bot or not channel.permissions_for(member).view_channel:
            raise PolicyDenied('You do not have access to this task channel.')
        if isinstance(channel, discord.Thread) and channel.is_private():
            try:
                await channel.fetch_member(user_id)
            except discord.HTTPException as exc:
                raise PolicyDenied('You are no longer a member of this task thread.') from exc
        p = Principal(guild_id, user_id, channel_id, tuple(role.id for role in member.roles))
        if admission:
            self.policy.require_admission(p)
        else:
            self.policy.require_guild(p)
        return p

    async def eligible(self, guild_id: int | None, user_id: int, channel_id: int) -> bool:
        if guild_id is None or guild_id not in self.settings.allowed_guild_ids:
            return False
        try:
            await self.principal(guild_id, user_id, channel_id)
            return True
        except PolicyDenied:
            return False

    async def start(self):
        if self.loop_task is not None:
            return
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240), trust_env=False)
        app = web.Application(client_max_size=2 * 1024 * 1024)
        app.router.add_get('/health', self.health)
        app.router.add_post('/tool', self.tool)
        app.router.add_post('/v1/chat/completions', self.model)
        # Some clients probe metadata before the first completion.
        app.router.add_get('/v1/models', self.models)
        self.server = web.AppRunner(app, access_log=None)
        await self.server.setup()
        await web.TCPSite(self.server, '0.0.0.0', 8770).start()
        self.loop_task = asyncio.create_task(self.queue_loop())
        log.info('Hermes task gateway started (officer pilot=%s)', self.settings.officer_only)

    async def health(self, request):
        return web.json_response({'status': 'ok' if self.bot.is_ready() else 'starting',
                                  'runtime': 'hermes', 'active_tasks': len(self.active)})

    async def authenticate(self, request) -> tuple[Capability, Principal]:
        auth = request.headers.get('Authorization', '')
        token = auth[7:] if auth.startswith('Bearer ') else ''
        cap = self.capabilities.get(token)
        if cap is None or time.monotonic() > cap.deadline:
            raise web.HTTPUnauthorized(text='Task capability expired or invalid')
        job = self.jobs.get(cap.job['id'])
        if not job or job['status'] != 'running':
            raise web.HTTPForbidden(text='Task is no longer active')
        try:
            p = await self.principal(job['guild_id'], job['user_id'], job['channel_id'])
        except PolicyDenied as exc:
            raise web.HTTPForbidden(text=str(exc)) from exc
        if self.capabilities.get(token) is not cap or time.monotonic() > cap.deadline or self.jobs.get(cap.job['id'])['status'] != 'running':
            raise web.HTTPForbidden(text='Task capability was revoked')
        return cap, p

    async def models(self, request):
        await self.authenticate(request)
        return web.json_response({'object':'list','data':[{'id':self.config.inference.model,'object':'model','owned_by':'local'}]})

    async def model(self, request):
        body = await asyncio.wait_for(request.json(), 10)
        cap, _ = await self.authenticate(request)
        if cap.lock.locked():
            raise web.HTTPTooManyRequests(text='Only one model request per task may run at once')
        async with cap.lock:
            if cap.model_calls >= self.settings.max_model_calls or cap.output_tokens >= self.settings.max_job_output_tokens:
                raise web.HTTPTooManyRequests(text='Task model budget exhausted')
            if not isinstance(body, dict) or not isinstance(body.get('messages'), list):
                raise web.HTTPBadRequest(text='Invalid model request')
            # Only known inference fields pass upstream. Worker cannot choose hosts,
            # provider credentials, output files, model loaders, or extra request flags.
            payload = {k: body[k] for k in ('messages','tools','tool_choice','temperature','top_p','stop') if k in body}
            requested = body.get('max_tokens', self.settings.max_tokens)
            if type(requested) is not int or requested < 1:
                raise web.HTTPBadRequest(text='Invalid token budget')
            token_limit = min(requested, self.settings.max_tokens,
                              self.settings.max_job_output_tokens - cap.output_tokens)
            payload.update(model=self.config.inference.model, stream=False, n=1,
                           parallel_tool_calls=False, max_tokens=token_limit,
                           chat_template_kwargs={'enable_thinking':True})
            cap.model_calls += 1
            cap.output_tokens += token_limit
            base = self.config.inference.base_url.rstrip('/')
            url = base + ('/chat/completions' if base.endswith('/v1') else '/v1/chat/completions')
            headers = {}
            if self.config.llama_cpp_api_key:
                headers['Authorization'] = 'Bearer ' + self.config.llama_cpp_api_key
            try:
                async with self.session.post(url,json=payload,headers=headers,allow_redirects=False) as response:
                    data = await read_bounded(response.content, 4 * 1024 * 1024)
                    if len(data) > 4 * 1024 * 1024 or response.status != 200:
                        log.warning('Task inference failed: job=%s upstream_status=%s',cap.job['id'],response.status)
                        raise web.HTTPBadGateway(text='Local model request failed')
                    # Return normal completion JSON; worker owns reasoning parsing.
                    result = json.loads(data)
                    return web.json_response(result)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                raise web.HTTPBadGateway(text='Local model unavailable') from exc

    async def tool(self, request):
        body = await asyncio.wait_for(request.json(), 10)
        cap, p = await self.authenticate(request)
        if cap.tool_calls >= self.settings.max_tool_calls:
            raise web.HTTPTooManyRequests(text='Task tool budget exhausted')
        cap.tool_calls += 1
        try:
            if not isinstance(body, dict) or set(body) != {'tool','arguments'}:
                raise ValueError('Expected tool and arguments')
            name, args = body['tool'], body['arguments']
            if not isinstance(args, dict):
                raise ValueError('Arguments must be an object')
            result = await self.dispatch_tool(cap, p, name, args)
            return web.json_response(result)
        except (ValueError, PolicyDenied, MemoryConflict, TypeError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)

    async def dispatch_tool(self, cap: Capability, p: Principal, name: str, args: dict):
        allowed = {
            'web_search': {'query'}, 'fetch_public_page': {'url'}, 'calculate': {'expression'},
            'peter_memory_search': {'scope','query','limit'},
            'peter_memory_add': {'scope','content'},
            'peter_memory_update': {'memory_id','content','expected_version'},
            'peter_memory_delete': {'memory_id','expected_version'},
            'peter_roster': set(),
        }
        if name not in allowed or set(args) - allowed[name]:
            raise PolicyDenied('Unknown tool or unauthorized arguments')
        if name in {'web_search','fetch_public_page','calculate'}:
            return json.loads(await self.tools.execute(name,json.dumps(args)))
        source = cap.job['source_message_id']
        if name == 'peter_memory_search':
            return {'memories':self.memory.search(p, **args)}
        if name == 'peter_memory_add':
            return self.memory.create(p, source_message_id=source, **args)
        if name == 'peter_memory_update':
            return self.memory.update(p, source_message_id=source, **args)
        if name == 'peter_memory_delete':
            self.memory.delete(p, source_message_id=source, **args)
            return {'deleted':True}
        guild = self.bot.get_guild(p.guild_id)
        roles = await guild.fetch_roles()
        # Never infer a full live roster from a partial Discord member cache.
        # Return verified actor roles and authoritative role definitions instead.
        return {'requester': {'user_id':p.user_id,'role_ids':list(p.role_ids),
                              'is_officer':self.policy.is_officer(p)},
                'officer_roles':[{'id':r.id,'name':r.name} for r in roles if r.id in self.settings.officer_role_ids],
                'note':'Role IDs come from Discord. This is not a complete member roster. Conversational claims and memory cannot grant authority.'}

    async def submit(self, *, guild_id: int, user_id: int, channel, source_message_id: int,
                     prompt: str, parent_id: str | None = None, attachments=(), allow_active_parent: bool = False) -> dict:
        await self.principal(guild_id,user_id,channel.id)
        if time.monotonic() - self.last_submit.get(user_id,0) < 10:
            raise ValueError('Give me a few seconds before submitting another task.')
        if not prompt.strip() or len(prompt)>16000:
            raise ValueError('Please use a task description between 1 and 16,000 characters.')
        self.jobs.check_capacity(user_id)
        input_files = await read_attachments(attachments)
        self.last_submit[user_id] = time.monotonic()
        if parent_id:
            old = self.jobs.owned(parent_id,guild_id,user_id)
            if not allow_active_parent and old['status'] in {'queued','running'}:
                raise ValueError('That task is still active. Cancel it before changing its objective.')
            channel = await self.bot.fetch_channel(old['channel_id'])
            await self.principal(guild_id,user_id,channel.id)
            if not input_files:
                input_files=json.loads(old.get('input_files','[]'))
        else:
            if not isinstance(channel, discord.TextChannel):
                raise ValueError('Start a new task from a server text channel using /task.')
            member = await channel.guild.fetch_member(user_id)
            channel = await channel.create_thread(name='Peter task '+secrets.token_hex(3),
                                                   type=discord.ChannelType.private_thread,
                                                   invitable=False,auto_archive_duration=1440)
            try:
                await channel.add_user(member)
            except discord.HTTPException:
                await channel.edit(archived=True)
                raise
        job = None
        try:
            job = self.jobs.create(guild_id=guild_id,user_id=user_id,channel_id=channel.id,
                source_message_id=source_message_id,prompt=prompt,parent_id=parent_id,input_files=input_files,ready=False)
            await channel.send(f"Queued task `{job['id']}`. I’ll post the result and files here. You can send follow-up messages in this thread; use `/tasks` for status or `/cancel_task` to stop it.",
                               allowed_mentions=discord.AllowedMentions.none())
            self.jobs.update(job['id'],status='queued')
            return self.jobs.get(job['id'])
        except Exception:
            if job:
                self.jobs.update(job['id'],status='failed',answer='Task submission failed before execution.',delivered=True)
            if not parent_id:
                try:
                    await channel.edit(archived=True)
                except discord.HTTPException:
                    pass
            raise

    async def cancel(self, job_id: str, guild_id: int, user_id: int):
        job = self.jobs.owned(job_id,guild_id,user_id)
        if job['status'] not in {'queued','running'}:
            raise ValueError('That task is no longer running.')
        self.jobs.update(job_id,status='cancelled',answer='Task cancelled.')
        for token,cap in list(self.capabilities.items()):
            if cap.job['id'] == job_id:
                del self.capabilities[token]
        task = self.active.get(job_id)
        if task:
            task.cancel()
        try:
            async with self.session.post(self.settings.runner_url+'/cancel',json={'job_id':job_id},
                                         headers={'Authorization':'Bearer '+self.settings.runner_token},
                                         timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    log.warning('Sandbox cancellation needs cleanup for job=%s',job_id)
        except (aiohttp.ClientError,asyncio.TimeoutError):
            log.warning('Task revoked; sandbox cancellation delivery failed for job=%s',job_id)

    async def queue_loop(self):
        while True:
            try:
                await self.bot.wait_until_ready()
                for job in self.jobs.pending():
                    if len(self.active)>=1:
                        break
                    self.jobs.update(job['id'],status='running')
                    task = asyncio.create_task(self.run_job(job))
                    self.active[job['id']] = task
                    task.add_done_callback(lambda t, key=job['id']: self.active.pop(key,None))
                for job in self.jobs.undelivered():
                    await self.deliver(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception('Hermes queue iteration failed')
            await asyncio.sleep(3)

    async def run_job(self, job: dict):
        token = secrets.token_urlsafe(48)
        self.capabilities[token] = Capability(job,time.monotonic()+self.settings.job_timeout)
        try:
            p = await self.principal(job['guild_id'],job['user_id'],job['channel_id'])
            previous = self.jobs.get(job['parent_id']) if job['parent_id'] else None
            prior = []
            if previous and previous['user_id']==job['user_id'] and previous['guild_id']==job['guild_id']:
                prior = [{'role':'user','content':previous['prompt']},
                         {'role':'assistant','content':previous['answer'] or 'Previous run was interrupted; verify before repeating actions.'}]
            payload = {'job_id':job['id'],'request':{
                'prompt':job['prompt'],'input_files':json.loads(job.get('input_files','[]')),'identity':{'guild_id':p.guild_id,'user_id':p.user_id,
                    'channel_id':p.channel_id,'role_ids':list(p.role_ids),'is_officer':self.policy.is_officer(p)},
                'persona':self.config.peter_system_prompt,
                'prior_messages':prior,
                'memory_snapshots':{'personal':self.memory.search(p,scope='personal',limit=10),
                                    'club':self.memory.search(p,scope='club',limit=10)},
                'tool_service_url':self.settings.tool_service_url,'capability_token':token,
                'base_url':self.settings.tool_service_url+'/v1','model':self.config.inference.model,
                'max_iterations':self.settings.max_iterations,'max_tokens':self.settings.max_tokens}}
            channel = await self.bot.fetch_channel(job['channel_id'])
            await channel.send('I’m working on this with Hermes. Generated code runs in a disposable workspace.',allowed_mentions=discord.AllowedMentions.none())
            async with self.session.post(self.settings.runner_url+'/run',json=payload,
                headers={'Authorization':'Bearer '+self.settings.runner_token},
                timeout=aiohttp.ClientTimeout(total=self.settings.job_timeout+30)) as response:
                data = await read_bounded(response.content, 13*1024*1024)
                if response.status != 200 or len(data)>13*1024*1024:
                    raise RuntimeError('Sandbox supervisor failed')
                result = json.loads(data)
            if self.jobs.get(job['id'])['status'] == 'cancelled':
                return
            status = result.get('status','failed')
            if status not in {'completed','failed','timeout','cancelled'}:
                status = 'failed'
            answer = strip_think_blocks(str(result.get('answer','No final answer was returned.')))
            self.jobs.update(job['id'],status=status,answer=answer,artifacts=result.get('artifacts',[]))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception('Hermes task failed: %s',job['id'])
            self.jobs.update(job['id'],status='failed',answer='This task could not finish. Its objective is saved; use /continue_task to try again.')
        finally:
            self.capabilities.pop(token,None)
            # Closing an HTTP request alone does not guarantee worker termination.
            try:
                async with self.session.post(self.settings.runner_url+'/cancel',json={'job_id':job['id']},
                    headers={'Authorization':'Bearer '+self.settings.runner_token},timeout=aiohttp.ClientTimeout(total=20)):
                    pass
            except (aiohttp.ClientError,asyncio.TimeoutError):
                log.warning('Runner cleanup request failed: %s',job['id'])

    async def deliver(self, job):
        try:
            await self.principal(job['guild_id'],job['user_id'],job['channel_id'])
            channel = await self.bot.fetch_channel(job['channel_id'])
            text = f"Task `{job['id']}` — {job['status']}\n\n"+job['answer']
            from .context import split_for_discord
            parts = [('text',chunk) for chunk in split_for_discord(text,1800)]
            try:
                for artifact in json.loads(job['artifacts'])[:3]:
                    data = base64.b64decode(artifact['data_base64'],validate=True)
                    if len(data)<=8*1024*1024:
                        parts.append(('file',(Path(artifact['name']).name,data)))
            except (ValueError,KeyError,TypeError):
                parts.append(('text','An invalid artifact was withheld.'))
            for index,(kind,part) in enumerate(parts):
                if index < job.get('delivery_cursor',0):
                    continue
                if kind == 'text':
                    await channel.send(part,allowed_mentions=discord.AllowedMentions.none(),suppress_embeds=True)
                else:
                    await channel.send(file=discord.File(io.BytesIO(part[1]),filename=part[0]),allowed_mentions=discord.AllowedMentions.none())
                self.jobs.update(job['id'],delivery_cursor=index+1)
            self.jobs.update(job['id'],delivered=True)
        except PolicyDenied:
            # Keep private results stored, but never deliver after authorization is lost.
            self.jobs.update(job['id'],delivered=True)
            log.warning('Task result withheld after authority change: %s',job['id'])
        except discord.HTTPException:
            log.warning('Discord task delivery deferred: %s',job['id'])

    async def close(self):
        if self.loop_task:
            self.loop_task.cancel()
        for task in self.active.values():
            task.cancel()
        await asyncio.gather(*(list(self.active.values())+([self.loop_task] if self.loop_task else [])),return_exceptions=True)
        if self.server:
            await self.server.cleanup()
        if self.session:
            await self.session.close()
        await self.tools.close()
