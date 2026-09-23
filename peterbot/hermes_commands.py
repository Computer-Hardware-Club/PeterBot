"""Discord UI for durable, isolated agent tasks."""
from __future__ import annotations

import discord
from .agent_policy import PolicyDenied
from .context import safe_send_interaction_message


def task_link(job: dict) -> str:
    return f"https://discord.com/channels/{job['guild_id']}/{job['channel_id']}"


async def submit_interaction(service, interaction, prompt, parent_id=None, attachment=None):
    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True)
    try:
        if not interaction.guild:
            raise PolicyDenied('Start tasks in the club server; DMs are disabled.')
        job = await service.submit(guild_id=interaction.guild.id,user_id=interaction.user.id,
                                   channel=interaction.channel,source_message_id=interaction.id,
                                   prompt=prompt,parent_id=parent_id,attachments=[attachment] if attachment else [])
        await safe_send_interaction_message(interaction,f"Task `{job['id']}` is queued: {task_link(job)}")
    except (ValueError, PolicyDenied) as exc:
        await safe_send_interaction_message(interaction,str(exc))
    except discord.HTTPException:
        await safe_send_interaction_message(interaction,'I could not create or access a private task thread. Check my thread permissions in this channel.')


def register_agent_commands(bot, service):
    @bot.tree.command(name='task',description='Give Peter a task in a private thread (officer pilot)')
    async def task(interaction: discord.Interaction, prompt: str, attachment: discord.Attachment | None = None):
        await submit_interaction(service,interaction,prompt,attachment=attachment)

    @bot.tree.command(name='tasks',description='Show your recent Peter tasks and their IDs')
    async def tasks(interaction: discord.Interaction):
        if not interaction.guild:
            return await safe_send_interaction_message(interaction,'Use this in the club server.')
        rows=service.jobs.list_owned(interaction.guild.id,interaction.user.id)
        text='\n'.join(f"`{r['id']}`: {r['status']}" for r in rows) or 'No saved tasks.'
        await safe_send_interaction_message(interaction,text)

    @bot.tree.command(name='cancel_task',description='Stop one of your queued or running Peter tasks')
    async def cancel_task(interaction: discord.Interaction, task_id: str):
        await interaction.response.defer(ephemeral=True)
        try:
            if not interaction.guild:
                raise ValueError('Use this in the club server.')
            await service.cancel(task_id,interaction.guild.id,interaction.user.id)
            await safe_send_interaction_message(interaction,
                "cancel requested. i'll keep any valid files")
        except (ValueError, PolicyDenied) as exc:
            await safe_send_interaction_message(interaction,str(exc))

    @bot.tree.command(name='continue_task',description='Continue a completed or interrupted task in its original private thread')
    async def continue_task(interaction: discord.Interaction, task_id: str, prompt: str):
        await submit_interaction(service,interaction,prompt,parent_id=task_id)

    @bot.tree.command(name='memory',description='Inspect the personal or public club memories Peter can use')
    @discord.app_commands.choices(scope=[discord.app_commands.Choice(name='My personal memory',value='personal'),
                                        discord.app_commands.Choice(name='Public club memory',value='club')])
    async def memory(interaction: discord.Interaction, scope: str='personal', query: str=''):
        await interaction.response.defer(ephemeral=True)
        try:
            if not interaction.guild:
                raise PolicyDenied('Use this in the club server.')
            p=await service.principal(interaction.guild.id,interaction.user.id,interaction.channel.id,admission=False)
            rows=service.memory.search(p,scope=scope,query=query,limit=10)
            text='\n\n'.join(f"`{r['id']}` (v{r['version']})\n{r['content'][:700]}" for r in rows) or 'No matching memories.'
            from .context import split_for_discord
            for chunk in split_for_discord(text,1800):
                await safe_send_interaction_message(interaction,chunk)
        except (ValueError,PolicyDenied) as exc:
            await safe_send_interaction_message(interaction,str(exc))

    @bot.tree.command(name='forget',description='Delete a memory you are authorized to change')
    async def forget(interaction: discord.Interaction, memory_id: str, version: int):
        await interaction.response.defer(ephemeral=True)
        try:
            if not interaction.guild:
                raise PolicyDenied('Use this in the club server.')
            p=await service.principal(interaction.guild.id,interaction.user.id,interaction.channel.id,admission=False)
            service.memory.delete(p,memory_id,source_message_id=interaction.id,expected_version=version)
            await safe_send_interaction_message(interaction,'Memory removed from recall. A restricted audit revision remains for accountability.')
        except (ValueError,PolicyDenied,KeyError,RuntimeError) as exc:
            await safe_send_interaction_message(interaction,str(exc))
