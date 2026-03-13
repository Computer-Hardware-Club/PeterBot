from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import discord
from discord.ext import commands, tasks

from .context import (
    build_current_mention_prompt_text,
    build_mention_context_bundle,
    build_recap_history,
    get_channel_context_messages,
    get_recent_channel_entries,
    load_mention_image_payloads,
    resolve_reply_target_entry,
    safe_send_interaction_message,
    send_chunked_followup,
    send_chunked_reply,
)
from .knowledge import rank_knowledge_chunks, resolve_channel_profile
from .logging_utils import (
    build_user_debug_message,
    interaction_log_context,
    log_exception_with_context,
    log_with_context,
    message_log_context,
    truncate_for_log,
)
from .prompts import (
    CHAT_MODE,
    MENTION_MODE,
    RECAP_MODE,
    build_context_line,
    build_system_prompt,
)
from .reminders import check_missed_reminders, deliver_reminder, parse_reminder_time
from .runtime import PeterBotRuntime


def build_prompt_artifacts(
    *,
    config: Any,
    knowledge_index: Any,
    prompt_text: str,
    author_name: Optional[str],
    guild_name: Optional[str],
    channel: Any,
    focus_note: Optional[str] = None,
    mode: str = CHAT_MODE,
    include_channel_profile: bool = True,
    include_knowledge: bool = True,
) -> tuple[str, list[Any]]:
    channel_profile = (
        resolve_channel_profile(channel, knowledge_index.channel_profiles)
        if include_channel_profile
        else None
    )
    knowledge_chunks = (
        rank_knowledge_chunks(
            prompt_text,
            knowledge_index.chunks,
            channel_profile=channel_profile,
        )
        if include_knowledge
        else []
    )
    context_line = build_context_line(
        author_name=author_name,
        guild_name=guild_name,
        channel_name=getattr(channel, "name", None),
    )
    system_prompt = build_system_prompt(
        config,
        context_line,
        mode=mode,
        focus_note=focus_note,
        channel_profile=channel_profile,
        knowledge_chunks=knowledge_chunks,
    )
    return system_prompt, knowledge_chunks


async def send_suggestion_to_channel(
    bot: commands.Bot,
    suggestion_channel_id: int,
    user_id: int,
    username: str,
    suggestion: str,
) -> bool:
    channel = bot.get_channel(suggestion_channel_id)
    if channel is None:
        try:
            channel = await bot.fetch_channel(suggestion_channel_id)
        except discord.HTTPException:
            log_exception_with_context(
                "Failed fetching suggestion channel",
                suggestion_channel_id=suggestion_channel_id,
                user_id=user_id,
            )
            return False

    if channel is None or not hasattr(channel, "send"):
        log_with_context(
            logging.ERROR,
            "Suggestion channel is not messageable",
            suggestion_channel_id=suggestion_channel_id,
            user_id=user_id,
        )
        return False

    embed = discord.Embed(
        title="New Suggestion",
        description=suggestion,
        color=0x00FF00,
        timestamp=datetime.now(),
    )
    embed.add_field(name="Suggested by", value=f"{username} (<@{user_id}>)", inline=False)
    embed.set_footer(text="PSS (Peter's Suggestion System)")

    try:
        await channel.send(embed=embed)
        return True
    except discord.HTTPException:
        log_exception_with_context(
            "Failed sending suggestion embed",
            suggestion_channel_id=suggestion_channel_id,
            user_id=user_id,
            suggestion_preview=truncate_for_log(suggestion),
        )
        return False


def clamp_recap_count(count: int, maximum: int) -> int:
    return max(5, min(count, maximum))


def register_handlers(bot: commands.Bot, runtime: PeterBotRuntime) -> None:
    config = runtime.config

    @tasks.loop(seconds=30)
    async def reminder_checker() -> None:
        due_reminders = runtime.reminder_manager.pop_due_reminders()
        if not due_reminders:
            return

        retry_count = 0
        for reminder in due_reminders:
            status = await deliver_reminder(
                bot,
                runtime.reminder_manager,
                reminder,
                missed=False,
            )
            if status == "retry":
                runtime.reminder_manager.requeue_reminder(reminder, runtime.retry_delay)
                retry_count += 1

        runtime.reminder_manager.save_reminders()
        if retry_count:
            log_with_context(
                logging.INFO,
                "Requeued due reminders due to transient delivery errors",
                retry_count=retry_count,
            )

    @reminder_checker.before_loop
    async def before_reminder_checker() -> None:
        await bot.wait_until_ready()

    @bot.event
    async def on_ready() -> None:
        log_with_context(
            logging.INFO,
            "Bot connected to Discord gateway",
            bot_user=bot.user,
            data_dir=config.data_dir,
            ollama_base_url=config.ollama_base_url,
            ollama_model=config.ollama_model,
            ollama_think=config.ollama_think,
            model_profile=config.model_profile.value,
            knowledge_chunks=len(runtime.knowledge_index.chunks),
            channel_profiles=len(runtime.knowledge_index.channel_profiles),
        )

        if not runtime.has_initialized:
            runtime.reminder_manager.load_reminders()
            await check_missed_reminders(
                bot,
                runtime.reminder_manager,
                retry_delay=runtime.retry_delay,
            )
            runtime.has_initialized = True

        if not runtime.has_synced_commands:
            try:
                synced = await bot.tree.sync()
                runtime.has_synced_commands = True
                log_with_context(logging.INFO, "Synced slash commands", count=len(synced))
            except Exception:
                log_exception_with_context("Failed syncing slash commands")

        if not reminder_checker.is_running():
            reminder_checker.start()

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return

        if bot.user and bot.user in message.mentions:
            content = build_current_mention_prompt_text(message, bot_user_id=bot.user.id)
            try:
                recent_entries = await get_recent_channel_entries(
                    message.channel,
                    bot_user_id=bot.user.id,
                    peter_name=config.peter_name,
                    limit=config.mention_context_fetch_limit,
                    before=message.created_at,
                    max_chars=config.max_context_message_chars,
                )
                explicit_reply_entry = await resolve_reply_target_entry(
                    message,
                    recent_entries,
                    bot_user_id=bot.user.id,
                    peter_name=config.peter_name,
                    max_chars=config.max_context_message_chars,
                )
                mention_images = await load_mention_image_payloads(
                    message,
                    limit=config.mention_image_limit,
                    max_bytes=config.mention_max_image_bytes,
                )
                mention_bundle = build_mention_context_bundle(
                    message,
                    content,
                    recent_entries,
                    focus_message_limit=config.mention_focus_message_limit,
                    active_gap_minutes=config.mention_active_gap_minutes,
                    max_background_age_minutes=config.mention_max_background_age_minutes,
                    assistant_tail_limit=config.mention_assistant_tail_limit,
                    explicit_reply_entry=explicit_reply_entry,
                )
                log_with_context(
                    logging.DEBUG,
                    "Built mention focus context",
                    prompt_preview=truncate_for_log(content),
                    selection_reason=mention_bundle["selection_reason"],
                    target_message_id=mention_bundle["target_message_id"],
                    target_age=mention_bundle["target_age_text"],
                    selected_count=mention_bundle["selected_count"],
                    needs_strong_target=mention_bundle["needs_strong_target"],
                    **message_log_context(message),
                )

                if mention_bundle["clarification_text"]:
                    log_with_context(
                        logging.INFO,
                        "Mention requires clarification instead of stale guess",
                        selection_reason=mention_bundle["selection_reason"],
                        prompt_preview=truncate_for_log(content),
                        **message_log_context(message),
                    )
                    await send_chunked_reply(
                        message,
                        mention_bundle["clarification_text"],
                        max_len=config.max_discord_message_chars,
                    )
                    await bot.process_commands(message)
                    return

                system_prompt, knowledge_chunks = build_prompt_artifacts(
                    config=config,
                    knowledge_index=runtime.knowledge_index,
                    prompt_text=content,
                    author_name=message.author.display_name,
                    guild_name=message.guild.name if message.guild else None,
                    channel=message.channel,
                    focus_note=mention_bundle["focus_note"],
                    mode=MENTION_MODE,
                )
                log_with_context(
                    logging.DEBUG,
                    "Resolved mention prompt artifacts",
                    knowledge_count=len(knowledge_chunks),
                    **message_log_context(message),
                )

                async with message.channel.typing():
                    reply = await runtime.ollama_client.call_chat(
                        prompt_text=content,
                        author_name=message.author.display_name,
                        guild_name=message.guild.name if message.guild else None,
                        channel_name=getattr(message.channel, "name", None),
                        conversation_history=mention_bundle["conversation_history"],
                        system_prompt=system_prompt,
                        user_content=mention_bundle["user_content"],
                        user_images=mention_images,
                        response_mode=MENTION_MODE,
                    )
                await send_chunked_reply(
                    message,
                    reply or "(No response)",
                    max_len=config.max_discord_message_chars,
                )
            except Exception:
                debug_id = log_exception_with_context(
                    "Failed handling mention response",
                    prompt_preview=truncate_for_log(content),
                    **message_log_context(message),
                )
                await send_chunked_reply(
                    message,
                    build_user_debug_message(
                        "I hit an internal error while generating a reply.",
                        debug_id,
                    ),
                    max_len=config.max_discord_message_chars,
                )

        await bot.process_commands(message)

    @bot.event
    async def on_disconnect() -> None:
        log_with_context(logging.INFO, "Bot disconnected from Discord gateway")
        runtime.reminder_manager.save_reminders()

    @bot.event
    async def on_error(event_method: str, *args: Any, **kwargs: Any) -> None:
        log_exception_with_context(
            "Unhandled discord.py event error",
            event_method=event_method,
            args_preview=truncate_for_log(args),
            kwargs_preview=truncate_for_log(kwargs),
        )

    @bot.tree.error
    async def on_app_command_error(
        interaction: discord.Interaction,
        error: discord.app_commands.AppCommandError,
    ) -> None:
        debug_id = log_exception_with_context(
            "Unhandled app command error",
            error=repr(error),
            **interaction_log_context(interaction),
        )
        await safe_send_interaction_message(
            interaction,
            build_user_debug_message(
                "I hit an internal error while running that command.",
                debug_id,
            ),
            ephemeral=True,
        )

    @bot.event
    async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return

        debug_id = log_exception_with_context(
            "Unhandled prefix command error",
            error=repr(error),
            command=getattr(ctx.command, "qualified_name", None),
            author_id=getattr(ctx.author, "id", None),
            channel_id=getattr(ctx.channel, "id", None),
            guild_id=getattr(ctx.guild, "id", None),
        )
        try:
            await ctx.send(build_user_debug_message("I hit an internal command error.", debug_id))
        except discord.HTTPException:
            log_exception_with_context(
                "Failed sending prefix command error message",
                debug_id=debug_id,
                command=getattr(ctx.command, "qualified_name", None),
            )

    @bot.tree.command(name="hello", description="Say hello to the bot")
    async def hello(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("Hello!", ephemeral=True)

    @bot.tree.command(name="ask", description="Ask Peter (Ollama) a question")
    @discord.app_commands.describe(prompt="Your question or prompt for Peter")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            context_messages = await get_channel_context_messages(
                interaction.channel,
                bot_user_id=getattr(bot.user, "id", None),
                peter_name=config.peter_name,
                limit=config.channel_context_limit,
                before=interaction.created_at,
                max_chars=config.max_context_message_chars,
            )
            system_prompt, knowledge_chunks = build_prompt_artifacts(
                config=config,
                knowledge_index=runtime.knowledge_index,
                prompt_text=prompt,
                author_name=interaction.user.display_name,
                guild_name=interaction.guild.name if interaction.guild else None,
                channel=interaction.channel,
                mode=CHAT_MODE,
            )
            log_with_context(
                logging.DEBUG,
                "Resolved /ask prompt artifacts",
                knowledge_count=len(knowledge_chunks),
                **interaction_log_context(interaction),
            )

            if hasattr(interaction.channel, "typing"):
                async with interaction.channel.typing():
                    reply = await runtime.ollama_client.call_chat(
                        prompt_text=prompt,
                        author_name=interaction.user.display_name,
                        guild_name=interaction.guild.name if interaction.guild else None,
                        channel_name=getattr(interaction.channel, "name", None),
                        conversation_history=context_messages,
                        system_prompt=system_prompt,
                        response_mode=CHAT_MODE,
                    )
            else:
                reply = await runtime.ollama_client.call_chat(
                    prompt_text=prompt,
                    author_name=interaction.user.display_name,
                    guild_name=interaction.guild.name if interaction.guild else None,
                    channel_name=getattr(interaction.channel, "name", None),
                    conversation_history=context_messages,
                    system_prompt=system_prompt,
                    response_mode=CHAT_MODE,
                )
            delivered = await send_chunked_followup(
                interaction,
                reply or "(No response)",
                ephemeral=True,
                max_len=config.max_discord_message_chars,
            )
            if not delivered:
                await safe_send_interaction_message(
                    interaction,
                    "I generated a reply but couldn't deliver it. Please try again.",
                    ephemeral=True,
                )
        except Exception:
            debug_id = log_exception_with_context(
                "Error in /ask command",
                prompt_preview=truncate_for_log(prompt),
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message(
                    "I hit an internal error while talking to the model.",
                    debug_id,
                ),
                ephemeral=True,
            )

    @bot.tree.command(name="recap", description="Summarize the recent discussion in this channel")
    @discord.app_commands.describe(count="How many recent messages to include in the recap")
    async def recap(interaction: discord.Interaction, count: int = 25) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            recap_count = clamp_recap_count(count, config.recap_max_messages)
            recent_entries = await get_recent_channel_entries(
                interaction.channel,
                bot_user_id=getattr(bot.user, "id", None),
                peter_name=config.peter_name,
                limit=recap_count,
                before=interaction.created_at,
                max_chars=config.max_context_message_chars,
            )
            if not recent_entries:
                await safe_send_interaction_message(
                    interaction,
                    "I couldn't find enough recent messages to recap.",
                    ephemeral=True,
                )
                return

            system_prompt, _ = build_prompt_artifacts(
                config=config,
                knowledge_index=runtime.knowledge_index,
                prompt_text="Summarize the recent channel discussion.",
                author_name=interaction.user.display_name,
                guild_name=interaction.guild.name if interaction.guild else None,
                channel=interaction.channel,
                mode=RECAP_MODE,
                include_channel_profile=False,
                include_knowledge=False,
            )
            reply = await runtime.ollama_client.call_chat(
                prompt_text=f"Summarize the last {len(recent_entries)} messages in this channel.",
                author_name=interaction.user.display_name,
                guild_name=interaction.guild.name if interaction.guild else None,
                channel_name=getattr(interaction.channel, "name", None),
                conversation_history=build_recap_history(recent_entries, interaction.created_at),
                system_prompt=system_prompt,
                user_content=(
                    f"[Recap request | now] {interaction.user.display_name}: "
                    f"Recap the last {len(recent_entries)} messages."
                ),
                response_mode=RECAP_MODE,
            )
            await send_chunked_followup(
                interaction,
                reply,
                ephemeral=True,
                max_len=config.max_discord_message_chars,
            )
        except Exception:
            debug_id = log_exception_with_context(
                "Error in /recap command",
                requested_count=count,
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message(
                    "I couldn't generate that recap right now.",
                    debug_id,
                ),
                ephemeral=True,
            )

    @bot.tree.command(name="suggest", description="Submit a suggestion to improve the bot")
    @discord.app_commands.describe(suggestion="Your suggestion for improving the bot")
    async def suggest(interaction: discord.Interaction, suggestion: str) -> None:
        suggestion_channel_id = config.suggestion_channel_id
        if not suggestion_channel_id:
            await safe_send_interaction_message(
                interaction,
                "Suggestion channel is not configured. Please ask an admin to set `SUGGESTION_CHANNEL_ID`.",
                ephemeral=True,
            )
            return

        try:
            ok = await send_suggestion_to_channel(
                bot,
                suggestion_channel_id,
                interaction.user.id,
                interaction.user.display_name,
                suggestion,
            )
            if not ok:
                await safe_send_interaction_message(
                    interaction,
                    "I couldn't submit your suggestion right now. Please try again later.",
                    ephemeral=True,
                )
                return

            await safe_send_interaction_message(
                interaction,
                "Thanks for the suggestion. It has been submitted.",
                ephemeral=True,
            )
        except Exception:
            debug_id = log_exception_with_context(
                "Error in /suggest command",
                suggestion_preview=truncate_for_log(suggestion),
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message(
                    "I couldn't submit your suggestion right now.",
                    debug_id,
                ),
                ephemeral=True,
            )

    @bot.tree.command(name="remindme", description="Set a reminder for yourself")
    @discord.app_commands.describe(
        message="What you want to be reminded about",
        time="When to remind you (supports many formats: '10/08/2025 14:30', '2:30 PM', 'tomorrow', 'in 30 minutes')",
    )
    async def remindme(interaction: discord.Interaction, message: str, time: str) -> None:
        try:
            remind_time = parse_reminder_time(time)
            if remind_time is None:
                await safe_send_interaction_message(
                    interaction,
                    "❌ Invalid time format. Supported examples:\n\n"
                    "• `10/08/2025 14:30`\n"
                    "• `10/08/25 2:30 PM`\n"
                    "• `2025-10-08 14:30`\n"
                    "• `10/08` or `10/08 14:30`\n"
                    "• `14:30` or `2:30 PM`\n"
                    "• `tomorrow` or `tomorrow at 9:00 AM`\n"
                    "• `in 45 minutes`",
                    ephemeral=True,
                )
                return
            if remind_time <= datetime.now():
                await safe_send_interaction_message(
                    interaction,
                    "❌ Please set a reminder for a future time!",
                    ephemeral=True,
                )
                return

            runtime.reminder_manager.add_reminder(interaction.user.id, message, remind_time)
            time_str = remind_time.strftime("%A, %b %d, %Y at %I:%M %p")
            await safe_send_interaction_message(
                interaction,
                f"✅ Reminder set. I'll remind you about **{message}** on {time_str}.",
                ephemeral=True,
            )
        except Exception:
            debug_id = log_exception_with_context(
                "Error in /remindme command",
                reminder_message=truncate_for_log(message),
                reminder_time_input=time,
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message(
                    "❌ I couldn't set that reminder due to an internal error.",
                    debug_id,
                ),
                ephemeral=True,
            )
