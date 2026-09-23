from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import discord
from discord.ext import commands, tasks

from .agent_policy import PolicyDenied
from .awareness import AwarenessRouter
from .foreground import DuplicateEvent, ForegroundCancelled
from .context import (
    build_current_mention_prompt_text,
    build_mention_context_bundle,
    build_recap_history,
    get_channel_context_messages,
    get_recent_channel_entries,
    load_mention_image_payloads,
    message_has_image_attachments,
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

IMAGE_ATTACHMENT_READ_FAILURE_MESSAGE = "I couldn't read the attached image. Try a smaller image or reupload it."


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


async def resolve_mention_images(
    message: discord.Message,
    *,
    image_limit: int,
    max_image_bytes: int,
) -> tuple[list[str], Optional[str]]:
    if not message_has_image_attachments(message):
        return [], None

    images = await load_mention_image_payloads(
        message,
        limit=image_limit,
        max_bytes=max_image_bytes,
    )
    if images:
        return images, None

    log_with_context(
        logging.INFO,
        "Mention included image attachments but none were usable",
        image_limit=image_limit,
        max_image_bytes=max_image_bytes,
        **message_log_context(message),
    )
    return [], IMAGE_ATTACHMENT_READ_FAILURE_MESSAGE


def register_handlers(bot: commands.Bot, runtime: PeterBotRuntime) -> None:
    config = runtime.config
    awareness: AwarenessRouter | None = None

    def configured_awareness() -> AwarenessRouter | None:
        nonlocal awareness
        settings = getattr(getattr(runtime, "hermes", None), "settings", None)
        if not settings or not getattr(settings, "listen_channel_ids", None) or not bot.user:
            return None
        if awareness is None or awareness.bot_user_id != bot.user.id:
            awareness = AwarenessRouter(
                guild_ids=settings.allowed_guild_ids,
                channel_ids=settings.listen_channel_ids,
                bot_user_id=bot.user.id,
                name=config.peter_name,
                lease_seconds=settings.conversation_lease_seconds,
            )
        return awareness

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
            inference_base_url=config.inference.base_url,
            inference_model=config.inference.model,
            bundled_llama_server=config.llama_server.enabled,
            model_profile=config.model_profile.value,
            knowledge_chunks=len(runtime.knowledge_index.chunks),
            channel_profiles=len(runtime.knowledge_index.channel_profiles),
        )

        if not runtime.has_initialized:
            # One-shot restart reconciliation before any queue pump starts.
            # With Hermes on, recovery happens inside hermes.start() strictly
            # after the singleton lease is acquired — a refused duplicate
            # process must never touch the live process's rows.
            if getattr(runtime, "hermes", None) is None \
                    and getattr(runtime, "foreground", None) is not None:
                summary = runtime.foreground.recover()
                if any(summary.values()):
                    log_with_context(logging.WARNING, "Foreground restart reconciliation", **summary)
            runtime.reminder_manager.load_reminders()
            await check_missed_reminders(
                bot,
                runtime.reminder_manager,
                retry_delay=runtime.retry_delay,
            )
            runtime.has_initialized = True

        if getattr(runtime, "hermes", None) is not None:
            await runtime.hermes.start()

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
        if message.author.bot or getattr(message, "webhook_id", None):
            return

        router = configured_awareness()
        hermes = getattr(runtime, "hermes", None)
        control_proposal = None
        if hermes is not None and message.guild is not None:
            from .control_requests import parse_control_request
            control_proposal = parse_control_request(message.content,
                bot_user_id=getattr(bot.user, 'id', None))
        direct_mention = bool(bot.user and bot.user in (getattr(message, "mentions", None) or []))
        address_reason = "mention" if direct_mention else (await router.addressed(message) if router else None)
        if control_proposal is not None and message.channel.id in hermes.settings.control_channel_ids:
            address_reason = address_reason or "control"
        if address_reason:
            content = build_current_mention_prompt_text(message, bot_user_id=bot.user.id)
            foreground = runtime.foreground
            guild_id = getattr(message.guild, "id", None) or message.channel.id
            # Deterministic ingress validation first: no slot, no queue row,
            # no model call for a message we would reject anyway.
            admitted, reason = runtime.request_guard.preflight(
                user_id=message.author.id, guild_id=getattr(message.guild, "id", None), prompt=content,
            )
            if not admitted:
                await send_chunked_reply(message, reason or "Please try again shortly.",
                                         max_len=config.max_discord_message_chars)
                await bot.process_commands(message)
                return
            # In the configured guild, a denied Hermes pilot request must not
            # fall through to the old unrestricted mention model path.
            hermes_path = hermes is not None and (
                control_proposal is not None
                or getattr(message.guild, "id", None) in getattr(
                    getattr(hermes, "settings", None), "allowed_guild_ids", frozenset())
                or await hermes.eligible(
                    getattr(message.guild, "id", None), message.author.id, message.channel.id)
            )

            def guarded(work):
                async def run():
                    ok, why = runtime.request_guard.acquire(
                        user_id=message.author.id,
                        guild_id=getattr(message.guild, "id", None), prompt=content)
                    if not ok:
                        raise ValueError(why or "Give me a moment.")
                    try:
                        return await work()
                    finally:
                        runtime.request_guard.release(user_id=message.author.id)
                return run

            async def hermes_work():
                try:
                    if control_proposal is not None:
                        if message.attachments:
                            raise ValueError('Club control requests cannot include attachments.')
                        if await hermes.handle_control_message(message, content):
                            return
                        raise ValueError('I could not recognize that control request clearly.')
                    await hermes.respond_to_message(message, content)
                except (ValueError, PolicyDenied):
                    raise
                except Exception:
                    log_exception_with_context('Conversation reply failed', **message_log_context(message))
                    await send_chunked_reply(message, 'Something went wrong. Try me again in a moment.')

            async def direct_mention_work():
                # Outer slack over the model deadline, so a slow round is reported by
                # call_chat's own timeout instead of as an internal error here.
                async with asyncio.timeout(config.agent.request_timeout_seconds + 15):
                    mention_images, image_error = await resolve_mention_images(
                        message,
                        image_limit=config.mention_image_limit,
                        max_image_bytes=config.mention_max_image_bytes,
                    )
                    if image_error:
                        await send_chunked_reply(
                            message,
                            image_error,
                            max_len=config.max_discord_message_chars,
                        )
                        return

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
                        reply = await runtime.llm_client.call_chat(
                            prompt_text=content,
                            author_name=message.author.display_name,
                            guild_name=message.guild.name if message.guild else None,
                            channel_name=getattr(message.channel, "name", None),
                            conversation_history=mention_bundle["conversation_history"],
                            system_prompt=system_prompt,
                            user_content=mention_bundle["user_content"],
                            user_images=mention_images or None,
                            response_mode=MENTION_MODE,
                        )
                    await send_chunked_reply(
                        message,
                        reply or "(No response)",
                        max_len=config.max_discord_message_chars,
                    )

            work = guarded(hermes_work if hermes_path else direct_mention_work)

            if router:
                router.remember(message, address_reason)

            async def acknowledge(position: int) -> None:
                # Transport-only queue ack: no model call, and it carries only
                # a count — never another requester's prompt or channel.
                await send_chunked_reply(
                    message,
                    f"{position} ahead of you, i'll reply here",
                    max_len=config.max_discord_message_chars)

            try:
                await foreground.run_one(
                    kind='chat', guild_id=guild_id, user_id=message.author.id,
                    channel_id=message.channel.id, source_message_id=message.id,
                    work=work, acknowledge=acknowledge)
            except DuplicateEvent:
                log_with_context(logging.DEBUG, "Suppressed duplicate Discord event",
                                 **message_log_context(message))
            except (ValueError, PolicyDenied) as exc:
                await send_chunked_reply(message, str(exc),
                                         max_len=config.max_discord_message_chars)
            except ForegroundCancelled as exc:
                await send_chunked_reply(message, str(exc),
                                         max_len=config.max_discord_message_chars)
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

    @bot.tree.command(name="ask", description="Ask Peter a question")
    @discord.app_commands.describe(prompt="Your question or prompt for Peter")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        guild_id = getattr(interaction.guild, "id", None)
        admitted, reason = runtime.request_guard.preflight(
            user_id=interaction.user.id, guild_id=guild_id, prompt=prompt,
        )
        if not admitted:
            await safe_send_interaction_message(interaction, reason or "Please try again shortly.")
            return
        # Defer first so a queue ack and the eventual answer both fit the
        # interaction lifetime; the ack is ephemeral like the answer, so a
        # private /ask never touches a public channel.
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            debug_id = log_exception_with_context(
                "Failed to defer /ask interaction",
                prompt_preview=truncate_for_log(prompt),
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message("I couldn't acknowledge that request. Try again.", debug_id),
            )
            return

        async def work():
            ok, why = runtime.request_guard.acquire(
                user_id=interaction.user.id, guild_id=guild_id, prompt=prompt)
            if not ok:
                raise ValueError(why or "Please try again shortly.")
            try:
                async with asyncio.timeout(config.agent.request_timeout_seconds) as turn_timeout:
                    context_messages = await get_channel_context_messages(
                        interaction.channel,
                        bot_user_id=getattr(bot.user, "id", None),
                        peter_name=config.peter_name,
                        limit=config.channel_context_limit,
                        before=interaction.created_at,
                        max_chars=config.max_context_message_chars,
                    )
                    hermes = getattr(runtime, 'hermes', None)
                    if hermes is not None:
                        if guild_id is None:
                            raise PolicyDenied('Use Peter in the club server; DMs are disabled.')
                        principal = await hermes.principal(guild_id, interaction.user.id,
                                                           interaction.channel.id)
                        reply = await hermes.conversational_reply(
                            principal, prompt, context_messages, audience='private',
                            budget_seconds=max(0.0, turn_timeout.when() - asyncio.get_running_loop().time()))
                        if reply is not None:
                            return reply
                        parent = hermes.jobs.latest_for_thread(
                            principal.guild_id, principal.user_id, principal.channel_id)
                        job = await hermes.submit(guild_id=principal.guild_id,
                            user_id=principal.user_id, channel=interaction.channel,
                            source_message_id=interaction.id, prompt=prompt,
                            parent_id=parent['id'] if parent else None)
                        return (f"This needs tools, so I started it in your task thread: "
                                f"https://discord.com/channels/{job['guild_id']}/{job['channel_id']}")
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
                            reply = await runtime.llm_client.call_chat(
                                prompt_text=prompt,
                                author_name=interaction.user.display_name,
                                guild_name=interaction.guild.name if interaction.guild else None,
                                channel_name=getattr(interaction.channel, "name", None),
                                conversation_history=context_messages,
                                system_prompt=system_prompt,
                                response_mode=CHAT_MODE,
                            )
                    else:
                        reply = await runtime.llm_client.call_chat(
                            prompt_text=prompt,
                            author_name=interaction.user.display_name,
                            guild_name=interaction.guild.name if interaction.guild else None,
                            channel_name=getattr(interaction.channel, "name", None),
                            conversation_history=context_messages,
                            system_prompt=system_prompt,
                            response_mode=CHAT_MODE,
                        )
                    return reply
            finally:
                runtime.request_guard.release(user_id=interaction.user.id)

        async def acknowledge(position: int) -> None:
            await safe_send_interaction_message(
                interaction,
                f"{position} ahead of you, i'll answer here",
                ephemeral=True,
            )

        try:
            _row, reply = await runtime.foreground.run_one(
                kind='ask',
                guild_id=guild_id or interaction.channel.id,
                user_id=interaction.user.id,
                channel_id=interaction.channel.id,
                source_message_id=interaction.id,
                work=work,
                acknowledge=acknowledge,
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
            elif getattr(runtime, 'hermes', None) is not None and reply and not reply.startswith('This needs tools,'):
                try:
                    runtime.hermes.conversations.append_turn(
                        guild_id=guild_id, user_id=interaction.user.id,
                        channel_id=interaction.channel.id, source_message_id=interaction.id,
                        audience='private', prompt=prompt, answer=reply)
                except Exception:
                    log_exception_with_context('Could not record a delivered /ask turn')
        except DuplicateEvent:
            log_with_context(logging.DEBUG, "Suppressed duplicate /ask interaction",
                             **interaction_log_context(interaction))
        except (ValueError, PolicyDenied, ForegroundCancelled) as exc:
            await safe_send_interaction_message(interaction, str(exc), ephemeral=True)
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
        guild_id = getattr(interaction.guild, "id", None)
        prompt_text = "Recap the recent channel discussion."
        admitted, reason = runtime.request_guard.preflight(
            user_id=interaction.user.id, guild_id=guild_id, prompt=prompt_text,
        )
        if not admitted:
            await safe_send_interaction_message(interaction, reason or "Please try again shortly.")
            return
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            debug_id = log_exception_with_context(
                "Failed to defer /recap interaction",
                requested_count=count,
                **interaction_log_context(interaction),
            )
            await safe_send_interaction_message(
                interaction,
                build_user_debug_message("I couldn't acknowledge that request. Try again.", debug_id),
            )
            return

        async def work():
            # History is only read once the slot is ours: a queued recap does
            # no Discord reads and no model call while it waits.
            ok, why = runtime.request_guard.acquire(
                user_id=interaction.user.id, guild_id=guild_id, prompt=prompt_text)
            if not ok:
                raise ValueError(why or "Please try again shortly.")
            try:
                async with asyncio.timeout(config.agent.request_timeout_seconds):
                    style_instruction = ''
                    hermes = getattr(runtime, 'hermes', None)
                    if hermes is not None and guild_id in getattr(
                            getattr(hermes, 'settings', None), 'allowed_guild_ids', frozenset()):
                        await hermes.principal(guild_id, interaction.user.id, interaction.channel.id)
                        style_instruction = hermes.style.instruction(guild_id)
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
                        return None

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
                    if style_instruction:
                        system_prompt += '\n\n' + style_instruction
                    reply = await runtime.llm_client.call_chat(
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
                    return reply
            finally:
                runtime.request_guard.release(user_id=interaction.user.id)

        async def acknowledge(position: int) -> None:
            await safe_send_interaction_message(
                interaction,
                f"{position} ahead of you, i'll recap it here",
                ephemeral=True,
            )

        try:
            _row, reply = await runtime.foreground.run_one(
                kind='recap',
                guild_id=guild_id or interaction.channel.id,
                user_id=interaction.user.id,
                channel_id=interaction.channel.id,
                source_message_id=interaction.id,
                work=work,
                acknowledge=acknowledge,
            )
            if reply is not None:
                await send_chunked_followup(
                    interaction,
                    reply,
                    ephemeral=True,
                    max_len=config.max_discord_message_chars,
                )
        except DuplicateEvent:
            log_with_context(logging.DEBUG, "Suppressed duplicate /recap interaction",
                             **interaction_log_context(interaction))
        except (ValueError, PolicyDenied, ForegroundCancelled) as exc:
            await safe_send_interaction_message(interaction, str(exc), ephemeral=True)
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
                "Suggestion channel is not configured. Please ask an admin to set `discord.suggestion_channel_id` in config.json.",
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
