from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import timedelta
from typing import Any

import discord
from discord.ext import commands

from .commands import register_handlers
from .config import AppConfig
from .knowledge import load_knowledge_index
from .logging_utils import configure_logging, log_exception_with_context, log_with_context, set_logging_flags
from .ollama_client import OllamaChatClient
from .reminders import ReminderManager
from .runtime import PeterBotRuntime


def create_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    return commands.Bot(command_prefix="!", intents=intents)


def build_runtime(bot: commands.Bot, config: AppConfig) -> PeterBotRuntime:
    knowledge_index = load_knowledge_index(
        knowledge_file=config.knowledge_file,
        channel_profiles_file=config.channel_profiles_file,
    )
    return PeterBotRuntime(
        bot=bot,
        config=config,
        ollama_client=OllamaChatClient(config),
        reminder_manager=ReminderManager(data_dir=config.data_dir),
        knowledge_index=knowledge_index,
        retry_delay=timedelta(minutes=config.reminder_retry_minutes),
    )


def validate_config(config: AppConfig) -> bool:
    valid = True
    if not config.discord_token:
        log_with_context(
            logging.ERROR,
            "DISCORD_TOKEN is not set. Add it to your environment or .env file.",
        )
        valid = False

    if not config.ollama_base_url.startswith(("http://", "https://")):
        log_with_context(
            logging.ERROR,
            "OLLAMA_BASE_URL must start with http:// or https://",
            ollama_base_url=config.ollama_base_url,
        )
        valid = False

    if not config.ollama_model.strip():
        log_with_context(logging.ERROR, "OLLAMA_MODEL must not be empty")
        valid = False

    if not config.data_dir:
        log_with_context(logging.ERROR, "Resolved DATA_DIR is empty")
        valid = False

    return valid


def register_signal_handlers(runtime: PeterBotRuntime) -> None:
    def signal_handler(signum: int, frame: Any) -> None:
        log_with_context(
            logging.INFO,
            "Received shutdown signal; shutting down gracefully",
            signal=signum,
        )
        runtime.reminder_manager.save_shutdown_time()
        runtime.reminder_manager.save_reminders()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run_bot() -> None:
    try:
        config = AppConfig.from_env()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    configure_logging(config.log_level, config.log_file)
    set_logging_flags(
        user_debug_ids_enabled=config.user_debug_ids_enabled,
        include_traceback_for_warning=config.include_traceback_for_warning,
    )

    if not validate_config(config):
        raise SystemExit(1)

    bot = create_bot()
    runtime = build_runtime(bot, config)
    register_handlers(bot, runtime)
    register_signal_handlers(runtime)

    log_with_context(
        logging.INFO,
        "Starting PeterBot",
        data_dir=config.data_dir,
        ollama_base_url=config.ollama_base_url,
        ollama_model=config.ollama_model,
        ollama_think=config.ollama_think,
        model_profile=config.model_profile.value,
        user_debug_ids=config.user_debug_ids_enabled,
    )

    try:
        bot.run(config.discord_token)
    except Exception:
        log_exception_with_context("Bot terminated unexpectedly in main loop")
        raise
    finally:
        runtime.reminder_manager.save_shutdown_time()
        runtime.reminder_manager.save_reminders()
        try:
            asyncio.run(runtime.ollama_client.close())
        except Exception:
            log_exception_with_context("Failed to close HTTP session cleanly")
