from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import discord
from discord.ext import commands

from .commands import register_handlers
from .config import AppConfig
from .knowledge import load_knowledge_index
from .llama_cpp_client import LlamaCppChatClient
from .logging_utils import configure_logging, log_exception_with_context, log_with_context, set_logging_flags
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
        llm_client=LlamaCppChatClient(config),
        reminder_manager=ReminderManager(data_dir=config.data_dir),
        knowledge_index=knowledge_index,
        retry_delay=timedelta(minutes=config.reminder_retry_minutes),
    )


def validate_config(config: AppConfig) -> bool:
    try:
        config.validate()
    except ValueError as exc:
        log_with_context(logging.ERROR, "Invalid configuration", error=str(exc), config_path=config.config_path)
        return False

    if config.llama_server.enabled and not Path(config.llama_server.model_path or "").is_file():
        log_with_context(
            logging.ERROR,
            "Bundled llama.cpp mode requires a readable GGUF model file",
            model_path=config.llama_server.model_path,
        )
        return False

    return True


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
        config = AppConfig.load()
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
        config_path=config.config_path,
        data_dir=config.data_dir,
        inference_base_url=config.inference.base_url,
        inference_model=config.inference.model,
        inference_timeout_seconds=config.inference.timeout_seconds,
        bundled_llama_server=config.llama_server.enabled,
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
            asyncio.run(runtime.llm_client.close())
        except Exception:
            log_exception_with_context("Failed to close HTTP session cleanly")
