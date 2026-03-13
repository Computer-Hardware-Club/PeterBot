from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .config import AppConfig
from .knowledge import KnowledgeIndex
from .ollama_client import OllamaChatClient
from .reminders import ReminderManager


@dataclass
class PeterBotRuntime:
    bot: Any
    config: AppConfig
    ollama_client: OllamaChatClient
    reminder_manager: ReminderManager
    knowledge_index: KnowledgeIndex
    retry_delay: timedelta
    has_initialized: bool = False
    has_synced_commands: bool = False
