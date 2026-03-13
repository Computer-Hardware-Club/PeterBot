import asyncio
import base64
import json
import logging
import os
import re
import signal
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

# Define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Load environment variables
load_dotenv()

def parse_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging() -> logging.Logger:
    raw_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)
    log_format = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        log_file_path = os.path.abspath(os.path.expanduser(log_file))
        log_file_dir = os.path.dirname(log_file_path)
        if log_file_dir:
            os.makedirs(log_file_dir, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                log_file_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
    )

    configured_logger = logging.getLogger("peterbot")
    if raw_level != logging.getLevelName(level):
        configured_logger.warning(
            "Invalid LOG_LEVEL %r; falling back to %s",
            raw_level,
            logging.getLevelName(level),
        )
    configured_logger.info(
        "Logging initialized | level=%s, log_file=%s",
        logging.getLevelName(level),
        log_file if log_file else "<disabled>",
    )
    return configured_logger


logger = configure_logging()
MAX_LOG_CONTEXT_CHARS = 320
USER_DEBUG_IDS_ENABLED = parse_env_bool("USER_DEBUG_IDS_ENABLED", default=True)
INCLUDE_TRACEBACK_FOR_WARNING = parse_env_bool(
    "INCLUDE_TRACEBACK_FOR_WARNING",
    default=False,
)


def truncate_for_log(value: Any, max_chars: int = MAX_LOG_CONTEXT_CHARS) -> str:
    try:
        text = str(value)
    except Exception:
        text = repr(value)

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1]}…"


def format_log_context(**context: Any) -> str:
    parts: List[str] = []
    for key, value in context.items():
        if value is None:
            continue
        parts.append(f"{key}={truncate_for_log(value)}")
    return ", ".join(parts)


def new_debug_id(prefix: str = "ERR") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def build_user_debug_message(base_message: str, debug_id: Optional[str]) -> str:
    if not USER_DEBUG_IDS_ENABLED or not debug_id:
        return base_message
    return f"{base_message}\nDebug ID: `{debug_id}`"


def log_with_context(level: int, message: str, **context: Any) -> None:
    context_text = format_log_context(**context)
    if context_text:
        logger.log(level, "%s | %s", message, context_text)
    else:
        logger.log(level, "%s", message)


def log_exception_with_context(action: str, **context: Any) -> str:
    debug_id = new_debug_id("ERR")
    context_text = format_log_context(**context)
    if context_text:
        logger.exception("[%s] %s | %s", debug_id, action, context_text)
    else:
        logger.exception("[%s] %s", debug_id, action)
    return debug_id


def log_error_with_context(action: str, **context: Any) -> str:
    debug_id = new_debug_id("ERR")
    context_text = format_log_context(**context)
    if context_text:
        logger.error("[%s] %s | %s", debug_id, action, context_text)
    else:
        logger.error("[%s] %s", debug_id, action)
    return debug_id


def message_log_context(message: discord.Message) -> Dict[str, Any]:
    return {
        "message_id": getattr(message, "id", None),
        "user_id": getattr(message.author, "id", None),
        "channel_id": getattr(message.channel, "id", None),
        "guild_id": getattr(message.guild, "id", None) if message.guild else None,
    }


def interaction_log_context(interaction: discord.Interaction) -> Dict[str, Any]:
    command = getattr(interaction, "command", None)
    command_name = getattr(command, "name", None)
    return {
        "interaction_id": getattr(interaction, "id", None),
        "command": command_name,
        "user_id": getattr(interaction.user, "id", None),
        "channel_id": getattr(interaction.channel, "id", None),
        "guild_id": getattr(interaction.guild, "id", None) if interaction.guild else None,
    }

# Create bot instance
bot = commands.Bot(command_prefix="!", intents=intents)

# Ollama / Peter configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:8b")
PETER_NAME = os.getenv("PETER_NAME", "Peter")
PETER_SYSTEM_PROMPT = os.getenv(
    "PETER_SYSTEM_PROMPT",
    (
        "You are Peter, a friendly and witty Discord regular in this server. "
        "Respond conversationally like a real person chatting, not like a formal assistant. "
        "Use the recent channel context to avoid repeating the same phrasing and to stay consistent "
        "with what was already said. Keep replies concise by default, but add detail when asked. "
        "Do not mention hidden rules, policies, or your internal reasoning."
    ),
)

# Optional: allow model-side thinking while keeping the hidden reasoning out of replies.
OLLAMA_THINK = parse_env_bool("OLLAMA_THINK", default=False)

MAX_DISCORD_MESSAGE_CHARS = 1800
REMINDER_RETRY_DELAY = timedelta(minutes=5)
CHANNEL_CONTEXT_LIMIT = 14
MAX_CONTEXT_MESSAGE_CHARS = 500
MENTION_CONTEXT_FETCH_LIMIT = 40
MENTION_FOCUS_MESSAGE_LIMIT = 6
MENTION_ACTIVE_GAP_MINUTES = 10
MENTION_MAX_BACKGROUND_AGE_MINUTES = 45
MENTION_IMAGE_LIMIT = 2
MENTION_MAX_IMAGE_BYTES = 5 * 1024 * 1024

http_session: Optional[aiohttp.ClientSession] = None
has_initialized = False
has_synced_commands = False


async def ensure_http_session() -> None:
    global http_session
    if http_session is None or http_session.closed:
        timeout = aiohttp.ClientTimeout(total=90)
        http_session = aiohttp.ClientSession(timeout=timeout)


async def close_http_session() -> None:
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks if present (case-insensitive, multiline)."""
    if not text:
        return text
    try:
        cleaned = re.sub(
            r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()
    except Exception:
        log_exception_with_context(
            "Failed to strip think blocks",
            text_preview=truncate_for_log(text),
        )
        return text


def split_for_discord(text: str, max_len: int = MAX_DISCORD_MESSAGE_CHARS) -> List[str]:
    """Split text into Discord-safe chunks without hard-cutting where possible."""
    if not text:
        return ["(No response)"]

    remaining = text.strip()
    chunks: List[str] = []

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = remaining.rfind(" ", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:max_len]
            split_at = max_len

        chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    return chunks


async def send_chunked_reply(message: discord.Message, text: str) -> bool:
    chunks = split_for_discord(text)
    try:
        await message.reply(chunks[0])
        for chunk in chunks[1:]:
            await message.channel.send(chunk)
        return True
    except discord.HTTPException:
        debug_id = log_exception_with_context(
            "Failed sending chunked message reply",
            chunk_count=len(chunks),
            **message_log_context(message),
        )
        try:
            await message.channel.send(
                build_user_debug_message(
                    "I couldn't send that response due to a Discord delivery error.",
                    debug_id,
                )
            )
        except discord.HTTPException:
            if INCLUDE_TRACEBACK_FOR_WARNING:
                logger.warning(
                    "[%s] Failed sending fallback channel message",
                    debug_id,
                    exc_info=True,
                )
            else:
                logger.warning("[%s] Failed sending fallback channel message", debug_id)
        return False


async def send_chunked_followup(
    interaction: discord.Interaction, text: str, ephemeral: bool = True
) -> bool:
    chunks = split_for_discord(text)
    try:
        for chunk in chunks:
            await interaction.followup.send(chunk, ephemeral=ephemeral)
        return True
    except discord.HTTPException:
        log_exception_with_context(
            "Failed sending chunked followup",
            chunk_count=len(chunks),
            ephemeral=ephemeral,
            **interaction_log_context(interaction),
        )
        return False


async def safe_send_interaction_message(
    interaction: discord.Interaction, text: str, *, ephemeral: bool = True
) -> bool:
    """Send to initial interaction response if possible, otherwise use followup."""
    try:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=ephemeral)
            return True
        await interaction.response.send_message(text, ephemeral=ephemeral)
        return True
    except discord.InteractionResponded:
        try:
            await interaction.followup.send(text, ephemeral=ephemeral)
            return True
        except discord.HTTPException:
            log_exception_with_context(
                "Failed sending interaction followup after InteractionResponded",
                ephemeral=ephemeral,
                text_preview=truncate_for_log(text),
                **interaction_log_context(interaction),
            )
            return False
    except discord.HTTPException:
        log_exception_with_context(
            "Failed sending interaction message",
            ephemeral=ephemeral,
            text_preview=truncate_for_log(text),
            **interaction_log_context(interaction),
        )
        return False


def build_context_line(
    *,
    author_name: Optional[str] = None,
    guild_name: Optional[str] = None,
    channel_name: Optional[str] = None,
) -> str:
    context_bits = []
    if guild_name:
        context_bits.append(f"Server: {guild_name}")
    if channel_name:
        context_bits.append(f"Channel: #{channel_name}")
    if author_name:
        context_bits.append(f"User: {author_name}")
    return f" ({', '.join(context_bits)})" if context_bits else ""


def build_system_prompt(context_line: str) -> str:
    return (
        f"{PETER_SYSTEM_PROMPT}\n\n"
        f"Your name is {PETER_NAME}.{context_line}\n"
        "Do not include <think> tags or chain-of-thought. "
        "Avoid repetitive filler and vary wording naturally."
    )


def build_mention_system_prompt(context_line: str, focus_note: Optional[str] = None) -> str:
    focus_text = f"\nFocused context: {focus_note}" if focus_note else ""
    return (
        f"{PETER_SYSTEM_PROMPT}\n\n"
        f"Your name is {PETER_NAME}.{context_line}{focus_text}\n"
        "You are replying to a direct mention in a live Discord channel. "
        "Reply to the newest addressed turn first. Use older messages only when they directly "
        "clarify what the current mention refers to, and never drift onto stale channel topics. "
        "If the current mention is vague and the focused context still is not enough, ask a brief "
        "clarifying question instead of guessing. "
        "Write like a real person in chat: relaxed, natural, and slightly varied in cadence. "
        "Do not sound clipped, robotic, or overly punctuated. "
        "Do not include <think> tags or chain-of-thought."
    )


def build_message_content(msg: Any) -> Optional[str]:
    content = (getattr(msg, "content", "") or "").strip()
    attachments = getattr(msg, "attachments", None) or []
    if attachments:
        attachment_names = ", ".join(
            getattr(attachment, "filename", "attachment") for attachment in attachments[:3]
        )
        attachment_text = f"[attachments: {attachment_names}]"
        content = f"{content}\n{attachment_text}".strip()

    if not content:
        return None

    if len(content) > MAX_CONTEXT_MESSAGE_CHARS:
        content = content[:MAX_CONTEXT_MESSAGE_CHARS] + "…"

    return content


def strip_bot_mentions(text: str) -> str:
    stripped = text or ""
    if bot.user:
        mention_str = f"<@{bot.user.id}>"
        mention_nick_str = f"<@!{bot.user.id}>"
        stripped = stripped.replace(mention_str, "").replace(mention_nick_str, "")
    return stripped.strip()


def is_image_attachment(attachment: Any) -> bool:
    content_type = (getattr(attachment, "content_type", "") or "").lower()
    if content_type.startswith("image/"):
        return True

    filename = (getattr(attachment, "filename", "") or "").lower()
    return filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def build_current_mention_prompt_text(message: discord.Message) -> str:
    stripped_content = strip_bot_mentions(getattr(message, "content", "") or "")
    attachments = getattr(message, "attachments", None) or []
    attachment_names = ", ".join(
        getattr(attachment, "filename", "attachment") for attachment in attachments[:3]
    )
    attachment_note = f"[attachments: {attachment_names}]" if attachment_names else ""

    if stripped_content:
        prompt_parts = [part for part in (stripped_content, attachment_note) if part]
        return "\n".join(prompt_parts)

    if attachment_note:
        if any(is_image_attachment(attachment) for attachment in attachments):
            return f"What do you think about this?\n{attachment_note}"
        return f"Can you take a look at this?\n{attachment_note}"

    return "Hello! How can I help?"


async def load_mention_image_payloads(message: discord.Message) -> List[str]:
    images: List[str] = []
    attachments = getattr(message, "attachments", None) or []

    for attachment in attachments:
        if len(images) >= MENTION_IMAGE_LIMIT:
            break
        if not is_image_attachment(attachment):
            continue

        size = getattr(attachment, "size", None)
        if size is not None and size > MENTION_MAX_IMAGE_BYTES:
            log_with_context(
                logging.INFO,
                "Skipping oversized mention image attachment",
                attachment_name=getattr(attachment, "filename", None),
                attachment_size=size,
                max_bytes=MENTION_MAX_IMAGE_BYTES,
                **message_log_context(message),
            )
            continue

        try:
            data = await attachment.read()
        except discord.HTTPException:
            log_exception_with_context(
                "Failed reading mention image attachment",
                attachment_name=getattr(attachment, "filename", None),
                attachment_size=size,
                **message_log_context(message),
            )
            continue

        if not data:
            continue

        if len(data) > MENTION_MAX_IMAGE_BYTES:
            log_with_context(
                logging.INFO,
                "Skipping downloaded mention image attachment exceeding size limit",
                attachment_name=getattr(attachment, "filename", None),
                attachment_size=len(data),
                max_bytes=MENTION_MAX_IMAGE_BYTES,
                **message_log_context(message),
            )
            continue

        images.append(base64.b64encode(data).decode("ascii"))

    return images


def get_message_reference_id(msg: Any) -> Optional[int]:
    reference = getattr(msg, "reference", None)
    if reference is None:
        return None

    message_id = getattr(reference, "message_id", None)
    if message_id is not None:
        return message_id

    resolved = getattr(reference, "resolved", None)
    resolved_id = getattr(resolved, "id", None)
    return resolved_id


def build_context_entry(msg: Any) -> Optional[Dict[str, Any]]:
    """Convert a Discord message into a richer context entry."""
    content = build_message_content(msg)
    if not content:
        return None

    author = getattr(msg, "author", None)
    author_id = getattr(author, "id", None)
    is_self = bool(bot.user and author_id == bot.user.id)
    author_name = getattr(author, "display_name", None) or getattr(author, "name", "Unknown")

    return {
        "message_id": getattr(msg, "id", None),
        "author_id": author_id,
        "author_name": author_name,
        "role": "assistant" if is_self else "user",
        "content": content,
        "created_at": getattr(msg, "created_at", None),
        "reply_to_message_id": get_message_reference_id(msg),
    }


def format_context_message(msg: discord.Message) -> Optional[Dict[str, str]]:
    """Convert a Discord message into an Ollama chat message."""
    entry = build_context_entry(msg)
    if entry is None:
        return None

    content = entry["content"]
    if entry["role"] == "user":
        content = f"{entry['author_name']}: {content}"

    return {"role": entry["role"], "content": content}


async def get_recent_channel_entries(
    channel: Any,
    *,
    limit: int = MENTION_CONTEXT_FETCH_LIMIT,
    before: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Fetch recent channel messages with metadata for mention focus selection."""
    if not hasattr(channel, "history"):
        return []

    recent_entries: List[Dict[str, Any]] = []
    try:
        async for msg in channel.history(limit=limit, before=before, oldest_first=False):
            if msg.author.bot and (not bot.user or msg.author.id != bot.user.id):
                continue

            formatted = build_context_entry(msg)
            if formatted:
                recent_entries.append(formatted)
    except discord.HTTPException:
        debug_id = log_exception_with_context(
            "Failed to fetch recent channel entries",
            channel_id=getattr(channel, "id", None),
            limit=limit,
            before=before,
        )
        logger.warning("[%s] Using empty rich context due to fetch failure", debug_id)
        return []

    recent_entries.reverse()
    return recent_entries


async def get_channel_context_messages(
    channel: Any,
    *,
    limit: int = CHANNEL_CONTEXT_LIMIT,
    before: Optional[datetime] = None,
) -> List[Dict[str, str]]:
    """Fetch recent channel messages and convert them to chat context."""
    if not hasattr(channel, "history"):
        return []

    context_messages: List[Dict[str, str]] = []
    try:
        async for msg in channel.history(limit=limit, before=before, oldest_first=False):
            # Keep humans and this bot, skip other bots to reduce noise.
            if msg.author.bot and (not bot.user or msg.author.id != bot.user.id):
                continue

            formatted = format_context_message(msg)
            if formatted:
                context_messages.append(formatted)
    except discord.HTTPException:
        debug_id = log_exception_with_context(
            "Failed to fetch channel context",
            channel_id=getattr(channel, "id", None),
            limit=limit,
            before=before,
        )
        logger.warning("[%s] Using empty context due to fetch failure", debug_id)
        return []

    context_messages.reverse()
    return context_messages


async def resolve_reply_target_entry(
    message: discord.Message,
    recent_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    target_message_id = get_message_reference_id(message)
    if target_message_id is None:
        return None

    for entry in recent_entries:
        if entry.get("message_id") == target_message_id:
            return entry

    reference = getattr(message, "reference", None)
    resolved = getattr(reference, "resolved", None) if reference is not None else None
    if resolved is not None:
        return build_context_entry(resolved)

    channel = getattr(message, "channel", None)
    if channel is None or not hasattr(channel, "fetch_message"):
        return None

    try:
        referenced_message = await channel.fetch_message(target_message_id)
    except discord.HTTPException:
        debug_id = log_exception_with_context(
            "Failed to fetch referenced message for mention context",
            target_message_id=target_message_id,
            **message_log_context(message),
        )
        logger.warning("[%s] Continuing without explicit reply target entry", debug_id)
        return None

    return build_context_entry(referenced_message)


def format_relative_age(message_time: Optional[datetime], current_time: Optional[datetime]) -> str:
    if message_time is None or current_time is None:
        return "unknown time"

    delta = current_time - message_time
    total_seconds = max(0, int(delta.total_seconds()))
    if total_seconds < 30:
        return "moments ago"
    if total_seconds < 90:
        return "1 minute ago"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        suffix = "" if minutes == 1 else "s"
        return f"{minutes} minute{suffix} ago"
    if current_time.date() == message_time.date():
        hours = total_seconds // 3600
        suffix = "" if hours == 1 else "s"
        return f"{hours} hour{suffix} ago"

    day_delta = (current_time.date() - message_time.date()).days
    if day_delta == 1:
        return "yesterday"

    suffix = "" if day_delta == 1 else "s"
    return f"{day_delta} day{suffix} ago"


def prompt_requires_strong_target(prompt_text: str) -> bool:
    normalized = re.sub(r"\s+", " ", prompt_text or "").strip().lower()
    if not normalized:
        return True

    exact_patterns = (
        r"^what do you think(?: about)?(?: that| this| it| those| these)?\??$",
        r"^what do you make of(?: that| this| it| those| these)?\??$",
        r"^what about(?: that| this| it| those| these)?\??$",
        r"^(?:and )?(?:that|this|it|those|these|them)\??$",
        r"^(?:thoughts|any thoughts|thought)\??$",
        r"^(?:agree|do you agree)\??$",
    )
    if any(re.match(pattern, normalized) for pattern in exact_patterns):
        return True

    tokens = re.findall(r"[a-z0-9']+", normalized)
    if not tokens or len(tokens) > 7:
        return False

    deictic_tokens = {"that", "this", "it", "those", "these", "them"}
    filler_tokens = {
        "what",
        "do",
        "you",
        "think",
        "about",
        "and",
        "any",
        "thoughts",
        "thought",
        "agree",
        "of",
        "make",
    }
    return any(token in deictic_tokens for token in tokens) and all(
        token in deictic_tokens or token in filler_tokens for token in tokens
    )


def find_message_entry_index(entries: List[Dict[str, Any]], message_id: Optional[int]) -> Optional[int]:
    if message_id is None:
        return None

    for index, entry in enumerate(entries):
        if entry.get("message_id") == message_id:
            return index
    return None


def build_recent_tail_entries(
    recent_entries: List[Dict[str, Any]],
    current_time: Optional[datetime],
) -> List[Dict[str, Any]]:
    if not recent_entries:
        return []

    gap_limit = timedelta(minutes=MENTION_ACTIVE_GAP_MINUTES)
    age_limit = timedelta(minutes=MENTION_MAX_BACKGROUND_AGE_MINUTES)
    start = len(recent_entries) - 1

    while start >= 0:
        entry_time = recent_entries[start].get("created_at")
        if entry_time is None:
            break
        if current_time is not None and current_time - entry_time > age_limit:
            start += 1
            break
        if start < len(recent_entries) - 1:
            next_time = recent_entries[start + 1].get("created_at")
            if next_time is None or next_time - entry_time > gap_limit:
                start += 1
                break
        start -= 1

    if start < 0:
        start = 0
    return recent_entries[start:]


def collect_message_cluster(
    entries: List[Dict[str, Any]],
    target_message_id: Optional[int],
    current_time: Optional[datetime],
    *,
    max_age_minutes: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    target_index = find_message_entry_index(entries, target_message_id)
    if target_index is None:
        return [], None

    gap_limit = timedelta(minutes=MENTION_ACTIVE_GAP_MINUTES)
    age_limit = timedelta(minutes=max_age_minutes) if max_age_minutes is not None else None
    target_entry = entries[target_index]
    target_time = target_entry.get("created_at")
    if age_limit is not None and current_time is not None and target_time is not None:
        if current_time - target_time > age_limit:
            return [], None

    start = target_index
    while start > 0:
        previous_entry = entries[start - 1]
        current_entry = entries[start]
        previous_time = previous_entry.get("created_at")
        current_entry_time = current_entry.get("created_at")
        if previous_time is None or current_entry_time is None:
            break
        if current_entry_time - previous_time > gap_limit:
            break
        if (
            age_limit is not None
            and current_time is not None
            and current_time - previous_time > age_limit
        ):
            break
        start -= 1

    end = target_index
    while end < len(entries) - 1:
        current_entry = entries[end]
        next_entry = entries[end + 1]
        current_entry_time = current_entry.get("created_at")
        next_time = next_entry.get("created_at")
        if current_entry_time is None or next_time is None:
            break
        if next_time - current_entry_time > gap_limit:
            break
        if age_limit is not None and current_time is not None and current_time - next_time > age_limit:
            break
        end += 1

    cluster = entries[start : end + 1]
    return cluster, target_index - start


def count_distinct_recent_user_authors(entries: List[Dict[str, Any]], window_size: int) -> int:
    author_keys = []
    for entry in entries[-window_size:]:
        if entry.get("role") != "user":
            continue
        author_key = entry.get("author_id") or entry.get("author_name")
        if author_key is not None:
            author_keys.append(author_key)
    return len(set(author_keys))


def extract_relevance_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    stop_words = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "be",
        "but",
        "do",
        "for",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "thoughts",
        "to",
        "what",
        "when",
        "which",
        "why",
        "with",
        "you",
        "your",
    }
    return [token for token in tokens if len(token) > 2 and token not in stop_words]


def score_focus_candidate(
    entry: Dict[str, Any],
    prompt_tokens: List[str],
    current_time: Optional[datetime],
    position_from_end: int,
) -> int:
    score = 0
    if entry.get("role") == "user":
        score += 15
    elif entry.get("role") == "assistant":
        score += 6

    entry_time = entry.get("created_at")
    if current_time is not None and entry_time is not None:
        age_seconds = max(0, int((current_time - entry_time).total_seconds()))
        score += max(0, 30 - (age_seconds // 60))

    if position_from_end == 0:
        score += 20
    elif position_from_end == 1:
        score += 10

    if prompt_tokens:
        entry_tokens = set(extract_relevance_tokens(entry.get("content", "")))
        score += len(entry_tokens.intersection(prompt_tokens)) * 20

    return score


def entries_are_linked(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_id = left.get("message_id")
    right_id = right.get("message_id")
    if left_id is None or right_id is None:
        return False

    return (
        left.get("reply_to_message_id") == right_id
        or right.get("reply_to_message_id") == left_id
    )


def collect_focus_thread(
    entries: List[Dict[str, Any]],
    target_index: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if target_index is None or target_index < 0 or target_index >= len(entries):
        return [], None

    selected = [entries[target_index]]
    selected_ids = {entries[target_index].get("message_id")}
    target_entry = entries[target_index]

    index = target_index - 1
    while index >= 0 and len(selected) < MENTION_FOCUS_MESSAGE_LIMIT:
        candidate = entries[index]
        if (
            candidate.get("author_id") == target_entry.get("author_id")
            or candidate.get("reply_to_message_id") in selected_ids
            or any(entry.get("reply_to_message_id") == candidate.get("message_id") for entry in selected)
            or any(entries_are_linked(candidate, entry) for entry in selected)
        ):
            selected.insert(0, candidate)
            selected_ids.add(candidate.get("message_id"))
            index -= 1
            continue
        break

    index = target_index + 1
    while index < len(entries) and len(selected) < MENTION_FOCUS_MESSAGE_LIMIT:
        candidate = entries[index]
        if (
            candidate.get("author_id") == target_entry.get("author_id")
            or candidate.get("reply_to_message_id") in selected_ids
            or any(entry.get("reply_to_message_id") == candidate.get("message_id") for entry in selected)
            or any(entries_are_linked(candidate, entry) for entry in selected)
        ):
            selected.append(candidate)
            selected_ids.add(candidate.get("message_id"))
            index += 1
            continue
        break

    resolved_target_index = find_message_entry_index(selected, target_entry.get("message_id"))
    return selected, resolved_target_index


def trim_focus_entries(
    entries: List[Dict[str, Any]],
    target_index: Optional[int],
    max_messages: int = MENTION_FOCUS_MESSAGE_LIMIT,
) -> List[Dict[str, Any]]:
    if len(entries) <= max_messages:
        return entries

    if target_index is None:
        return entries[-max_messages:]

    start = max(0, target_index - (max_messages // 2))
    end = start + max_messages
    if end > len(entries):
        end = len(entries)
        start = end - max_messages

    return entries[start:end]


def build_mention_conversation_history(
    entries: List[Dict[str, Any]],
    current_time: Optional[datetime],
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for entry in entries:
        author_name = entry.get("author_name") or PETER_NAME
        age_text = format_relative_age(entry.get("created_at"), current_time)
        reply_to_message_id = entry.get("reply_to_message_id")
        reply_note = f" [reply to {reply_to_message_id}]" if reply_to_message_id else ""
        content = f"[{age_text}] {author_name}{reply_note}: {entry.get('content', '')}"
        history.append({"role": entry.get("role", "user"), "content": content})
    return history


def build_mention_user_content(author_name: Optional[str], prompt_text: str) -> str:
    speaker = author_name or "User"
    return f"[Current message | now] {speaker}: {prompt_text.strip()}"


def select_mention_focus_target(
    recent_entries: List[Dict[str, Any]],
    current_time: Optional[datetime],
    *,
    prompt_text: str,
    explicit_reply_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, List[Dict[str, Any]], Optional[int]]:
    if explicit_reply_entry is not None:
        target_message_id = explicit_reply_entry.get("message_id")
        cluster, target_index = collect_message_cluster(
            recent_entries,
            target_message_id,
            current_time,
            max_age_minutes=None,
        )
        if not cluster:
            cluster = [explicit_reply_entry]
            target_index = 0
        focus_entries, focus_index = collect_focus_thread(cluster, target_index)
        if focus_entries:
            return explicit_reply_entry, "explicit_reply", focus_entries, focus_index
        return explicit_reply_entry, "explicit_reply", cluster, target_index

    if not recent_entries:
        return None, "no_target", [], None

    local_tail = build_recent_tail_entries(recent_entries, current_time)
    if not local_tail:
        return None, "no_target", [], None

    needs_strong_target = prompt_requires_strong_target(prompt_text)
    if needs_strong_target:
        if count_distinct_recent_user_authors(local_tail, window_size=3) >= 2:
            return None, "ambiguous_recent_turns", [], None

        last_entry = local_tail[-1]
        if last_entry.get("role") == "user":
            focus_entries, focus_index = collect_focus_thread(local_tail, len(local_tail) - 1)
            return last_entry, "immediate_previous_turn", focus_entries, focus_index

        if count_distinct_recent_user_authors(local_tail, window_size=4) >= 2:
            return None, "ambiguous_recent_turns", [], None

        last_reply_target_id = last_entry.get("reply_to_message_id")
        if last_reply_target_id is not None:
            reply_target_index = find_message_entry_index(local_tail, last_reply_target_id)
            if reply_target_index is not None and local_tail[reply_target_index].get("role") == "user":
                focus_entries, focus_index = collect_focus_thread(local_tail, reply_target_index)
                return (
                    local_tail[reply_target_index],
                    "recent_peter_exchange",
                    focus_entries,
                    focus_index,
                )

        for index in range(len(local_tail) - 2, max(-1, len(local_tail) - 4), -1):
            entry = local_tail[index]
            if entry.get("role") == "user":
                focus_entries, focus_index = collect_focus_thread(local_tail, index)
                return entry, "recent_peter_exchange", focus_entries, focus_index

        return None, "no_target", [], None

    prompt_tokens = extract_relevance_tokens(prompt_text)
    scored_candidates: List[Tuple[int, int]] = []
    for index, entry in enumerate(local_tail):
        score = score_focus_candidate(
            entry,
            prompt_tokens,
            current_time,
            position_from_end=(len(local_tail) - 1) - index,
        )
        if prompt_tokens:
            entry_tokens = set(extract_relevance_tokens(entry.get("content", "")))
            if not entry_tokens.intersection(prompt_tokens):
                continue
        scored_candidates.append((score, index))

    if scored_candidates:
        _, best_index = max(scored_candidates)
        target_entry = local_tail[best_index]
        focus_entries, focus_index = collect_focus_thread(local_tail, best_index)
        selection_reason = (
            "lexical_match_user_turn" if target_entry.get("role") == "user" else "lexical_match_recent_exchange"
        )
        return target_entry, selection_reason, focus_entries, focus_index

    last_entry = local_tail[-1]
    if last_entry.get("role") == "user":
        focus_entries, focus_index = collect_focus_thread(local_tail, len(local_tail) - 1)
        return last_entry, "immediate_previous_turn", focus_entries, focus_index

    for index in range(len(local_tail) - 2, -1, -1):
        entry = local_tail[index]
        if entry.get("role") == "user":
            focus_entries, focus_index = collect_focus_thread(local_tail, index)
            return entry, "recent_peter_exchange", focus_entries, focus_index

    return None, "no_target", [], None


def build_mention_focus_note(
    selection_reason: str,
    target_entry: Optional[Dict[str, Any]],
    current_time: Optional[datetime],
) -> Optional[str]:
    if target_entry is None:
        return None

    author_name = target_entry.get("author_name") or "someone"
    age_text = format_relative_age(target_entry.get("created_at"), current_time)
    reason_text = {
        "explicit_reply": "This is the direct reply target",
        "immediate_previous_turn": "This is the immediately preceding human turn",
        "recent_peter_exchange": "This is the latest nearby Peter exchange",
        "lexical_match_user_turn": "This is the recent human turn that best matches the current topic",
        "lexical_match_recent_exchange": "This is the recent Peter exchange that best matches the current topic",
    }.get(selection_reason, "This is the best available focus target")
    return f"{reason_text}: {author_name} said it {age_text}."


def build_mention_context_bundle(
    message: Any,
    prompt_text: str,
    recent_entries: List[Dict[str, Any]],
    *,
    explicit_reply_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_time = getattr(message, "created_at", None)
    target_entry, selection_reason, cluster_entries, target_index = select_mention_focus_target(
        recent_entries,
        current_time,
        prompt_text=prompt_text,
        explicit_reply_entry=explicit_reply_entry,
    )
    trimmed_entries = trim_focus_entries(cluster_entries, target_index)
    focus_note = build_mention_focus_note(selection_reason, target_entry, current_time)
    needs_strong_target = prompt_requires_strong_target(prompt_text)

    clarification_text = None
    if needs_strong_target and target_entry is None:
        clarification_text = "Which message are you asking me about?"

    return {
        "conversation_history": build_mention_conversation_history(trimmed_entries, current_time),
        "user_content": build_mention_user_content(
            getattr(getattr(message, "author", None), "display_name", None),
            prompt_text,
        ),
        "focus_note": focus_note,
        "target_message_id": target_entry.get("message_id") if target_entry else None,
        "target_age_text": format_relative_age(target_entry.get("created_at"), current_time)
        if target_entry
        else None,
        "selection_reason": selection_reason,
        "selected_count": len(trimmed_entries),
        "clarification_text": clarification_text,
        "needs_strong_target": needs_strong_target,
    }


def add_no_think_suffix(text: str, *, allow_thinking: bool = False) -> str:
    if allow_thinking or "/no_think" in text:
        return text
    stripped = text.rstrip()
    if not stripped:
        return text
    return f"{stripped} /no_think"


def build_chat_messages(
    prompt_text: str,
    *,
    author_name: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: str,
    user_content: Optional[str] = None,
    user_images: Optional[List[str]] = None,
    allow_thinking: bool = False,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)

    if user_content is None:
        user_content = f"{author_name}: {prompt_text}" if author_name else prompt_text

    user_message: Dict[str, Any] = {
        "role": "user",
        "content": add_no_think_suffix(user_content, allow_thinking=allow_thinking),
    }
    if user_images:
        user_message["images"] = user_images
    messages.append(user_message)
    return messages


def build_ollama_payload(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    think: bool = False,
) -> Dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "think": think,
        "messages": messages,
    }


def extract_ollama_response_content(data: Dict[str, Any]) -> Optional[str]:
    msg = data.get("message", {}) if isinstance(data, dict) else {}
    content = msg.get("content")
    if not content:
        # Some older servers return just 'response'
        content = data.get("response") if isinstance(data, dict) else None
    return strip_think_blocks(content)


def resolve_data_directory(configured_dir: Optional[str] = None) -> str:
    """Resolve where persistent bot data is stored."""
    default_dir = os.path.dirname(os.path.abspath(__file__))
    raw = configured_dir if configured_dir is not None else os.getenv("PETERBOT_DATA_DIR")
    candidate = raw.strip() if raw and raw.strip() else default_dir
    data_dir = os.path.abspath(os.path.expanduser(candidate))

    try:
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    except OSError as exc:
        logger.error(
            "Failed to create data dir %s (%s). Falling back to %s.",
            data_dir,
            exc,
            default_dir,
        )
        os.makedirs(default_dir, exist_ok=True)
        return default_dir


DATA_DIR = resolve_data_directory()


def write_json_atomic(path: str, data: Any) -> None:
    """Atomically persist JSON to disk."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            delete=False,
        ) as temp_file:
            json.dump(data, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name

        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                if INCLUDE_TRACEBACK_FOR_WARNING:
                    logger.warning(
                        "Failed cleaning temporary file | temp_path=%s",
                        temp_path,
                        exc_info=True,
                    )
                else:
                    logger.warning("Failed cleaning temporary file | temp_path=%s", temp_path)


async def call_ollama_chat(
    prompt_text: str,
    author_name: Optional[str] = None,
    guild_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_content: Optional[str] = None,
    user_images: Optional[List[str]] = None,
) -> str:
    """Call Ollama /api/chat and return assistant content or an error string."""
    await ensure_http_session()
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    request_debug_id = new_debug_id("REQ")

    context_line = build_context_line(
        author_name=author_name,
        guild_name=guild_name,
        channel_name=channel_name,
    )
    resolved_system_prompt = system_prompt or build_system_prompt(context_line)
    messages = build_chat_messages(
        prompt_text,
        author_name=author_name,
        conversation_history=conversation_history,
        system_prompt=resolved_system_prompt,
        user_content=user_content,
        user_images=user_images,
        allow_thinking=OLLAMA_THINK,
    )

    payload = build_ollama_payload(
        OLLAMA_MODEL,
        messages,
        think=OLLAMA_THINK,
    )

    log_with_context(
        logging.DEBUG,
        f"[{request_debug_id}] Sending Ollama chat request",
        url=url,
        model=OLLAMA_MODEL,
        author_name=author_name,
        guild_name=guild_name,
        channel_name=channel_name,
        prompt_preview=truncate_for_log(prompt_text),
        user_content_preview=truncate_for_log(messages[-1]["content"]),
        history_count=len(conversation_history or []),
        user_image_count=len(user_images or []),
    )

    try:
        if http_session is None:
            debug_id = log_error_with_context(
                "HTTP session unavailable before Ollama request",
                request_id=request_debug_id,
                url=url,
                model=OLLAMA_MODEL,
            )
            return build_user_debug_message(
                "Sorry, my model backend failed to initialize.",
                debug_id,
            )

        allow_image_retry = bool(user_images)
        while True:
            async with http_session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if allow_image_retry:
                        allow_image_retry = False
                        retry_messages = [dict(message) for message in payload["messages"]]
                        retry_messages[-1].pop("images", None)
                        payload = {
                            **payload,
                            "messages": retry_messages,
                        }
                        log_with_context(
                            logging.WARNING,
                            "Retrying Ollama chat without images after multimodal failure",
                            request_id=request_debug_id,
                            status=resp.status,
                            model=OLLAMA_MODEL,
                            response_preview=truncate_for_log(error_text, max_chars=500),
                        )
                        continue

                    debug_id = new_debug_id("OLL")
                    log_with_context(
                        logging.ERROR,
                        f"[{debug_id}] Ollama chat failed with non-200 status",
                        request_id=request_debug_id,
                        status=resp.status,
                        response_preview=truncate_for_log(error_text, max_chars=500),
                        url=url,
                        model=OLLAMA_MODEL,
                    )
                    return build_user_debug_message(
                        "Sorry, I couldn't reach the model service right now.",
                        debug_id,
                    )

                data = await resp.json(content_type=None)
                content = extract_ollama_response_content(data)
                return content or "(No response from model)"
    except asyncio.TimeoutError:
        debug_id = log_exception_with_context(
            "Ollama request timed out",
            request_id=request_debug_id,
            url=url,
            model=OLLAMA_MODEL,
            author_name=author_name,
            guild_name=guild_name,
            channel_name=channel_name,
        )
        return build_user_debug_message(
            "Sorry, the model took too long to respond.",
            debug_id,
        )
    except aiohttp.ClientError:
        debug_id = log_exception_with_context(
            "Ollama connection error",
            request_id=request_debug_id,
            url=url,
            model=OLLAMA_MODEL,
            author_name=author_name,
            guild_name=guild_name,
            channel_name=channel_name,
        )
        return build_user_debug_message(
            "Sorry, my model backend is unavailable right now.",
            debug_id,
        )
    except Exception:
        debug_id = log_exception_with_context(
            "Unexpected Ollama error",
            request_id=request_debug_id,
            url=url,
            model=OLLAMA_MODEL,
            author_name=author_name,
            guild_name=guild_name,
            channel_name=channel_name,
        )
        return build_user_debug_message(
            "Sorry, something went wrong while generating a response.",
            debug_id,
        )


# Reminder system
class ReminderManager:
    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.reminders: List[Dict[str, Any]] = []
        self.data_dir = resolve_data_directory(data_dir if data_dir is not None else DATA_DIR)
        self.reminders_file = os.path.join(self.data_dir, "reminders.json")
        self.shutdown_file = os.path.join(self.data_dir, "bot_shutdown.json")
        self.legacy_reminders_file = os.path.abspath(os.path.join(os.getcwd(), "reminders.json"))
        self.legacy_shutdown_file = os.path.abspath(
            os.path.join(os.getcwd(), "bot_shutdown.json")
        )

    def _sort_reminders(self) -> None:
        self.reminders.sort(key=lambda r: r["remind_time"])

    def save_reminders(self) -> None:
        """Save reminders to JSON file"""
        try:
            data = [
                {
                    "user_id": r["user_id"],
                    "message": r["message"],
                    "remind_time": r["remind_time"].isoformat(),
                    "created_at": r["created_at"].isoformat(),
                }
                for r in self.reminders
            ]

            write_json_atomic(self.reminders_file, data)
        except Exception:
            log_exception_with_context(
                "Failed saving reminders",
                reminders_file=self.reminders_file,
                reminder_count=len(self.reminders),
            )

    def _load_json_with_legacy_fallback(
        self,
        primary_path: str,
        legacy_path: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as f:
                return json.load(f), primary_path

        if legacy_path != primary_path and os.path.exists(legacy_path):
            log_with_context(
                logging.INFO,
                "Loading legacy data file",
                legacy_path=legacy_path,
                primary_path=primary_path,
            )
            with open(legacy_path, "r", encoding="utf-8") as f:
                return json.load(f), legacy_path

        return None, None

    def load_reminders(self) -> None:
        """Load reminders from JSON file"""
        try:
            data, source_path = self._load_json_with_legacy_fallback(
                self.reminders_file, self.legacy_reminders_file
            )
            if data is None:
                return

            if not isinstance(data, list):
                log_with_context(
                    logging.ERROR,
                    "Reminder data is malformed; expected a list",
                    source_path=source_path,
                    data_type=type(data).__name__,
                )
                self.reminders = []
                return

            loaded: List[Dict[str, Any]] = []
            for reminder in data:
                try:
                    loaded.append(
                        {
                            "user_id": reminder["user_id"],
                            "message": reminder["message"],
                            "remind_time": datetime.fromisoformat(reminder["remind_time"]),
                            "created_at": datetime.fromisoformat(reminder["created_at"]),
                        }
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    if INCLUDE_TRACEBACK_FOR_WARNING:
                        logger.warning(
                            "Skipping malformed reminder | reminder=%s, error=%s",
                            truncate_for_log(reminder),
                            exc,
                            exc_info=True,
                        )
                    else:
                        logger.warning(
                            "Skipping malformed reminder | reminder=%s, error=%s",
                            truncate_for_log(reminder),
                            exc,
                        )

            self.reminders = loaded
            self._sort_reminders()
            log_with_context(
                logging.INFO,
                "Loaded reminders",
                reminder_count=len(self.reminders),
                source_path=source_path,
            )
        except Exception:
            log_exception_with_context(
                "Failed loading reminders",
                reminders_file=self.reminders_file,
                legacy_reminders_file=self.legacy_reminders_file,
            )
            self.reminders = []

    def save_shutdown_time(self) -> None:
        """Save shutdown timestamp"""
        try:
            write_json_atomic(
                self.shutdown_file,
                {"shutdown_time": datetime.now().isoformat()},
            )
        except Exception:
            log_exception_with_context(
                "Failed saving shutdown time",
                shutdown_file=self.shutdown_file,
            )

    def get_downtime(self) -> Optional[timedelta]:
        """Get downtime duration and clean up file"""
        try:
            data, source_path = self._load_json_with_legacy_fallback(
                self.shutdown_file, self.legacy_shutdown_file
            )
            if not data:
                return None

            if not isinstance(data, dict) or "shutdown_time" not in data:
                log_with_context(
                    logging.ERROR,
                    "Shutdown data is malformed",
                    source_path=source_path,
                    data_type=type(data).__name__,
                )
                return None

            downtime = datetime.now() - datetime.fromisoformat(data["shutdown_time"])
            if source_path and os.path.exists(source_path):
                os.remove(source_path)
            return downtime
        except Exception:
            log_exception_with_context(
                "Failed reading shutdown time",
                shutdown_file=self.shutdown_file,
                legacy_shutdown_file=self.legacy_shutdown_file,
            )
        return None

    def add_reminder(self, user_id: int, message: str, remind_time: datetime) -> None:
        """Add a new reminder"""
        self.reminders.append(
            {
                "user_id": user_id,
                "message": message,
                "remind_time": remind_time,
                "created_at": datetime.now(),
            }
        )
        self._sort_reminders()
        self.save_reminders()

    def pop_due_reminders(self) -> List[Dict[str, Any]]:
        """Pop reminders that are due now."""
        now = datetime.now()
        due = [r for r in self.reminders if r["remind_time"] <= now]
        self.reminders = [r for r in self.reminders if r["remind_time"] > now]
        return due

    def requeue_reminder(
        self, reminder: Dict[str, Any], delay: timedelta = REMINDER_RETRY_DELAY
    ) -> None:
        """Requeue a reminder for retry after a transient failure."""
        updated = reminder.copy()
        updated["remind_time"] = datetime.now() + delay
        self.reminders.append(updated)
        self._sort_reminders()

    def format_duration(self, duration: timedelta) -> str:
        """Format duration in human-readable format."""
        total_seconds = max(0, int(duration.total_seconds()))

        if total_seconds < 60:
            value = total_seconds
            unit = "second"
        elif total_seconds < 3600:
            value = total_seconds // 60
            unit = "minute"
        elif total_seconds < 86400:
            value = total_seconds // 3600
            unit = "hour"
        else:
            value = total_seconds // 86400
            unit = "day"

        suffix = "" if value == 1 else "s"
        return f"{value} {unit}{suffix}"


# Initialize reminder manager
reminder_manager = ReminderManager()


async def resolve_user(user_id: int) -> Optional[discord.User]:
    user = bot.get_user(user_id)
    if user is not None:
        return user

    try:
        return await bot.fetch_user(user_id)
    except discord.NotFound:
        log_with_context(logging.WARNING, "Reminder target user not found", user_id=user_id)
    except discord.HTTPException:
        log_exception_with_context("Failed fetching reminder target user", user_id=user_id)
    return None


def build_reminder_embed(
    reminder: Dict[str, Any],
    *,
    missed: bool,
    downtime: Optional[timedelta] = None,
) -> discord.Embed:
    now = datetime.now()

    if missed:
        delay = now - reminder["remind_time"]
        embed = discord.Embed(
            title="Missed Reminder",
            description=(
                "I was offline when this reminder was due.\n\n"
                f"**Original reminder:** {reminder['message']}"
            ),
            color=0xFF6B6B,
            timestamp=now,
        )
        if downtime:
            embed.add_field(
                name="Bot downtime",
                value=f"{reminder_manager.format_duration(downtime)}",
                inline=False,
            )
        else:
            embed.add_field(
                name="Bot downtime",
                value="Offline duration unavailable",
                inline=False,
            )
        embed.add_field(
            name="How late",
            value=f"{reminder_manager.format_duration(delay)} overdue",
            inline=False,
        )
        embed.add_field(
            name="Original time",
            value=reminder["remind_time"].strftime("%m/%d/%Y %H:%M"),
            inline=False,
        )
    else:
        embed = discord.Embed(
            title="Reminder",
            description=reminder["message"],
            color=0xFFA500,
            timestamp=now,
        )

    embed.set_footer(text="Reminder from PeterBot")
    return embed


async def deliver_reminder(
    reminder: Dict[str, Any],
    *,
    missed: bool,
    downtime: Optional[timedelta] = None,
) -> str:
    """Deliver a reminder and return one of: sent, retry, drop."""
    user = await resolve_user(reminder["user_id"])
    if not user:
        return "drop"

    embed = build_reminder_embed(reminder, missed=missed, downtime=downtime)

    try:
        await user.send(embed=embed)
        return "sent"
    except discord.Forbidden:
        log_with_context(
            logging.INFO,
            "Cannot DM user; dropping reminder",
            user_id=reminder["user_id"],
            reminder_preview=truncate_for_log(reminder.get("message")),
        )
        return "drop"
    except discord.HTTPException:
        log_exception_with_context(
            "Transient Discord error while sending reminder",
            user_id=reminder["user_id"],
            remind_time=reminder.get("remind_time"),
            missed=missed,
        )
        return "retry"


async def check_missed_reminders() -> None:
    """Check for reminders that should have been sent while bot was offline."""
    downtime = reminder_manager.get_downtime()
    missed_reminders = reminder_manager.pop_due_reminders()

    if not missed_reminders:
        return

    log_with_context(
        logging.INFO,
        "Found missed reminders after downtime",
        missed_count=len(missed_reminders),
        downtime=reminder_manager.format_duration(downtime) if downtime else None,
    )

    retry_count = 0
    for reminder in missed_reminders:
        status = await deliver_reminder(reminder, missed=True, downtime=downtime)
        if status == "retry":
            reminder_manager.requeue_reminder(reminder)
            retry_count += 1

    reminder_manager.save_reminders()
    if retry_count:
        log_with_context(
            logging.INFO,
            "Requeued missed reminders due to transient delivery errors",
            retry_count=retry_count,
        )


# Function to send suggestion to a specific channel
async def send_suggestion_to_channel(
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


def get_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        log_with_context(
            logging.ERROR,
            "Environment variable must be an integer",
            variable=name,
            raw_value=raw,
        )
        return None


@bot.event
async def on_ready() -> None:
    global has_initialized, has_synced_commands

    log_with_context(
        logging.INFO,
        "Bot connected to Discord gateway",
        bot_user=bot.user,
        data_dir=DATA_DIR,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_model=OLLAMA_MODEL,
        ollama_think=OLLAMA_THINK,
    )

    if not has_initialized:
        reminder_manager.load_reminders()
        await check_missed_reminders()
        has_initialized = True

    if not has_synced_commands:
        try:
            synced = await bot.tree.sync()
            has_synced_commands = True
            log_with_context(logging.INFO, "Synced slash commands", count=len(synced))
        except Exception:
            log_exception_with_context("Failed syncing slash commands")

    if not reminder_checker.is_running():
        reminder_checker.start()


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore self and bots
    if message.author.bot:
        return

    # Only act when bot is mentioned
    if bot.user and bot.user in message.mentions:
        content = build_current_mention_prompt_text(message)

        try:
            recent_entries = await get_recent_channel_entries(
                message.channel,
                before=message.created_at,
            )
            explicit_reply_entry = await resolve_reply_target_entry(message, recent_entries)
            mention_images = await load_mention_image_payloads(message)
            mention_bundle = build_mention_context_bundle(
                message,
                content,
                recent_entries,
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
                await send_chunked_reply(message, mention_bundle["clarification_text"])
                await bot.process_commands(message)
                return

            context_line = build_context_line(
                author_name=message.author.display_name,
                guild_name=message.guild.name if message.guild else None,
                channel_name=(
                    message.channel.name if isinstance(message.channel, discord.TextChannel) else None
                ),
            )
            async with message.channel.typing():
                reply = await call_ollama_chat(
                    prompt_text=content,
                    author_name=message.author.display_name,
                    guild_name=message.guild.name if message.guild else None,
                    channel_name=(
                        message.channel.name
                        if isinstance(message.channel, discord.TextChannel)
                        else None
                    ),
                    conversation_history=mention_bundle["conversation_history"],
                    system_prompt=build_mention_system_prompt(
                        context_line,
                        focus_note=mention_bundle["focus_note"],
                    ),
                    user_content=mention_bundle["user_content"],
                    user_images=mention_images,
                )
            await send_chunked_reply(message, reply or "(No response)")
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
            )

    # Allow commands to still work
    await bot.process_commands(message)


@bot.event
async def on_disconnect() -> None:
    log_with_context(logging.INFO, "Bot disconnected from Discord gateway")
    reminder_manager.save_reminders()


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
    interaction: discord.Interaction, error: discord.app_commands.AppCommandError
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


# Background task to check for due reminders
@tasks.loop(seconds=30)
async def reminder_checker() -> None:
    due_reminders = reminder_manager.pop_due_reminders()
    if not due_reminders:
        return

    retry_count = 0
    for reminder in due_reminders:
        status = await deliver_reminder(reminder, missed=False)
        if status == "retry":
            reminder_manager.requeue_reminder(reminder)
            retry_count += 1

    reminder_manager.save_reminders()
    if retry_count:
        log_with_context(
            logging.INFO,
            "Requeued due reminders due to transient delivery errors",
            retry_count=retry_count,
        )


@reminder_checker.before_loop
async def before_reminder_checker() -> None:
    await bot.wait_until_ready()


# Slash command for hello
@bot.tree.command(name="hello", description="Say hello to the bot")
async def hello(interaction: discord.Interaction) -> None:
    await interaction.response.send_message("Hello!", ephemeral=True)


# Slash command to query Peter via Ollama
@bot.tree.command(name="ask", description="Ask Peter (Ollama) a question")
@discord.app_commands.describe(prompt="Your question or prompt for Peter")
async def ask(interaction: discord.Interaction, prompt: str) -> None:
    try:
        await interaction.response.defer(ephemeral=True)
        context_messages = await get_channel_context_messages(
            interaction.channel, before=interaction.created_at
        )

        if hasattr(interaction.channel, "typing"):
            async with interaction.channel.typing():
                reply = await call_ollama_chat(
                    prompt_text=prompt,
                    author_name=interaction.user.display_name,
                    guild_name=interaction.guild.name if interaction.guild else None,
                    channel_name=interaction.channel.name if hasattr(interaction.channel, "name") else None,
                    conversation_history=context_messages,
                )
        else:
            reply = await call_ollama_chat(
                prompt_text=prompt,
                author_name=interaction.user.display_name,
                guild_name=interaction.guild.name if interaction.guild else None,
                channel_name=interaction.channel.name if hasattr(interaction.channel, "name") else None,
                conversation_history=context_messages,
            )
        delivered = await send_chunked_followup(
            interaction, reply or "(No response)", ephemeral=True
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


# Slash command for suggestions
@bot.tree.command(name="suggest", description="Submit a suggestion to improve the bot")
@discord.app_commands.describe(suggestion="Your suggestion for improving the bot")
async def suggest(interaction: discord.Interaction, suggestion: str) -> None:
    suggestion_channel_id = get_env_int("SUGGESTION_CHANNEL_ID")

    if not suggestion_channel_id:
        await safe_send_interaction_message(
            interaction,
            "Suggestion channel is not configured. Please ask an admin to set `SUGGESTION_CHANNEL_ID`.",
            ephemeral=True,
        )
        return

    try:
        ok = await send_suggestion_to_channel(
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


# Slash command for reminders
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

        reminder_manager.add_reminder(interaction.user.id, message, remind_time)

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


def add_one_year(dt: datetime) -> datetime:
    """Add one year while handling leap-year edge cases."""
    try:
        return dt.replace(year=dt.year + 1)
    except ValueError:
        # Feb 29 -> Feb 28 in non-leap years
        return dt.replace(year=dt.year + 1, month=2, day=28)


def normalize_2_digit_year(dt: datetime) -> datetime:
    """Map strptime's 19xx values for %y into the 20xx range."""
    if dt.year < 2000:
        return dt.replace(year=dt.year + 100)
    return dt


def parse_reminder_time(time_str: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse supported reminder formats into a datetime object."""
    if now is None:
        now = datetime.now()

    raw = time_str.strip()
    if not raw:
        return None

    lowered = raw.lower()

    # Relative format, e.g. "in 30 minutes", "in 2h", "in 45s"
    relative_match = re.fullmatch(
        r"in\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)",
        lowered,
    )
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)
        if amount <= 0:
            return None

        if unit.startswith(("second", "sec", "s")):
            return now + timedelta(seconds=amount)
        if unit.startswith(("minute", "min", "m")):
            return now + timedelta(minutes=amount)
        if unit.startswith(("hour", "hr", "h")):
            return now + timedelta(hours=amount)
        return now + timedelta(days=amount)

    # "tomorrow" / "tomorrow at 9:15 PM"
    if lowered in {"tomorrow", "tmr", "tmrw"}:
        return (now + timedelta(days=1)).replace(second=0, microsecond=0)

    tomorrow_with_time = re.fullmatch(r"tomorrow(?:\s+at)?\s+(.+)", lowered)
    if tomorrow_with_time:
        time_part = tomorrow_with_time.group(1)
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
            try:
                parsed_time = datetime.strptime(time_part, fmt)
                return (now + timedelta(days=1)).replace(
                    hour=parsed_time.hour,
                    minute=parsed_time.minute,
                    second=0,
                    microsecond=0,
                )
            except ValueError:
                continue

    # Date + time with explicit year
    date_time_with_year = [
        "%m/%d/%Y %H:%M",
        "%m-%d-%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %I:%M%p",
        "%m-%d-%Y %I:%M %p",
        "%m-%d-%Y %I:%M%p",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d %I:%M%p",
        "%m/%d/%y %H:%M",
        "%m-%d-%y %H:%M",
        "%m/%d/%y %I:%M %p",
        "%m/%d/%y %I:%M%p",
        "%m-%d-%y %I:%M %p",
        "%m-%d-%y %I:%M%p",
    ]
    for fmt in date_time_with_year:
        try:
            parsed = datetime.strptime(raw, fmt)
            if "%y" in fmt:
                parsed = normalize_2_digit_year(parsed)
            return parsed
        except ValueError:
            continue

    # Date + time without year
    date_time_without_year = [
        "%m/%d %H:%M",
        "%m-%d %H:%M",
        "%m/%d %I:%M %p",
        "%m/%d %I:%M%p",
        "%m-%d %I:%M %p",
        "%m-%d %I:%M%p",
    ]
    for fmt in date_time_without_year:
        try:
            parsed = datetime.strptime(raw, fmt)
            target = parsed.replace(year=now.year, second=0, microsecond=0)
            if target <= now:
                target = add_one_year(target)
            return target
        except ValueError:
            continue

    # Date-only with explicit year (uses current time)
    date_only_with_year = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y-%m-%d",
        "%m/%d/%y",
        "%m-%d-%y",
    ]
    for fmt in date_only_with_year:
        try:
            parsed = datetime.strptime(raw, fmt)
            if "%y" in fmt:
                parsed = normalize_2_digit_year(parsed)
            return parsed.replace(
                hour=now.hour,
                minute=now.minute,
                second=0,
                microsecond=0,
            )
        except ValueError:
            continue

    # Date-only without year (uses current time, rolls to next year if already passed)
    for fmt in ("%m/%d", "%m-%d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            target = parsed.replace(
                year=now.year,
                hour=now.hour,
                minute=now.minute,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target = add_one_year(target)
            return target
        except ValueError:
            continue

    # Time-only formats (assume today; roll over to tomorrow if already passed)
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            parsed_time = datetime.strptime(raw, fmt)
            target = now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            return target
        except ValueError:
            continue

    return None


# Signal handler for graceful shutdown
def signal_handler(signum: int, frame: Any) -> None:
    log_with_context(
        logging.INFO,
        "Received shutdown signal; shutting down gracefully",
        signal=signum,
    )
    reminder_manager.save_shutdown_time()
    reminder_manager.save_reminders()
    sys.exit(0)


def register_signal_handlers() -> None:
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def validate_config() -> bool:
    valid = True
    if not DISCORD_TOKEN:
        log_with_context(
            logging.ERROR,
            "DISCORD_TOKEN is not set. Add it to your environment or .env file.",
        )
        valid = False

    if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
        log_with_context(
            logging.ERROR,
            "OLLAMA_BASE_URL must start with http:// or https://",
            ollama_base_url=OLLAMA_BASE_URL,
        )
        valid = False

    if not OLLAMA_MODEL.strip():
        log_with_context(logging.ERROR, "OLLAMA_MODEL must not be empty")
        valid = False

    if not os.path.isdir(DATA_DIR):
        log_with_context(
            logging.ERROR,
            "Resolved DATA_DIR does not exist or is not a directory",
            data_dir=DATA_DIR,
        )
        valid = False

    return valid


def run_bot() -> None:
    if not validate_config():
        raise SystemExit(1)

    log_with_context(
        logging.INFO,
        "Starting PeterBot",
        data_dir=DATA_DIR,
        ollama_base_url=OLLAMA_BASE_URL,
        ollama_model=OLLAMA_MODEL,
        ollama_think=OLLAMA_THINK,
        user_debug_ids=USER_DEBUG_IDS_ENABLED,
    )

    register_signal_handlers()

    try:
        bot.run(DISCORD_TOKEN)
    except Exception:
        log_exception_with_context("Bot terminated unexpectedly in main loop")
        raise
    finally:
        reminder_manager.save_shutdown_time()
        reminder_manager.save_reminders()
        if http_session and not http_session.closed:
            try:
                asyncio.run(close_http_session())
            except Exception:
                log_exception_with_context("Failed to close HTTP session cleanly")


if __name__ == "__main__":
    run_bot()
