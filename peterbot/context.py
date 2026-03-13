from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import discord

from .logging_utils import (
    build_user_debug_message,
    interaction_log_context,
    log_exception_with_context,
    log_with_context,
    logger,
    message_log_context,
    truncate_for_log,
)


def split_for_discord(text: str, max_len: int = 1800) -> List[str]:
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


async def send_chunked_reply(message: discord.Message, text: str, *, max_len: int = 1800) -> bool:
    chunks = split_for_discord(text, max_len=max_len)
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
            logger.warning("[%s] Failed sending fallback channel message", debug_id)
        return False


async def send_chunked_followup(
    interaction: discord.Interaction,
    text: str,
    *,
    ephemeral: bool = True,
    max_len: int = 1800,
) -> bool:
    chunks = split_for_discord(text, max_len=max_len)
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
    interaction: discord.Interaction,
    text: str,
    *,
    ephemeral: bool = True,
) -> bool:
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


def build_message_content(msg: Any, *, max_chars: int = 500) -> Optional[str]:
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

    if len(content) > max_chars:
        content = content[:max_chars] + "…"
    return content


def strip_bot_mentions(text: str, bot_user_id: Optional[int]) -> str:
    stripped = text or ""
    if bot_user_id:
        stripped = stripped.replace(f"<@{bot_user_id}>", "").replace(f"<@!{bot_user_id}>", "")
    return stripped.strip()


def is_image_attachment(attachment: Any) -> bool:
    content_type = (getattr(attachment, "content_type", "") or "").lower()
    if content_type.startswith("image/"):
        return True
    filename = (getattr(attachment, "filename", "") or "").lower()
    return filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))


def build_current_mention_prompt_text(message: discord.Message, *, bot_user_id: Optional[int]) -> str:
    stripped_content = strip_bot_mentions(getattr(message, "content", "") or "", bot_user_id)
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


async def load_mention_image_payloads(
    message: discord.Message,
    *,
    limit: int,
    max_bytes: int,
) -> List[str]:
    images: List[str] = []
    attachments = getattr(message, "attachments", None) or []
    for attachment in attachments:
        if len(images) >= limit:
            break
        if not is_image_attachment(attachment):
            continue

        size = getattr(attachment, "size", None)
        if size is not None and size > max_bytes:
            log_with_context(
                logging.INFO,
                "Skipping oversized mention image attachment",
                attachment_name=getattr(attachment, "filename", None),
                attachment_size=size,
                max_bytes=max_bytes,
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

        if not data or len(data) > max_bytes:
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
    return getattr(resolved, "id", None)


def build_context_entry(
    msg: Any,
    *,
    bot_user_id: Optional[int],
    peter_name: str,
    max_chars: int = 500,
) -> Optional[Dict[str, Any]]:
    content = build_message_content(msg, max_chars=max_chars)
    if not content:
        return None

    author = getattr(msg, "author", None)
    author_id = getattr(author, "id", None)
    is_self = bool(bot_user_id and author_id == bot_user_id)
    author_name = getattr(author, "display_name", None) or getattr(author, "name", None) or peter_name
    return {
        "message_id": getattr(msg, "id", None),
        "author_id": author_id,
        "author_name": author_name,
        "role": "assistant" if is_self else "user",
        "content": content,
        "created_at": getattr(msg, "created_at", None),
        "reply_to_message_id": get_message_reference_id(msg),
    }


def format_context_message(
    msg: Any,
    *,
    bot_user_id: Optional[int],
    peter_name: str,
    max_chars: int = 500,
) -> Optional[Dict[str, str]]:
    entry = build_context_entry(
        msg,
        bot_user_id=bot_user_id,
        peter_name=peter_name,
        max_chars=max_chars,
    )
    if entry is None:
        return None

    content = entry["content"]
    if entry["role"] == "user":
        content = f"{entry['author_name']}: {content}"
    return {"role": entry["role"], "content": content}


async def get_recent_channel_entries(
    channel: Any,
    *,
    bot_user_id: Optional[int],
    peter_name: str,
    limit: int,
    before: Optional[datetime] = None,
    max_chars: int = 500,
) -> List[Dict[str, Any]]:
    if not hasattr(channel, "history"):
        return []

    recent_entries: List[Dict[str, Any]] = []
    try:
        async for msg in channel.history(limit=limit, before=before, oldest_first=False):
            if msg.author.bot and (not bot_user_id or msg.author.id != bot_user_id):
                continue
            formatted = build_context_entry(
                msg,
                bot_user_id=bot_user_id,
                peter_name=peter_name,
                max_chars=max_chars,
            )
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
    bot_user_id: Optional[int],
    peter_name: str,
    limit: int,
    before: Optional[datetime] = None,
    max_chars: int = 500,
) -> List[Dict[str, str]]:
    if not hasattr(channel, "history"):
        return []

    context_messages: List[Dict[str, str]] = []
    try:
        async for msg in channel.history(limit=limit, before=before, oldest_first=False):
            if msg.author.bot and (not bot_user_id or msg.author.id != bot_user_id):
                continue
            formatted = format_context_message(
                msg,
                bot_user_id=bot_user_id,
                peter_name=peter_name,
                max_chars=max_chars,
            )
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
    *,
    bot_user_id: Optional[int],
    peter_name: str,
    max_chars: int = 500,
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
        return build_context_entry(
            resolved,
            bot_user_id=bot_user_id,
            peter_name=peter_name,
            max_chars=max_chars,
        )

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

    return build_context_entry(
        referenced_message,
        bot_user_id=bot_user_id,
        peter_name=peter_name,
        max_chars=max_chars,
    )


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


def find_message_entry_index(entries: Sequence[Dict[str, Any]], message_id: Optional[int]) -> Optional[int]:
    if message_id is None:
        return None
    for index, entry in enumerate(entries):
        if entry.get("message_id") == message_id:
            return index
    return None


def build_recent_tail_entries(
    recent_entries: List[Dict[str, Any]],
    current_time: Optional[datetime],
    *,
    active_gap_minutes: int,
    max_background_age_minutes: int,
) -> List[Dict[str, Any]]:
    if not recent_entries:
        return []

    gap_limit = timedelta(minutes=active_gap_minutes)
    age_limit = timedelta(minutes=max_background_age_minutes)
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
    active_gap_minutes: int,
    max_age_minutes: Optional[int],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    target_index = find_message_entry_index(entries, target_message_id)
    if target_index is None:
        return [], None

    gap_limit = timedelta(minutes=active_gap_minutes)
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
        if age_limit is not None and current_time is not None and current_time - previous_time > age_limit:
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
    return left.get("reply_to_message_id") == right_id or right.get("reply_to_message_id") == left_id


def collect_focus_thread(
    entries: List[Dict[str, Any]],
    target_index: Optional[int],
    *,
    max_messages: int,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    if target_index is None or target_index < 0 or target_index >= len(entries):
        return [], None

    selected = [entries[target_index]]
    selected_ids = {entries[target_index].get("message_id")}
    target_entry = entries[target_index]

    index = target_index - 1
    while index >= 0 and len(selected) < max_messages:
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
    while index < len(entries) and len(selected) < max_messages:
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
    *,
    max_messages: int,
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


def append_recent_assistant_tail(
    focus_entries: List[Dict[str, Any]],
    recent_entries: Sequence[Dict[str, Any]],
    *,
    max_assistant_entries: int,
) -> List[Dict[str, Any]]:
    if max_assistant_entries <= 0:
        return focus_entries

    existing_ids = {entry.get("message_id") for entry in focus_entries}
    focus_ids = {entry.get("message_id") for entry in focus_entries if entry.get("message_id") is not None}
    assistant_tail = [
        entry
        for entry in recent_entries
        if entry.get("role") == "assistant"
        and entry.get("message_id") not in existing_ids
        and (
            entry.get("reply_to_message_id") in focus_ids
            or any(existing.get("reply_to_message_id") == entry.get("message_id") for existing in focus_entries)
        )
    ]
    if not assistant_tail:
        return focus_entries

    selected_tail = assistant_tail[-max_assistant_entries:]
    combined = focus_entries + selected_tail
    combined.sort(key=lambda entry: entry.get("created_at") or datetime.min)
    return combined


def build_mention_conversation_history(
    entries: Sequence[Dict[str, Any]],
    current_time: Optional[datetime],
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for entry in entries:
        author_name = entry.get("author_name") or "Peter"
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
    focus_message_limit: int,
    active_gap_minutes: int,
    max_background_age_minutes: int,
    explicit_reply_entry: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, List[Dict[str, Any]], Optional[int]]:
    if explicit_reply_entry is not None:
        target_message_id = explicit_reply_entry.get("message_id")
        cluster, target_index = collect_message_cluster(
            recent_entries,
            target_message_id,
            current_time,
            active_gap_minutes=active_gap_minutes,
            max_age_minutes=None,
        )
        if not cluster:
            cluster = [explicit_reply_entry]
            target_index = 0
        focus_entries, focus_index = collect_focus_thread(
            cluster,
            target_index,
            max_messages=focus_message_limit,
        )
        if focus_entries:
            return explicit_reply_entry, "explicit_reply", focus_entries, focus_index
        return explicit_reply_entry, "explicit_reply", cluster, target_index

    if not recent_entries:
        return None, "no_target", [], None

    local_tail = build_recent_tail_entries(
        recent_entries,
        current_time,
        active_gap_minutes=active_gap_minutes,
        max_background_age_minutes=max_background_age_minutes,
    )
    if not local_tail:
        return None, "no_target", [], None

    needs_strong_target = prompt_requires_strong_target(prompt_text)
    if needs_strong_target:
        if count_distinct_recent_user_authors(local_tail, window_size=3) >= 2:
            return None, "ambiguous_recent_turns", [], None

        last_entry = local_tail[-1]
        if last_entry.get("role") == "user":
            focus_entries, focus_index = collect_focus_thread(
                local_tail,
                len(local_tail) - 1,
                max_messages=focus_message_limit,
            )
            return last_entry, "immediate_previous_turn", focus_entries, focus_index

        if count_distinct_recent_user_authors(local_tail, window_size=4) >= 2:
            return None, "ambiguous_recent_turns", [], None

        last_reply_target_id = last_entry.get("reply_to_message_id")
        if last_reply_target_id is not None:
            reply_target_index = find_message_entry_index(local_tail, last_reply_target_id)
            if reply_target_index is not None and local_tail[reply_target_index].get("role") == "user":
                focus_entries, focus_index = collect_focus_thread(
                    local_tail,
                    reply_target_index,
                    max_messages=focus_message_limit,
                )
                return local_tail[reply_target_index], "recent_peter_exchange", focus_entries, focus_index

        for index in range(len(local_tail) - 2, max(-1, len(local_tail) - 4), -1):
            entry = local_tail[index]
            if entry.get("role") == "user":
                focus_entries, focus_index = collect_focus_thread(
                    local_tail,
                    index,
                    max_messages=focus_message_limit,
                )
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
        focus_entries, focus_index = collect_focus_thread(
            local_tail,
            best_index,
            max_messages=focus_message_limit,
        )
        selection_reason = (
            "lexical_match_user_turn"
            if target_entry.get("role") == "user"
            else "lexical_match_recent_exchange"
        )
        return target_entry, selection_reason, focus_entries, focus_index

    last_entry = local_tail[-1]
    if last_entry.get("role") == "user":
        focus_entries, focus_index = collect_focus_thread(
            local_tail,
            len(local_tail) - 1,
            max_messages=focus_message_limit,
        )
        return last_entry, "immediate_previous_turn", focus_entries, focus_index

    for index in range(len(local_tail) - 2, -1, -1):
        entry = local_tail[index]
        if entry.get("role") == "user":
            focus_entries, focus_index = collect_focus_thread(
                local_tail,
                index,
                max_messages=focus_message_limit,
            )
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
    focus_message_limit: int,
    active_gap_minutes: int,
    max_background_age_minutes: int,
    assistant_tail_limit: int,
    explicit_reply_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_time = getattr(message, "created_at", None)
    target_entry, selection_reason, cluster_entries, target_index = select_mention_focus_target(
        recent_entries,
        current_time,
        prompt_text=prompt_text,
        focus_message_limit=focus_message_limit,
        active_gap_minutes=active_gap_minutes,
        max_background_age_minutes=max_background_age_minutes,
        explicit_reply_entry=explicit_reply_entry,
    )
    trimmed_entries = trim_focus_entries(
        cluster_entries,
        target_index,
        max_messages=focus_message_limit,
    )
    packed_entries = append_recent_assistant_tail(
        trimmed_entries,
        recent_entries,
        max_assistant_entries=assistant_tail_limit,
    )
    focus_note = build_mention_focus_note(selection_reason, target_entry, current_time)
    needs_strong_target = prompt_requires_strong_target(prompt_text)
    clarification_text = None
    if needs_strong_target and target_entry is None:
        clarification_text = "Which message are you asking me about?"

    return {
        "conversation_history": build_mention_conversation_history(packed_entries, current_time),
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
        "selected_count": len(packed_entries),
        "clarification_text": clarification_text,
        "needs_strong_target": needs_strong_target,
    }


def build_recap_history(
    entries: Sequence[Dict[str, Any]],
    current_time: Optional[datetime],
) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for entry in entries:
        author_name = entry.get("author_name") or "Unknown"
        age_text = format_relative_age(entry.get("created_at"), current_time)
        content = f"[{age_text}] {author_name}: {entry.get('content', '')}"
        history.append({"role": entry.get("role", "user"), "content": content})
    return history
