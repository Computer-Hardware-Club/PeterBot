"""Cheap Discord addressing filter for explicitly enabled channels."""
from __future__ import annotations

from collections import deque
import re
import time

import discord


class AwarenessRouter:
    def __init__(self, *, guild_ids: frozenset[int], channel_ids: frozenset[int],
                 bot_user_id: int, name: str = "Peter", lease_seconds: int = 120,
                 clock=time.monotonic):
        self.guild_ids = guild_ids
        self.channel_ids = channel_ids
        self.bot_user_id = bot_user_id
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.leases: dict[tuple[int, int, int], float] = {}
        self.seen = set()
        self.seen_order = deque(maxlen=4096)
        escaped = re.escape(name)
        self.name = re.compile(rf"^(?:(?:hey|hi|hello|yo|okay|ok|thanks|thank you)[,\s]+)?{escaped}\b(?:[\s,!:?]+|$)", re.I)
        self.trailing_name = re.compile(rf"\n\s*{escaped}[,.!?]?\s*$", re.I)
        self.third_person = re.compile(rf"^{escaped}\s+(?:said|says|was|is|has|had|did|does|went|sent|wrote|told)\b", re.I)

    def _key(self, message):
        return (message.guild.id, message.channel.id, message.author.id)

    def _allowed(self, message) -> bool:
        guild = getattr(message, "guild", None)
        channel = getattr(message, "channel", None)
        if guild is None or channel is None or guild.id not in self.guild_ids:
            return False
        return channel.id in self.channel_ids or getattr(channel, "parent_id", None) in self.channel_ids

    @staticmethod
    def _plain_text(content: str) -> str:
        content = re.sub(r"```[\s\S]*?```", " ", content)
        content = re.sub(r"`[^`]*`", " ", content)
        content = re.sub(r"https?://\S+", " ", content)
        return "\n".join(line for line in content.splitlines() if not line.lstrip().startswith(">")).strip()

    async def addressed(self, message) -> str | None:
        if not self._allowed(message) or getattr(message.author, "bot", False) or getattr(message, "webhook_id", None):
            return None
        message_type = getattr(message, "type", discord.MessageType.default)
        if message_type not in (discord.MessageType.default, discord.MessageType.reply):
            return None
        message_id = getattr(message, "id", None)
        if message_id is not None and message_id in self.seen:
            return None
        content = self._plain_text(getattr(message, "content", "") or "")
        if not content or content.startswith("!"):
            return None
        reference = getattr(message, "reference", None)
        if reference is not None:
            target = getattr(reference, "resolved", None)
            if target is None and getattr(reference, "message_id", None) and hasattr(message.channel, "fetch_message"):
                try:
                    target = await message.channel.fetch_message(reference.message_id)
                except (discord.HTTPException, LookupError):
                    target = None
            if getattr(getattr(target, "author", None), "id", None) == self.bot_user_id:
                return "reply"
        if (self.name.match(content) and not self.third_person.match(content)) or self.trailing_name.search(content):
            return "name"
        # The owner's private task thread is itself the addressing context:
        # a natural follow-up there continues that task even after the lease
        # expired and without a name or reply. Only the bot-created private
        # thread qualifies; the job lookup downstream is owner-bound, so a
        # member of someone else's thread still reaches no session.
        channel = getattr(message, "channel", None)
        is_private = getattr(channel, "is_private", None)
        if callable(is_private) and is_private() \
                and getattr(channel, "owner_id", None) == self.bot_user_id:
            return "thread"
        key = self._key(message)
        if self.leases.get(key, 0) <= self.clock():
            self.leases.pop(key, None)
            return None
        if re.match(r"^(?:bye|goodbye|never\s?mind|stop|ignore that)\b", content, re.I):
            self.leases.pop(key, None)
            return None
        if re.match(r"^(?:@\w+[,:]?\s+|<@!?\d+>)", content):
            self.leases.pop(key, None)
            return None
        # A discourse marker such as "nah," is not another person's name.
        # Only abandon the conversation for a plain name when Discord can
        # resolve it to an actual member of this guild.
        other = re.match(r"^([\w.'-]+)[,:]\s+", content)
        lookup = getattr(message.guild, "get_member_named", None)
        if other and callable(lookup) and lookup(other.group(1)) is not None:
            self.leases.pop(key, None)
            return None
        return "followup"

    def remember(self, message, reason: str) -> None:
        if not self._allowed(message):
            return
        message_id = getattr(message, "id", None)
        if message_id is not None and message_id not in self.seen:
            if len(self.seen_order) == self.seen_order.maxlen:
                self.seen.discard(self.seen_order.popleft())
            self.seen_order.append(message_id)
            self.seen.add(message_id)
        if reason in {"mention", "name", "reply", "followup"}:
            self.leases[self._key(message)] = self.clock() + self.lease_seconds
