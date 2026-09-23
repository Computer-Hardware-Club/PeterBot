"""One bounded Discord announcement send using the durable outbox nonce.

The pinned discord.py high-level ``channel.send`` does not expose Discord's
``enforce_nonce`` field. Its HTTPClient still handles rate limits and auth, so
the trusted gateway uses that client for this one narrow API request. The
outbox owns retry/unknown decisions; this function never retries itself.
"""
from __future__ import annotations

import discord
from discord.http import Route


class InvalidAnnouncementReceipt(ValueError):
    """A response cannot be bound to the requested destination."""


async def send_announcement(bot: discord.Client, record: dict,
                            channel: discord.abc.GuildChannel) -> int:
    if record.get("status") != "sending":
        raise ValueError("Announcement must be claimed before sending")
    target = record.get("target_channel_id")
    guild = record.get("guild_id")
    nonce = record.get("nonce")
    content = record.get("content")
    if (type(target) is not int or type(guild) is not int or target <= 0 or guild <= 0
            or getattr(channel, "id", None) != target
            or getattr(getattr(channel, "guild", None), "id", None) != guild
            or not isinstance(nonce, str) or not 1 <= len(nonce) <= 25
            or not isinstance(content, str) or not 1 <= len(content) <= 1800):
        raise ValueError("Announcement destination or payload is invalid")
    route = Route("POST", "/channels/{channel_id}/messages", channel_id=target)
    payload = {
        "content": content,
        "nonce": nonce,
        "enforce_nonce": True,
        "allowed_mentions": {"parse": [], "users": [], "roles": [], "replied_user": False},
        "flags": 4,  # SUPPRESS_EMBEDS; links cannot unexpectedly expand in an announcement.
    }
    result = await bot.http.request(route, json=payload)
    if not isinstance(result, dict):
        raise InvalidAnnouncementReceipt("Discord did not return a message receipt")
    message_id = result.get("id")
    channel_id = result.get("channel_id")
    if (not isinstance(message_id, str) or not message_id.isdecimal()
            or not isinstance(channel_id, str) or not channel_id.isdecimal()
            or int(channel_id) != target or int(message_id) <= 0):
        raise InvalidAnnouncementReceipt("Discord returned an unbound message receipt")
    return int(message_id)
