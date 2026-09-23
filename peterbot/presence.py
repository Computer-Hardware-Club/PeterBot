"""Make a slow turn visibly alive in Discord.

Discord's own typing indicator is already kept alive for as long as a ``typing()``
context is open, but it carries no information and nothing holds it open while a
sandbox task runs in the background. A member asking something hard therefore saw
either nothing, or a "typing..." that stopped, for minutes at a time.

This module owns at most one member-facing message per turn:

* a quick turn never posts one, so ordinary banter does not flicker a placeholder;
* once work passes a threshold the message appears and says what is happening;
* it is edited in place as the work continues and finally *becomes* the answer, so the
  member reads one message instead of a stale placeholder above a reply.

Only elapsed time and a fixed, authenticated worker stage are reported. There
is no invented completion percentage or private reasoning text.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import discord

from .context import split_for_discord
from .logging_utils import log_with_context

log = logging.getLogger(__name__)

# Long enough that a normal reply never flashes a placeholder, short enough that a
# member waiting on real work sees something move.
STATUS_AFTER_SECONDS = 6.0
# Discord rate-limits message edits; a few seconds is also less frantic to read.
MIN_EDIT_SECONDS = 3.0
# How often a running task refreshes its own status line.
PROGRESS_EVERY_SECONDS = 20.0
DEFAULT_SEND_CHARS = 1800
STAGE_LABELS = {
    'queued': 'queued', 'starting': 'starting', 'working': 'working through the request',
    'researching': 'researching', 'running_code': 'running code',
    'reading_files': 'reading files', 'editing_files': 'working on files',
    'checking_memory': 'checking saved context',
    'calculating': 'calculating', 'preparing_answer': 'preparing the answer',
}


class Presence:
    """One status message that can become the final answer.

    Duck-typed on the channel: it only needs ``send``, and ``fetch_message`` if the
    status message has to be recovered after a restart.
    """

    def __init__(self, channel: Any, *, reply_to: Any = None, status_after: float = STATUS_AFTER_SECONDS,
                 now=time.monotonic, max_chars: int = DEFAULT_SEND_CHARS) -> None:
        self.channel = channel
        self.reply_to = reply_to
        self.status_after = status_after
        self.now = now
        self.max_chars = max_chars
        self.message: Optional[Any] = None
        self._poster: Optional[asyncio.Task] = None
        self._last_edit = 0.0
        self._last_text = ''
        self.partial_delivery = False

    @property
    def message_id(self) -> Optional[int]:
        return None if self.message is None else getattr(self.message, 'id', None)

    def _reference(self) -> dict:
        if self.reply_to is None:
            return {}
        return {'reference': discord.MessageReference(
            message_id=getattr(self.reply_to, 'id', None),
            channel_id=getattr(getattr(self.reply_to, 'channel', None), 'id', None),
            guild_id=getattr(getattr(self.reply_to, 'guild', None), 'id', None),
            fail_if_not_exists=False)}

    @classmethod
    def adopt(cls, channel: Any, message: Any, *, now=time.monotonic, max_chars: int = DEFAULT_SEND_CHARS) -> 'Presence':
        """Wrap a status message that is already posted (for example after a restart)."""
        presence = cls(channel, now=now, max_chars=max_chars)
        presence.message = message
        presence._last_text = str(getattr(message, 'content', ''))
        return presence

    async def begin(self) -> None:
        """Start the delayed poster. Harmless to call once per turn."""
        if self._poster is None:
            self._poster = asyncio.create_task(self._post_later())

    async def end(self) -> None:
        if self._poster is not None:
            self._poster.cancel()
            try:
                await self._poster
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - never let cleanup raise
                pass
            self._poster = None

    async def __aenter__(self) -> 'Presence':
        await self.begin()
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        await self.end()
        return False

    async def _post_later(self) -> None:
        try:
            await asyncio.sleep(self.status_after)
            await self.show('on it — thinking this through…')
        except asyncio.CancelledError:
            raise
        except discord.HTTPException:
            log_with_context(logging.WARNING, 'Could not post a working status message',
                             error='HTTPException')
        except Exception:  # noqa: BLE001 - presence is never worth failing a turn over
            log_with_context(logging.WARNING, 'Could not post a working status message')

    async def show(self, text: str, *, force: bool = False) -> Optional[Any]:
        """Post or edit the status line. Throttled unless ``force``."""
        text = text.strip()[:self.max_chars]
        if self.message is None:
            try:
                self.message = await self.channel.send(text, allowed_mentions=discord.AllowedMentions.none(),
                                                       suppress_embeds=True, **self._reference())
            except discord.HTTPException:
                log_with_context(logging.WARNING, 'Could not post a working status message', error='HTTPException')
                return None
            self._last_text, self._last_edit = text, self.now()
            return self.message
        if text == self._last_text:
            return self.message
        if not force and self.now() - self._last_edit < MIN_EDIT_SECONDS:
            return self.message
        try:
            await self.message.edit(content=text)
            self._last_text, self._last_edit = text, self.now()
        except discord.HTTPException:
            # A deleted or uneditable message must not stop the work or the answer.
            log_with_context(logging.WARNING, 'Could not update the working status message',
                             error='HTTPException')
        return self.message

    async def finish(self, text: str) -> bool:
        """Turn the status message into the answer. Returns True if it delivered it.

        False means nothing was sent, or a later chunk failed. In the latter
        case ``partial_delivery`` is true and callers must not replay the first
        chunk. Long replies normally use the durable delivery cursor instead.
        """
        chunks = split_for_discord(text, self.max_chars)
        if self.message is None or not chunks:
            return False
        try:
            await self.message.edit(content=chunks[0])
        except discord.HTTPException:
            # The message was deleted, or we lost permission to edit it. The answer still
            # has to reach the member, so let the caller deliver it normally.
            log_with_context(logging.WARNING, 'Could not turn the working message into the answer',
                             error='HTTPException')
            return False
        self._last_text = chunks[0]
        for chunk in chunks[1:]:
            try:
                await self.channel.send(chunk, allowed_mentions=discord.AllowedMentions.none(),
                                        suppress_embeds=True)
            except discord.HTTPException:
                log_with_context(logging.WARNING, 'Could not send the rest of the answer',
                                 error='HTTPException')
                self.partial_delivery = True
                return False
        return True


def elapsed_label(seconds: float) -> str:
    whole = max(0, int(seconds))
    if whole < 60:
        return f'{whole}s'
    minutes, remainder = divmod(whole, 60)
    return f'{minutes}m {remainder:02d}s'


async def watch_task(presence: Presence, job_id: str, *, interval: float = PROGRESS_EVERY_SECONDS,
                     status: str = 'running', status_getter: Callable[[], str] | None = None) -> None:
    """Keep a long task's status line honest until it is done.

    A trusted getter reads only a fixed stage key, never worker-generated text.
    """
    started = time.monotonic()
    try:
        while True:
            await asyncio.sleep(interval)
            if status_getter is None:
                text = (f'still working — {elapsed_label(time.monotonic() - started)} in '
                        f'({status}). I will post the result here.')
            else:
                stage = status_getter()
                label = STAGE_LABELS.get(stage, 'working')
                text = f'{label} — {elapsed_label(time.monotonic() - started)} elapsed. I will post the result here.'
            await presence.show(text)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a status wobble must never affect the task
        log_with_context(logging.WARNING, 'Task status updater failed', job_id=job_id)
