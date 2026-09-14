"""Bounded, process-local admission control for Discord model requests."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GuardLimits:
    max_concurrent: int = 1
    user_requests_per_minute: int = 3
    guild_requests_per_minute: int = 15
    max_prompt_chars: int = 4000
    allowed_guild_ids: tuple[int, ...] = ()
    allow_dms: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_concurrent",
            "user_requests_per_minute",
            "guild_requests_per_minute",
            "max_prompt_chars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.allowed_guild_ids, tuple) or any(
            type(guild_id) is not int or guild_id <= 0
            for guild_id in self.allowed_guild_ids
        ):
            raise ValueError("allowed_guild_ids must be a tuple of positive integers")
        if type(self.allow_dms) is not bool:
            raise ValueError("allow_dms must be a boolean")


class RequestGuard:
    """Call acquire/release on one event loop; neither call yields or queues work.

    Quotas count admitted requests, including requests that subsequently fail.
    They survive release and expire exactly 60 seconds after admission. This
    state is per process and resets on restart; run one bot process.
    """

    _MAX_QUOTA_ENTRIES = 10_000
    _WINDOW_SECONDS = 60.0

    def __init__(
        self, limits: GuardLimits, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._active_users: set[int] = set()
        self._user_requests: OrderedDict[int, deque[float]] = OrderedDict()
        self._guild_requests: OrderedDict[int, deque[float]] = OrderedDict()

    @staticmethod
    def _prune_identities(
        records: OrderedDict[int, deque[float]], cutoff: float
    ) -> None:
        # Accepted-last-use order makes whole-identity expiry amortized O(1).
        while records:
            identity = next(iter(records))
            if records[identity][-1] > cutoff:
                break
            del records[identity]

    @staticmethod
    def _recent_count(
        records: OrderedDict[int, deque[float]], identity: int, cutoff: float
    ) -> int:
        timestamps = records.get(identity)
        if timestamps is None:
            return 0
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        return len(timestamps)

    @staticmethod
    def _record(
        records: OrderedDict[int, deque[float]], identity: int, now: float
    ) -> None:
        if identity not in records:
            records[identity] = deque()
        records[identity].append(now)
        records.move_to_end(identity)

    def acquire(
        self, *, user_id: int, guild_id: int | None, prompt: str
    ) -> tuple[bool, str | None]:
        if guild_id is None:
            if not self.limits.allow_dms:
                return False, "Please ask me in the club server; DMs are disabled."
        elif (
            self.limits.allowed_guild_ids
            and guild_id not in self.limits.allowed_guild_ids
        ):
            return False, "I'm only available in the configured club server."
        if not isinstance(prompt, str) or not prompt.strip():
            return False, "Please include a question or message."
        if len(prompt) > self.limits.max_prompt_chars:
            return False, f"Please keep your message to {self.limits.max_prompt_chars} characters or fewer."
        if user_id in self._active_users:
            return False, "I'm still working on your previous request. Please wait for it to finish."
        if len(self._active_users) >= self.limits.max_concurrent:
            return False, "I'm busy with another request. Please try again shortly."

        now = self._clock()
        cutoff = now - self._WINDOW_SECONDS
        self._prune_identities(self._user_requests, cutoff)
        self._prune_identities(self._guild_requests, cutoff)
        if self._recent_count(
            self._user_requests, user_id, cutoff
        ) >= self.limits.user_requests_per_minute:
            return False, "You've reached your request limit. Please try again in a minute."
        if guild_id is not None and self._recent_count(
            self._guild_requests, guild_id, cutoff
        ) >= self.limits.guild_requests_per_minute:
            return False, "This server has reached its request limit. Please try again in a minute."

        new_entries = int(user_id not in self._user_requests)
        if guild_id is not None:
            new_entries += int(guild_id not in self._guild_requests)
        if (
            len(self._user_requests) + len(self._guild_requests) + new_entries
            > self._MAX_QUOTA_ENTRIES
        ):
            return False, "I'm handling too many requests right now. Please try again in a minute."

        self._record(self._user_requests, user_id, now)
        if guild_id is not None:
            self._record(self._guild_requests, guild_id, now)
        self._active_users.add(user_id)
        return True, None

    def release(self, *, user_id: int) -> None:
        self._active_users.discard(user_id)
