from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import discord

from .config import resolve_data_directory
from .logging_utils import (
    log_exception_with_context,
    log_with_context,
    truncate_for_log,
)


def write_json_atomic(path: str, data: Any) -> None:
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
                log_with_context(
                    logging.WARNING,
                    "Failed cleaning temporary reminder file",
                    temp_path=temp_path,
                )


class ReminderManager:
    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.reminders: List[Dict[str, Any]] = []
        self.data_dir = resolve_data_directory(data_dir)
        self.reminders_file = os.path.join(self.data_dir, "reminders.json")
        self.shutdown_file = os.path.join(self.data_dir, "bot_shutdown.json")
        self.legacy_reminders_file = os.path.abspath(os.path.join(os.getcwd(), "reminders.json"))
        self.legacy_shutdown_file = os.path.abspath(os.path.join(os.getcwd(), "bot_shutdown.json"))

    def _sort_reminders(self) -> None:
        self.reminders.sort(key=lambda reminder: reminder["remind_time"])

    def save_reminders(self) -> None:
        try:
            data = [
                {
                    "user_id": reminder["user_id"],
                    "message": reminder["message"],
                    "remind_time": reminder["remind_time"].isoformat(),
                    "created_at": reminder["created_at"].isoformat(),
                }
                for reminder in self.reminders
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
    ) -> tuple[Optional[Any], Optional[str]]:
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as handle:
                return json.load(handle), primary_path

        if legacy_path != primary_path and os.path.exists(legacy_path):
            log_with_context(
                logging.INFO,
                "Loading legacy data file",
                legacy_path=legacy_path,
                primary_path=primary_path,
            )
            with open(legacy_path, "r", encoding="utf-8") as handle:
                return json.load(handle), legacy_path
        return None, None

    def load_reminders(self) -> None:
        try:
            data, source_path = self._load_json_with_legacy_fallback(
                self.reminders_file,
                self.legacy_reminders_file,
            )
            if data is None:
                return
            if not isinstance(data, list):
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
                except (KeyError, ValueError, TypeError):
                    continue
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
        try:
            data, source_path = self._load_json_with_legacy_fallback(
                self.shutdown_file,
                self.legacy_shutdown_file,
            )
            if not data:
                return None
            if not isinstance(data, dict) or "shutdown_time" not in data:
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
        now = datetime.now()
        due = [reminder for reminder in self.reminders if reminder["remind_time"] <= now]
        self.reminders = [reminder for reminder in self.reminders if reminder["remind_time"] > now]
        return due

    def requeue_reminder(
        self,
        reminder: Dict[str, Any],
        delay: timedelta,
    ) -> None:
        updated = reminder.copy()
        updated["remind_time"] = datetime.now() + delay
        self.reminders.append(updated)
        self._sort_reminders()

    def format_duration(self, duration: timedelta) -> str:
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


async def resolve_user(bot: Any, user_id: int) -> Optional[discord.User]:
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
    manager: ReminderManager,
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
        embed.add_field(
            name="Bot downtime",
            value=manager.format_duration(downtime) if downtime else "Offline duration unavailable",
            inline=False,
        )
        embed.add_field(
            name="How late",
            value=f"{manager.format_duration(delay)} overdue",
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
    bot: Any,
    manager: ReminderManager,
    reminder: Dict[str, Any],
    *,
    missed: bool,
    downtime: Optional[timedelta] = None,
) -> str:
    user = await resolve_user(bot, reminder["user_id"])
    if not user:
        return "drop"

    embed = build_reminder_embed(manager, reminder, missed=missed, downtime=downtime)
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


async def check_missed_reminders(
    bot: Any,
    manager: ReminderManager,
    *,
    retry_delay: timedelta,
) -> None:
    downtime = manager.get_downtime()
    missed_reminders = manager.pop_due_reminders()
    if not missed_reminders:
        return

    log_with_context(
        logging.INFO,
        "Found missed reminders after downtime",
        missed_count=len(missed_reminders),
        downtime=manager.format_duration(downtime) if downtime else None,
    )

    retry_count = 0
    for reminder in missed_reminders:
        status = await deliver_reminder(
            bot,
            manager,
            reminder,
            missed=True,
            downtime=downtime,
        )
        if status == "retry":
            manager.requeue_reminder(reminder, retry_delay)
            retry_count += 1

    manager.save_reminders()
    if retry_count:
        log_with_context(
            logging.INFO,
            "Requeued missed reminders due to transient delivery errors",
            retry_count=retry_count,
        )


def add_one_year(dt: datetime) -> datetime:
    try:
        return dt.replace(year=dt.year + 1)
    except ValueError:
        return dt.replace(year=dt.year + 1, month=2, day=28)


def normalize_2_digit_year(dt: datetime) -> datetime:
    if dt.year < 2000:
        return dt.replace(year=dt.year + 100)
    return dt


def parse_reminder_time(time_str: str, now: Optional[datetime] = None) -> Optional[datetime]:
    if now is None:
        now = datetime.now()

    raw = time_str.strip()
    if not raw:
        return None

    lowered = raw.lower()

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
