import asyncio
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

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

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("peterbot")

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

# Optional: control model-side thinking output if supported by the model/server
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MAX_DISCORD_MESSAGE_CHARS = 1800
REMINDER_RETRY_DELAY = timedelta(minutes=5)
CHANNEL_CONTEXT_LIMIT = 14
MAX_CONTEXT_MESSAGE_CHARS = 500

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


async def send_chunked_reply(message: discord.Message, text: str) -> None:
    chunks = split_for_discord(text)
    await message.reply(chunks[0])
    for chunk in chunks[1:]:
        await message.channel.send(chunk)


async def send_chunked_followup(
    interaction: discord.Interaction, text: str, ephemeral: bool = True
) -> None:
    chunks = split_for_discord(text)
    for chunk in chunks:
        await interaction.followup.send(chunk, ephemeral=ephemeral)


def build_system_prompt(context_line: str) -> str:
    return (
        f"{PETER_SYSTEM_PROMPT}\n\n"
        f"Your name is {PETER_NAME}.{context_line}\n"
        "Do not include <think> tags or chain-of-thought. "
        "Avoid repetitive filler and vary wording naturally."
    )


def format_context_message(msg: discord.Message) -> Optional[Dict[str, str]]:
    """Convert a Discord message into an Ollama chat message."""
    content = (msg.content or "").strip()
    if msg.attachments:
        attachment_names = ", ".join(a.filename for a in msg.attachments[:3])
        attachment_text = f"[attachments: {attachment_names}]"
        content = f"{content}\n{attachment_text}".strip()

    if not content:
        return None

    if len(content) > MAX_CONTEXT_MESSAGE_CHARS:
        content = content[:MAX_CONTEXT_MESSAGE_CHARS] + "…"

    is_self = bool(bot.user and msg.author.id == bot.user.id)
    role = "assistant" if is_self else "user"
    author_name = getattr(msg.author, "display_name", msg.author.name)

    if role == "user":
        content = f"{author_name}: {content}"

    return {"role": role, "content": content}


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
    except discord.HTTPException as exc:
        logger.warning("Failed to fetch channel context: %s", exc)
        return []

    context_messages.reverse()
    return context_messages


async def call_ollama_chat(
    prompt_text: str,
    author_name: Optional[str] = None,
    guild_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Call Ollama /api/chat and return assistant content or an error string."""
    await ensure_http_session()
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    context_bits = []
    if guild_name:
        context_bits.append(f"Server: {guild_name}")
    if channel_name:
        context_bits.append(f"Channel: #{channel_name}")
    if author_name:
        context_bits.append(f"User: {author_name}")
    context_line = f" ({', '.join(context_bits)})" if context_bits else ""

    # Append '/no_think' inline (not at start of a line) for models that support it.
    if "/no_think" not in prompt_text:
        prompt_text = f"{prompt_text.rstrip()} /no_think"

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(context_line),
        }
    ]
    if conversation_history:
        messages.extend(conversation_history)

    user_content = f"{author_name}: {prompt_text}" if author_name else prompt_text
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            "think": OLLAMA_THINK,
        },
        "messages": messages,
    }

    try:
        assert http_session is not None
        async with http_session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.warning(
                    "Ollama chat failed with HTTP %s: %s", resp.status, error_text[:500]
                )
                return "Sorry, I couldn't reach the model service right now."

            data = await resp.json(content_type=None)
            msg = data.get("message", {})
            content = msg.get("content")
            if not content:
                # Some older servers return just 'response'
                content = data.get("response")
            content = strip_think_blocks(content)
            return content or "(No response from model)"
    except asyncio.TimeoutError:
        logger.warning("Ollama request timed out")
        return "Sorry, the model took too long to respond."
    except aiohttp.ClientError as exc:
        logger.warning("Ollama connection error: %s", exc)
        return "Sorry, my model backend is unavailable right now."
    except Exception as exc:
        logger.exception("Unexpected Ollama error: %s", exc)
        return "Sorry, something went wrong while generating a response."


# Reminder system
class ReminderManager:
    def __init__(self) -> None:
        self.reminders: List[Dict[str, Any]] = []
        self.reminders_file = "reminders.json"
        self.shutdown_file = "bot_shutdown.json"

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

            with open(self.reminders_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.error("Error saving reminders: %s", exc)

    def load_reminders(self) -> None:
        """Load reminders from JSON file"""
        try:
            if not os.path.exists(self.reminders_file):
                return

            with open(self.reminders_file, "r", encoding="utf-8") as f:
                data = json.load(f)

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
                    logger.warning("Skipping malformed reminder %s: %s", reminder, exc)

            self.reminders = loaded
            self._sort_reminders()
            logger.info("Loaded %s reminders", len(self.reminders))
        except Exception as exc:
            logger.error("Error loading reminders: %s", exc)
            self.reminders = []

    def save_shutdown_time(self) -> None:
        """Save shutdown timestamp"""
        try:
            with open(self.shutdown_file, "w", encoding="utf-8") as f:
                json.dump({"shutdown_time": datetime.now().isoformat()}, f)
        except Exception as exc:
            logger.error("Error saving shutdown time: %s", exc)

    def get_downtime(self) -> Optional[timedelta]:
        """Get downtime duration and clean up file"""
        try:
            if os.path.exists(self.shutdown_file):
                with open(self.shutdown_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                downtime = datetime.now() - datetime.fromisoformat(data["shutdown_time"])
                os.remove(self.shutdown_file)
                return downtime
        except Exception as exc:
            logger.error("Error reading shutdown time: %s", exc)
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
        logger.warning("User %s not found while sending reminder", user_id)
    except discord.HTTPException as exc:
        logger.warning("Failed to fetch user %s: %s", user_id, exc)
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
        logger.info(
            "Cannot DM user %s; dropping reminder", reminder["user_id"]
        )
        return "drop"
    except discord.HTTPException as exc:
        logger.warning(
            "Transient Discord error sending reminder to user %s: %s",
            reminder["user_id"],
            exc,
        )
        return "retry"


async def check_missed_reminders() -> None:
    """Check for reminders that should have been sent while bot was offline."""
    downtime = reminder_manager.get_downtime()
    missed_reminders = reminder_manager.pop_due_reminders()

    if not missed_reminders:
        return

    logger.info("Found %s missed reminders", len(missed_reminders))

    retry_count = 0
    for reminder in missed_reminders:
        status = await deliver_reminder(reminder, missed=True, downtime=downtime)
        if status == "retry":
            reminder_manager.requeue_reminder(reminder)
            retry_count += 1

    reminder_manager.save_reminders()
    if retry_count:
        logger.info("Requeued %s missed reminders for retry", retry_count)


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
        except discord.HTTPException as exc:
            logger.error(
                "Could not fetch suggestion channel with ID %s: %s",
                suggestion_channel_id,
                exc,
            )
            return False

    if channel is None or not hasattr(channel, "send"):
        logger.error("Channel ID %s is not messageable", suggestion_channel_id)
        return False

    embed = discord.Embed(
        title="New Suggestion",
        description=suggestion,
        color=0x00FF00,
        timestamp=datetime.now(),
    )
    embed.add_field(name="Suggested by", value=f"{username} (<@{user_id}>)", inline=False)
    embed.set_footer(text="PSS (Peter's Suggestion System)")

    await channel.send(embed=embed)
    return True


def get_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.error("Environment variable %s must be an integer, got %r", name, raw)
        return None


@bot.event
async def on_ready() -> None:
    global has_initialized, has_synced_commands

    logger.info("Logged in as %s", bot.user)

    if not has_initialized:
        reminder_manager.load_reminders()
        await check_missed_reminders()
        has_initialized = True

    if not has_synced_commands:
        try:
            synced = await bot.tree.sync()
            has_synced_commands = True
            logger.info("Synced %s command(s)", len(synced))
        except Exception as exc:
            logger.error("Failed to sync commands: %s", exc)

    if not reminder_checker.is_running():
        reminder_checker.start()


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore self and bots
    if message.author.bot:
        return

    # Only act when bot is mentioned
    if bot.user and bot.user in message.mentions:
        mention_str = f"<@{bot.user.id}>"
        mention_nick_str = f"<@!{bot.user.id}>"
        content = message.content.replace(mention_str, "").replace(mention_nick_str, "").strip()
        if not content:
            content = "Hello! How can I help?"

        try:
            context_messages = await get_channel_context_messages(
                message.channel, before=message.created_at
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
                    conversation_history=context_messages,
                )
            await send_chunked_reply(message, reply or "(No response)")
        except Exception as exc:
            logger.exception("Error in mention response: %s", exc)
            await send_chunked_reply(
                message,
                "I hit an internal error while generating a reply.",
            )

    # Allow commands to still work
    await bot.process_commands(message)


@bot.event
async def on_disconnect() -> None:
    logger.info("Bot disconnected")
    reminder_manager.save_reminders()


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
        logger.info("Requeued %s reminders due to transient errors", retry_count)


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
        await send_chunked_followup(interaction, reply or "(No response)", ephemeral=True)
    except Exception as exc:
        logger.exception("Error in /ask command: %s", exc)
        await interaction.followup.send(
            "I hit an internal error while talking to the model.",
            ephemeral=True,
        )


# Slash command for suggestions
@bot.tree.command(name="suggest", description="Submit a suggestion to improve the bot")
@discord.app_commands.describe(suggestion="Your suggestion for improving the bot")
async def suggest(interaction: discord.Interaction, suggestion: str) -> None:
    suggestion_channel_id = get_env_int("SUGGESTION_CHANNEL_ID")

    if not suggestion_channel_id:
        await interaction.response.send_message(
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
            await interaction.response.send_message(
                "I couldn't submit your suggestion right now. Please try again later.",
                ephemeral=True,
            )
            return

        await interaction.response.send_message(
            "Thanks for the suggestion. It has been submitted.",
            ephemeral=True,
        )
    except Exception as exc:
        logger.exception("Error in /suggest command: %s", exc)
        await interaction.response.send_message(
            "I couldn't submit your suggestion right now.",
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
            await interaction.response.send_message(
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
            await interaction.response.send_message(
                "❌ Please set a reminder for a future time!",
                ephemeral=True,
            )
            return

        reminder_manager.add_reminder(interaction.user.id, message, remind_time)

        time_str = remind_time.strftime("%A, %b %d, %Y at %I:%M %p")
        await interaction.response.send_message(
            f"✅ Reminder set. I'll remind you about **{message}** on {time_str}.",
            ephemeral=True,
        )

    except Exception as exc:
        logger.exception("Error setting reminder: %s", exc)
        await interaction.response.send_message(
            "❌ I couldn't set that reminder due to an internal error.",
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
    logger.info("Received signal %s. Shutting down gracefully...", signum)
    reminder_manager.save_shutdown_time()
    reminder_manager.save_reminders()
    sys.exit(0)


def register_signal_handlers() -> None:
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def validate_config() -> bool:
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN is not set. Add it to your environment or .env file.")
        return False
    return True


def run_bot() -> None:
    if not validate_config():
        raise SystemExit(1)

    register_signal_handlers()

    try:
        bot.run(DISCORD_TOKEN)
    finally:
        reminder_manager.save_shutdown_time()
        reminder_manager.save_reminders()
        if http_session and not http_session.closed:
            try:
                asyncio.run(close_http_session())
            except Exception:
                logger.debug("Failed to close HTTP session cleanly", exc_info=True)


if __name__ == "__main__":
    run_bot()
