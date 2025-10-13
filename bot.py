import discord
from discord.ext import commands, tasks
import os  # Operating system interface
import asyncio  # Asynchronous I/O for handling concurrent operations
import json  # JSON data format handling for saving/loading reminders
import signal  # Signal handling for graceful bot shutdown
import sys  # System-specific parameters and functions for exit handling
from datetime import datetime, timedelta  # Date and time manipulation for reminders
from dotenv import load_dotenv  # Load environment variables from .env file
import aiohttp  # HTTP client for calling Ollama
import re  # Regex for stripping <think> blocks

#define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Load environment variables
load_dotenv()

#create bot instance
bot = commands.Bot(command_prefix='!', intents=intents)

# Ollama / Peter configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
PETER_NAME = os.getenv('PETER_NAME', 'Peter')
PETER_SYSTEM_PROMPT = os.getenv('PETER_SYSTEM_PROMPT')

http_session = None

# Optional: control model-side thinking output if supported by the model/server
OLLAMA_THINK = os.getenv('OLLAMA_THINK', 'false').lower() in ('1', 'true', 'yes', 'on')

async def ensure_http_session():
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90))

def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks if present (case-insensitive, multiline)."""
    if not text:
        return text
    try:
        cleaned = re.sub(r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
        return cleaned.strip()
    except Exception:
        return text

async def call_ollama_chat(prompt_text, author_name=None, guild_name=None, channel_name=None):
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
    context_line = (" (" + ", ".join(context_bits) + ")") if context_bits else ""

    # Append '/no_think' inline (not at start of a line) per Reddit guidance for Qwen3
    # Avoid duplication if user already included it
    if "/no_think" not in prompt_text:
        prompt_text = f"{prompt_text.rstrip()} /no_think"

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {
            # Some models (e.g., Qwen reasoning variants) honor a 'think' flag
            "think": OLLAMA_THINK,
        },
        "messages": [
            {
                "role": "system",
                "content": f"{PETER_SYSTEM_PROMPT}\n\nYour name is {PETER_NAME}.{context_line}\nDo not include <think> tags or chain-of-thought. Provide only the final answer.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
    }

    try:
        async with http_session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return f"Sorry, I couldn't reach the model (HTTP {resp.status})."
            data = await resp.json()
            msg = data.get("message", {})
            content = msg.get("content")
            if not content:
                # Some older servers return just 'response'
                content = data.get("response")
            content = strip_think_blocks(content)
            return content or "(No response from model)"
    except Exception as e:
        return f"Sorry, my brain (LLM) is unavailable right now. ({e})"

# Reminder system
class ReminderManager:
    def __init__(self):
        self.reminders = []
        self.reminders_file = "reminders.json"
        self.shutdown_file = "bot_shutdown.json"
    
    def save_reminders(self):
        """Save reminders to JSON file"""
        try:
            data = [{
                'user_id': r['user_id'],
                'message': r['message'],
                'remind_time': r['remind_time'].isoformat(),
                'created_at': r['created_at'].isoformat()
            } for r in self.reminders]
            
            with open(self.reminders_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving reminders: {e}")
    
    def load_reminders(self):
        """Load reminders from JSON file"""
        try:
            if os.path.exists(self.reminders_file):
                with open(self.reminders_file, 'r') as f:
                    data = json.load(f)
                
                self.reminders = [{
                    'user_id': r['user_id'],
                    'message': r['message'],
                    'remind_time': datetime.fromisoformat(r['remind_time']),
                    'created_at': datetime.fromisoformat(r['created_at'])
                } for r in data]
                
                print(f"Loaded {len(self.reminders)} reminders")
        except Exception as e:
            print(f"Error loading reminders: {e}")
            self.reminders = []
    
    def save_shutdown_time(self):
        """Save shutdown timestamp"""
        try:
            with open(self.shutdown_file, 'w') as f:
                json.dump({'shutdown_time': datetime.now().isoformat()}, f)
        except Exception as e:
            print(f"Error saving shutdown time: {e}")
    
    def get_downtime(self):
        """Get downtime duration and clean up file"""
        try:
            if os.path.exists(self.shutdown_file):
                with open(self.shutdown_file, 'r') as f:
                    data = json.load(f)
                downtime = datetime.now() - datetime.fromisoformat(data['shutdown_time'])
                os.remove(self.shutdown_file)
                return downtime
        except Exception as e:
            print(f"Error reading shutdown time: {e}")
        return None
    
    def add_reminder(self, user_id, message, remind_time):
        """Add a new reminder"""
        self.reminders.append({
            'user_id': user_id,
            'message': message,
            'remind_time': remind_time,
            'created_at': datetime.now()
        })
        self.save_reminders()
    
    def get_due_reminders(self):
        """Get reminders that are due"""
        now = datetime.now()
        due = [r for r in self.reminders if r['remind_time'] <= now]
        self.reminders = [r for r in self.reminders if r['remind_time'] > now]
        return due
    
    def format_duration(self, duration):
        """Format duration in human-readable format"""
        hours = duration.total_seconds() / 3600
        if hours < 1:
            return f"{int(duration.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{int(hours)} hours"
        else:
            return f"{int(hours / 24)} days"

# Initialize reminder manager
reminder_manager = ReminderManager()

async def check_missed_reminders():
    """Check for reminders that should have been sent while bot was offline"""
    downtime = reminder_manager.get_downtime()
    missed_reminders = reminder_manager.get_due_reminders()
    
    if missed_reminders:
        print(f"Found {len(missed_reminders)} missed reminders")
        
        for reminder in missed_reminders:
            try:
                user = bot.get_user(reminder['user_id'])
                if user:
                    delay = datetime.now() - reminder['remind_time']
                    
                    embed = discord.Embed(
                        title="😔 Missed Reminder",
                        description=f"I'm sorry, but I was offline when your reminder was due.\n\n**Original reminder:** {reminder['message']}",
                        color=0xff6b6b,
                        timestamp=datetime.now()
                    )
                    
                    if downtime:
                        embed.add_field(name="🔧 Bot downtime", value=f"I was offline for exactly {reminder_manager.format_duration(downtime)}", inline=False)
                    else:
                        embed.add_field(name="🔧 Bot downtime", value="I was offline (exact duration unknown)", inline=False)
                    
                    embed.add_field(name="⏰ How late", value=f"Your reminder was {reminder_manager.format_duration(delay)} overdue", inline=False)
                    embed.add_field(name="📅 Original time", value=reminder['remind_time'].strftime("%m/%d/%y %H:%M"), inline=False)
                    embed.set_footer(text="Sorry for the inconvenience! 😔")
                    
                    await user.send(embed=embed)
                    
            except Exception as e:
                print(f"Failed to send apology to user {reminder['user_id']}: {e}")
        
        reminder_manager.save_reminders()

# Function to send suggestion to a specific channel
async def send_suggestion_to_channel(bot, suggestion_channel_id, user_id, username, suggestion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    channel = bot.get_channel(suggestion_channel_id)
    
    if channel:
        embed = discord.Embed(
            title="💡 New Suggestion",
            description=suggestion,
            color=0x00ff00,  # Green color
            timestamp=datetime.now()
        )
        embed.add_field(name="Suggested by", value=f"{username}", inline=False)
        embed.set_footer(text="PSS (Peter's Suggestion System)")
        
        await channel.send(embed=embed)
    else:
        print(f"Could not find channel with ID: {suggestion_channel_id}")

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')
    
    # Load reminders and check for missed ones
    reminder_manager.load_reminders()
    await check_missed_reminders()
    
    try:
        synced = await bot.tree.sync()
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Failed to sync commands: {e}')
    
    reminder_checker.start()

@bot.event
async def on_message(message: discord.Message):
    # Ignore self and bots
    if message.author.bot:
        return

    # Only act when bot is mentioned
    mentioned_ids = [m.id for m in message.mentions]
    if bot.user and bot.user.id in mentioned_ids:
        # Build the prompt by removing the mention text
        mention_str = f"<@{bot.user.id}>"
        mention_nick_str = f"<@!{bot.user.id}>"  # Discord sometimes includes '!'
        content = message.content.replace(mention_str, '').replace(mention_nick_str, '').strip()
        if not content:
            content = "Hello! How can I help?"

        try:
            reply = await call_ollama_chat(
                prompt_text=content,
                author_name=message.author.display_name,
                guild_name=message.guild.name if message.guild else None,
                channel_name=message.channel.name if isinstance(message.channel, discord.TextChannel) else None,
            )

            # Keep replies reasonably short to avoid flooding
            if reply and len(reply) > 1800:
                reply = reply[:1800] + "…"

            await message.reply(reply or "(No response)")
        except Exception as e:
            await message.reply(f"There was an error talking to the model: {e}")

    # Allow commands to still work
    await bot.process_commands(message)

@bot.event
async def on_disconnect():
    print("Bot is disconnecting...")
    reminder_manager.save_shutdown_time()
    reminder_manager.save_reminders()

# Background task to check for due reminders
@tasks.loop(seconds=30)
async def reminder_checker():
    due_reminders = reminder_manager.get_due_reminders()
    
    for reminder in due_reminders:
        try:
            user = bot.get_user(reminder['user_id'])
            if user:
                embed = discord.Embed(
                    title="⏰ Reminder",
                    description=reminder['message'],
                    color=0xffa500,
                    timestamp=datetime.now()
                )
                embed.set_footer(text="Reminder from PeterBot")
                await user.send(embed=embed)
        except Exception as e:
            print(f"Failed to send reminder to user {reminder['user_id']}: {e}")
    
    if due_reminders:
        reminder_manager.save_reminders()

# Slash command for hello
@bot.tree.command(name="hello", description="Say hello to the bot")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message('Hello!', ephemeral=True)

# Slash command to query Peter via Ollama
@bot.tree.command(name="ask", description="Ask Peter (Ollama) a question")
@discord.app_commands.describe(prompt="Your question or prompt for Peter")
async def ask(interaction: discord.Interaction, prompt: str):
    try:
        await interaction.response.defer(ephemeral=True)
        reply = await call_ollama_chat(
            prompt_text=prompt,
            author_name=interaction.user.display_name,
            guild_name=interaction.guild.name if interaction.guild else None,
            channel_name=interaction.channel.name if hasattr(interaction.channel, 'name') else None,
        )
        if reply and len(reply) > 1800:
            reply = reply[:1800] + "…"
        await interaction.followup.send(reply or "(No response)", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"There was an error talking to the model: {e}", ephemeral=True)

# (sync command removed by request)

# Slash command for suggestions
@bot.tree.command(name="suggest", description="Submit a suggestion to improve the bot")
@discord.app_commands.describe(suggestion="Your suggestion for improving the bot")
async def suggest(interaction: discord.Interaction, suggestion: str):
    # Get suggestion channel ID from environment variable
    suggestion_channel_id = int(os.getenv('SUGGESTION_CHANNEL_ID', 0))
    
    if suggestion_channel_id:
        # Send suggestion to the designated channel
        await send_suggestion_to_channel(bot, suggestion_channel_id, interaction.user.id, interaction.user.display_name, suggestion)
        
        # Send confirmation message
        await interaction.response.send_message(f"Thanks for the suggestion! I will get to work on that! 💡", ephemeral=True)
    else:
        await interaction.response.send_message("Suggestion system is broken, ask Scott to fix it.", ephemeral=True)

# Slash command for reminders
@bot.tree.command(name="remindme", description="Set a reminder for yourself")
@discord.app_commands.describe(
    message="What you want to be reminded about",
    time="When to remind you (supports many formats: '10/08/2025 14:30', '2:30 PM', 'tomorrow', etc.)"
)
async def remindme(interaction: discord.Interaction, message: str, time: str):
    try:
        # Parse the time input
        remind_time = parse_reminder_time(time)
        
        if remind_time is None:
            await interaction.response.send_message(
                "❌ Invalid time format. Here are the supported formats:\n\n"
                "**Date + Time:**\n"
                "• `10/08/2025 14:30` (4-digit year)\n"
                "• `10/08/25 14:30` (2-digit year)\n"
                "• `10-08-2025 14:30` (dash format)\n"
                "• `2025-10-08 14:30` (ISO format)\n\n"
                "**Date Only:**\n"
                "• `10/08/2025` (uses current time)\n"
                "• `10/08/25`\n\n"
                "**Time Only:**\n"
                "• `14:30` (24-hour format)\n"
                "• `2:30 PM` (12-hour format)\n"
                "• `2:30PM`", 
                ephemeral=True
            )
            return
        
        if remind_time <= datetime.now():
            await interaction.response.send_message(
                "❌ Please set a reminder for a future time!", 
                ephemeral=True
            )
            return
        
        # Add reminder using the manager
        reminder_manager.add_reminder(interaction.user.id, message, remind_time)
        
        # Format the reminder time for display
        time_str = remind_time.strftime("%m/%d/%y @ %H:%M")
        
        await interaction.response.send_message(
            f"✅ Reminder set! I'll remind you about **{message}** on {time_str}", 
            ephemeral=True
        )
        
    except Exception as e:
        await interaction.response.send_message(
            f"❌ Error setting reminder: {str(e)}", 
            ephemeral=True
        )

def parse_reminder_time(time_str):
    """Parse various date/time formats into a datetime object"""
    time_str = time_str.strip()
    now = datetime.now()
    
    # List of supported formats to try
    formats = [
        # 4-digit year formats
        "%m/%d/%Y %H:%M",      # 10/08/2025 14:30
        "%m-%d-%Y %H:%M",      # 10-08-2025 14:30
        "%Y-%m-%d %H:%M",      # 2025-10-08 14:30
        "%m/%d/%Y",            # 10/08/2025 (time defaults to current time)
        "%m-%d-%Y",            # 10-08-2025
        "%Y-%m-%d",            # 2025-10-08
        
        # 2-digit year formats (with smart year interpretation)
        "%m/%d/%y %H:%M",      # 10/08/25 14:30
        "%m-%d-%y %H:%M",      # 10-08-25 14:30
        "%m/%d/%y",            # 10/08/25
        "%m-%d-%y",            # 10-08-25
        
        # Time-only formats (assumes today)
        "%H:%M",               # 14:30
        "%I:%M %p",            # 2:30 PM
        "%I:%M%p",             # 2:30PM
    ]
    
    for fmt in formats:
        try:
            parsed_time = datetime.strptime(time_str, fmt)
            
            # Handle 2-digit years with smart interpretation
            if fmt.endswith('%y'):
                current_year = now.year
                year = parsed_time.year
                
                # Convert 2-digit year to 4-digit year
                if year < 100:
                    # Assume years 00-99 are 2000s (2000-2099)
                    parsed_time = parsed_time.replace(year=year + 2000)
                
                # Ensure the year is not in the past
                if parsed_time.year < current_year:
                    # If the year is in the past, assume it's next century
                    parsed_time = parsed_time.replace(year=parsed_time.year + 100)
            
            # Handle time-only formats (use current date)
            if fmt in ["%H:%M", "%I:%M %p", "%I:%M%p"]:
                parsed_time = parsed_time.replace(year=now.year, month=now.month, day=now.day)
            
            # If no time was specified, use current time
            if fmt in ["%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%y"]:
                parsed_time = parsed_time.replace(hour=now.hour, minute=now.minute)
            
            # Ensure 4-digit years are not in the past
            if parsed_time.year < now.year:
                # If the year is in the past, assume it's next century
                parsed_time = parsed_time.replace(year=parsed_time.year + 100)
            
            return parsed_time
            
        except ValueError:
            continue
    
    return None


# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    reminder_manager.save_shutdown_time()
    reminder_manager.save_reminders()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Termination signal

#run bot w/ token from environment
bot.run(os.getenv('DISCORD_TOKEN'))