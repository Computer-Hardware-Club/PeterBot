import discord
from discord.ext import commands
import os
from datetime import datetime
from dotenv import load_dotenv

#define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Load environment variables
load_dotenv()

#create bot instance
bot = commands.Bot(command_prefix='!', intents=intents)

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
    print(f'We have logged in as {bot.user}') #login message
    try:
        synced = await bot.tree.sync() #syncs commands with discord
        print(f'Synced {len(synced)} command(s)')
    except Exception as e:
        print(f'Failed to sync commands: {e}')

# Slash command for hello
@bot.tree.command(name="hello", description="Say hello to the bot")
async def hello(interaction: discord.Interaction):
    await interaction.response.send_message('Hello!', ephemeral=True)

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

#run bot w/ token from environment
bot.run(os.getenv('DISCORD_TOKEN'))