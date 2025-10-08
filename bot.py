import discord
import os
from dotenv import load_dotenv

#define intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Load environment variables
load_dotenv()

#create bot instance
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_message(message):
    # A simple command that replies to "!hello"
    if message.content.startswith('!hello'):
        await message.channel.send('Hello!')

#run bot w/ token from environment
bot.run(os.getenv('DISCORD_TOKEN'))