# PeterBot

Discord bot with:
- mention-based chat responses via Ollama
- `/ask`, `/recap`, `/suggest`, and `/remindme` slash commands
- reminder persistence across restarts
- dry/direct club-bot prompting with model profiles and response cleanup
- optional club knowledge and channel tone profiles
- structured logging with user-facing debug IDs

## Requirements

- Python 3.9+
- Discord bot token
- Ollama server reachable from the bot process

## Setup

1. Install runtime dependencies:
```bash
python3 -m pip install -r requirements.txt
```
2. (Optional) Install development/test dependencies:
```bash
python3 -m pip install -r requirements-dev.txt
```
3. Create `.env` and configure at least `DISCORD_TOKEN`.
4. Run the bot:
```bash
python3 bot.py
```

## Environment Variables

### Required

- `DISCORD_TOKEN`: Discord bot token.

### Core behavior

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`): Ollama base URL.
- `OLLAMA_MODEL` (default: `qwen3.5`): model used for chat. Set this to your Ollama alias if needed.
- `PETER_NAME` (default: `Peter`): name injected into system prompt.
- `PETER_SYSTEM_PROMPT`: persona seed used by the layered prompt builder. Hard style rules still keep Peter in a dry/direct club-bot voice.
- `OLLAMA_THINK` (default: `false`): forwarded to Ollama's top-level `think` flag. When enabled, Peter lets the model use hidden reasoning but only sends the final answer back to Discord.
- `PETER_MODEL_PROFILE` (default: `auto`): one of `auto`, `generic`, or `qwen`. `auto` selects `qwen` whenever `OLLAMA_MODEL` contains `qwen`.
- `OLLAMA_OPTIONS_JSON` (optional): JSON object forwarded to Ollama as `options`, for example `{"temperature":0.3}`.
- `SUGGESTION_CHANNEL_ID`: channel ID for `/suggest`.
- `PETER_KNOWLEDGE_FILE` (optional): Markdown file with `##` and `###` sections used as lightweight club knowledge.
- `PETER_CHANNEL_PROFILES_FILE` (optional): JSON file keyed by channel name or channel ID with `tone`, `reply_length`, and `topics`.

### Persistence

- `PETERBOT_DATA_DIR` (default: directory containing `bot.py`):
  directory for `reminders.json` and `bot_shutdown.json`.
  If new-path files do not exist, the bot attempts a one-time legacy read from current working directory.

## Optional Local Config

### Knowledge file

Example `PETER_KNOWLEDGE_FILE`:

```md
## Meetings
We meet every Thursday at 6:30 PM in the hardware lab.

## Resources
The club GitHub lives at https://github.com/Computer-Hardware-Club.
```

### Channel profile file

Example `PETER_CHANNEL_PROFILES_FILE`:

```json
{
  "hardware-help": {
    "tone": "practical, direct, low-fluff",
    "reply_length": "short unless troubleshooting needs detail",
    "topics": ["PC builds", "parts advice", "benchmarking"]
  },
  "123456789012345678": {
    "tone": "casual club chatter",
    "reply_length": "compact",
    "topics": ["meeting reminders", "event planning"]
  }
}
```

## Commands

- Mention Peter in-channel to get a context-aware reply.
- `/ask`: ask Peter a question using the latest channel context.
- `/recap`: summarize the latest discussion into `What happened`, `Decisions`, and `Open questions`.
- `/suggest`: send a suggestion to the configured suggestions channel.
- `/remindme`: schedule a DM reminder.

### Logging and debugging

- `LOG_LEVEL` (default: `INFO`): standard Python logging level.
- `LOG_FILE` (optional): if set, enables rotating file logs (5 MB, 5 backups).
- `USER_DEBUG_IDS_ENABLED` (default: `true`): include debug IDs in user-facing error messages.
- `INCLUDE_TRACEBACK_FOR_WARNING` (default: `false`): include traceback details for warning-level paths.

## Debugging Workflow

When a user-facing failure occurs, the bot returns a debug ID like:

```text
Debug ID: ERR-1a2b3c4d
```

Use that ID to search logs:

```bash
rg "ERR-1a2b3c4d" -n .
```

Recommended production logging setup:

```bash
LOG_LEVEL=INFO
LOG_FILE=./logs/peterbot.log
USER_DEBUG_IDS_ENABLED=true
INCLUDE_TRACEBACK_FOR_WARNING=false
```

Recommended local debugging setup:

```bash
LOG_LEVEL=DEBUG
LOG_FILE=./logs/peterbot-debug.log
USER_DEBUG_IDS_ENABLED=true
INCLUDE_TRACEBACK_FOR_WARNING=true
```

## Testing

Run syntax and tests:

```bash
python3 -m py_compile bot.py
python3 -m pytest -q
```

## Notes

- Runtime files (`reminders.json`, `bot_shutdown.json`, logs) are gitignored.
- If reminders fail to deliver (for example DM permissions), delivery is retried for transient Discord errors and dropped for permanent permission errors.
- Peter is meant to sound like a plain club bot, not a human server regular.
