# PeterBot

Discord bot with:
- mention-based chat responses via Ollama
- `/ask`, `/suggest`, and `/remindme` slash commands
- reminder persistence across restarts
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
- `OLLAMA_MODEL` (default: `ministral-3:8b`): model used for chat.
- `PETER_NAME` (default: `Peter`): name injected into system prompt.
- `PETER_SYSTEM_PROMPT`: overrides default persona/system prompt.
- `OLLAMA_THINK` (default: `false`): forwarded to Ollama `options.think`.
- `SUGGESTION_CHANNEL_ID`: channel ID for `/suggest`.

### Persistence

- `PETERBOT_DATA_DIR` (default: directory containing `bot.py`):
  directory for `reminders.json` and `bot_shutdown.json`.
  If new-path files do not exist, the bot attempts a one-time legacy read from current working directory.

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
