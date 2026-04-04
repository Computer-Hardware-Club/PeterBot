# PeterBot

Discord bot with:
- mention-based chat replies
- `/ask`, `/recap`, `/suggest`, and `/remindme` slash commands
- reminder persistence across restarts
- optional club knowledge and channel tone profiles
- Docker-first deployment with `llama.cpp`
- structured logging with user-facing debug IDs

## Runtime Model

PeterBot now uses the `llama.cpp` HTTP server through its OpenAI-compatible chat API.
Mention image support requires a multimodal `llama.cpp` model. Text-only models will still answer normal text mentions, but they cannot analyze attached images.

Supported deployment modes:
- default `docker compose` flow: one bot image that also includes `llama-server`
- `compose.bundled.yml`: explicit compatibility alias for the bundled flow
- `compose.sidecar.yml`: optional advanced mode with a separate `llama.cpp` server container
- native local Python run: still supported for development and simple local use

## Configuration

### `config.json`

All non-secret settings live in [`config.json`](/Users/ofhd/Developer/PeterBot/config.json).

Sections:
- `persona`: bot name, system prompt, model profile
- `discord`: Discord-specific IDs such as `suggestion_channel_id`
- `inference`: `llama.cpp` API base URL, model alias, request tuning
- `llama_server`: local bundled server settings used when `enabled` is `true`
- `paths`: persistent data, optional knowledge/profile files, log file
- `logging`: log level and debug-id behavior
- `behavior`: message/context limits and reminder retry tuning

Relative paths in `config.json` resolve from the config file directory.

### `.env`

Only secrets belong in `.env`.

Supported variables:
- `DISCORD_TOKEN`: required
- `LLAMA_CPP_API_KEY`: optional, only if your `llama.cpp` server requires Bearer auth
- `PETERBOT_CONFIG_FILE`: config file path, defaults to `/app/config.json` in Docker

Start from [`.env.example`](/Users/ofhd/Developer/PeterBot/.env.example).

## Docker

Docker is the primary deployment path.

### Bundled Quick Start

Bundled mode is the default and recommended deployment path. It starts PeterBot and the packaged `llama-server` together with plain `docker compose up --build`.

### 1. Prepare secrets

```bash
cp .env.example .env
```

Set at least:

```env
DISCORD_TOKEN=your-discord-token
```

### 2. Put a GGUF model in `./models`

The repo does not ship model weights. Put your GGUF model in a local `./models` directory:

```bash
mkdir -p models
```

Default example model path:

```text
./models/peterbot.gguf
```

If you use a different filename, update [`docker/config.bundled.json`](/Users/ofhd/Developer/PeterBot/docker/config.bundled.json). If you also use sidecar mode, update [`docker/config.sidecar.json`](/Users/ofhd/Developer/PeterBot/docker/config.sidecar.json) and the `llama-cpp` command in [`compose.sidecar.yml`](/Users/ofhd/Developer/PeterBot/compose.sidecar.yml).

### 3. Start PeterBot

```bash
docker compose up --build
```

Behavior:
- the bundled image includes the `llama-server` binary
- the GGUF model is mounted from `./models`
- PeterBot uses [`docker/config.bundled.json`](/Users/ofhd/Developer/PeterBot/docker/config.bundled.json)
- bot state persists in `./peterbot-data`

If you want Peter to analyze Discord image attachments in mention replies, run a multimodal GGUF. Some models also require a separate multimodal projector, which you can pass through `llama_server.extra_args` in [`docker/config.bundled.json`](/Users/ofhd/Developer/PeterBot/docker/config.bundled.json), for example:

```json
"extra_args": ["--mmproj", "/models/mmproj-your-model.gguf"]
```

`compose.bundled.yml` remains available as a compatibility alias if you want an explicit file:

```bash
docker compose -f compose.bundled.yml up --build
```

### Optional Advanced Sidecar Mode

Use sidecar mode only if you intentionally want an external `llama.cpp` container.

```bash
docker compose -f compose.sidecar.yml up --build
```

Behavior:
- `llama.cpp` serves the GGUF model from `./models`
- PeterBot uses [`docker/config.sidecar.json`](/Users/ofhd/Developer/PeterBot/docker/config.sidecar.json)
- bot state persists in `./peterbot-data`

For mention image support in sidecar mode, the `llama-cpp` service must run a multimodal model. If the model needs a separate projector, add it to the `llama-cpp` command in [`compose.sidecar.yml`](/Users/ofhd/Developer/PeterBot/compose.sidecar.yml), for example:

```yaml
      - --mmproj
      - /models/mmproj-your-model.gguf
```

### Build targets

Bundled image:

```bash
docker build --target bundled -t peterbot:bundled .
```

Bot-only image for sidecar deployments:

```bash
docker build --target bot -t peterbot:latest .
```

## Native Local Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create `.env`, adjust [`config.json`](/Users/ofhd/Developer/PeterBot/config.json), and start the bot:

```bash
python3 bot.py
```

For native local use with a separate `llama.cpp` server, set `inference.base_url` in `config.json` to the correct host and port and keep `llama_server.enabled` as `false`.

## Optional Local Content

### Knowledge file

Example `paths.knowledge_file`:

```md
## Meetings
We meet every Thursday at 6:30 PM in the hardware lab.

## Resources
The club GitHub lives at https://github.com/Computer-Hardware-Club.
```

### Channel profile file

Example `paths.channel_profiles_file`:

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
- Mention Peter with attached images to get an image-aware reply when the backend is running a multimodal vision model.
- `/ask`: ask Peter a question using recent channel context.
- `/recap`: summarize the latest discussion into `What happened`, `Decisions`, and `Open questions`.
- `/suggest`: send a suggestion to the configured suggestions channel.
- `/remindme`: schedule a DM reminder.

## Logging and Debugging

Important config keys:
- `logging.level`
- `paths.log_file`
- `logging.user_debug_ids_enabled`
- `logging.include_traceback_for_warning`

When a user-facing failure occurs, the bot can return a debug ID like:

```text
Debug ID: ERR-1a2b3c4d
```

Use that ID to search logs:

```bash
rg "ERR-1a2b3c4d" -n .
```

## Verification

Syntax and tests:

```bash
python3 -m py_compile bot.py peterbot/*.py
python3 -m pytest -q
```

Docker config checks:

```bash
docker compose config
docker compose -f compose.sidecar.yml config
```

## Troubleshooting

If the container exits immediately with `Configuration error: DISCORD_TOKEN is not set. Add it to .env.`, copy [`.env.example`](/Users/ofhd/Developer/PeterBot/.env.example) to `.env` and set `DISCORD_TOKEN`.

If the bundled container exits immediately with `Configuration error: llama_server.model_path does not exist: /models/peterbot.gguf`, mount or place your GGUF model at `./models/peterbot.gguf` or update [`docker/config.bundled.json`](/Users/ofhd/Developer/PeterBot/docker/config.bundled.json) to match your filename.

## Notes

- Runtime data should stay on a mounted persistent volume or bind mount.
- Bundled mode includes the `llama-server` binary, not the model weights.
- Mention image attachments are forwarded to `llama.cpp` for mention replies when the backend is configured with a multimodal model.
- If the backend is text-only or missing multimodal setup, Peter replies with a short setup hint instead of pretending to analyze the image.
- Runtime files, `.env`, models, and local data dirs are gitignored.
