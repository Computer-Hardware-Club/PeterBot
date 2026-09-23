# PeterBot

Discord bot with:
- mention-based chat replies
- bounded tool calls for web search, public webpages, and arithmetic
- `/ask`, `/recap`, `/suggest`, and `/remindme` slash commands
- reminder persistence across restarts
- optional club knowledge and channel tone profiles
- Docker deployment with a remote vLLM server or local `llama.cpp`
- structured logging with user-facing debug IDs

## Runtime Model

PeterBot uses an OpenAI-compatible chat API. The remote deployment supports vLLM with structured tool calling. The model chooses from a fixed tool allowlist; application code validates and executes each call. Image mentions require a multimodal backend; set `agent.vision_enabled` to false for text-only deployments.

## Remote model and tools

Use `docker compose -f compose.remote.yml up --build -d` for the bot-only deployment. First set `inference.base_url`, `inference.model`, and `agent.search_base_url` in your deployment configuration. Loopback values in the repository are examples and must be replaced with endpoints reachable **from the bot container**. Keep production addresses and secrets in an external configuration/Compose environment, outside Git. Mount the production configuration at `/app/config.json`.

Set `agent.allowed_guild_ids` to your club's Discord server ID before starting the bot. An empty list permits any guild the bot has joined; DMs remain disabled by default. vLLM must support automatic structured tool calls and the served model's tool parser. The default request disables thinking using `chat_template_kwargs.enable_thinking` and requests one completion.

Peter's tools are:

- `web_search`: SearXNG search results with snippets and source URLs. Search failures and partial results are reported. Queries go to public search providers.
- `fetch_public_page`: reads public HTML or text pages, including the club website. It verifies DNS answers and each redirect, connects only to public IPs, and rejects private/local/Tailscale/metadata addresses, credentials and non-web ports. It does not render JavaScript or fetch linked assets. Firecrawl is not connected in this first version.
- `calculate`: bounded arithmetic, including powers, with no Python execution or imports.

The default agent budget is two tool rounds, four tool calls, at most three model requests and 3,072 allocated output tokens per answer. A 60-second deadline covers request handling. Admission limits are one active request globally, one per user, three requests per user per minute and fifteen per guild per minute. Oversized requests are rejected, outputs are capped, and generated Discord mentions are suppressed. `/recap` shares admission limits but does not use external tools.

There are no model tools for shell access, files, credentials, Discord administration or server changes. Existing explicit `/remindme` and `/suggest` commands remain separate from model tools. Tool-capable rounds see the current question and attachments plus the static public persona. They never receive other members' channel history, author identity or dynamic reply context. Channel context returns only in the final stage, where further tools are disabled and any attempted tool call is rejected. Keep the static persona public. No cross-channel search or durable chat memory is added. Output suppresses automatic link previews as well as mentions.

Web content remains untrusted: prompt instructions help guide behavior, but the tool/network limits are enforced by code. Prompt injection can still affect answer quality; source links are not an endorsement of accuracy. Current questions and attachments are input to tool planning, so users should not include secrets. Rate limits are process-local, reset on restart, and assume a single bot process. An administrator controls configured service endpoints; use private networking and existing service authentication where available. `LLAMA_CPP_API_KEY` is also supported for vLLM and is never sent by the separate web-tool sessions.

References: [vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/) and [SearXNG search API](https://docs.searxng.org/dev/search_api.html).

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
- `agent`: tools, request budgets, quotas, guild access and image capability

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
