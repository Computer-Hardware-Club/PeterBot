# PeterBot

Peter is the Computer Hardware Club at Oregon State University's Discord bot. He answers club questions, joins conversations when addressed, researches public sources, and can build and return real project files in an isolated worker. The current P910 deployment serves `Qwen3.8-Flash-Next` through an OpenAI-compatible vLLM endpoint.

## What members can do

- Say “Peter,” mention him, reply to him, or continue a recent conversation in a configured channel. Short greetings get short answers; longer work shows one editable progress message that becomes the answer.
- Ask for club facts, explanations, current research, calculations, or source files. Peter uses bounded web tools and sends coding work to a disposable Hermes worker. He returns generated files as Discord attachments and can continue saved project work later.
- Use `/ask` for a private answer, `/recap` for a channel summary, `/suggest` for a suggestion, and `/remindme` for a DM reminder.
- Use `/task` for an explicit private work thread, `/tasks` to see saved tasks, `/continue_task` to resume one, and `/cancel_task` to stop one. `/memory` inspects scoped memories; `/forget` removes an authorized memory from recall while retaining its audit revision.

Peter keeps conversation state across gateway restarts. Personal memory and private task history stay scoped to the requester; public club notes can inform later shared answers. Authorized officers can update club facts and Peter's style through source-bound requests in configured private control channels. A request to post an announcement needs an allowed destination and a one-time confirmation; it is recorded in a recoverable outbox. Ordinary conversation does not grant officer authority.

The current member rollout and its verification are recorded in [release evidence](docs/release-evidence.md). The [cutover runbook](docs/p910-cutover.md) describes a later deployment or rollback. Merging code to GitHub does **not** deploy a new image to P910.

## Architecture

| Component | Responsibility |
| --- | --- |
| Discord gateway | Identity and role checks, conversation routing, public tools, model access, durable queue, scoped memory and projects, delivery, and announcement outbox. |
| Trusted runner | Starts and cancels one bounded worker at a time; owns the Docker socket inside the worker VM. |
| Disposable Hermes worker | Runs code and file tools with no Discord token, host mounts, Docker socket, or general network access. Model and allowed package requests pass through task-bound gateway capabilities. |
| OpenAI-compatible model server | Serves Qwen for conversation and worker turns. Its endpoint and credentials remain outside Git. |

The P910 production topology uses a [dedicated worker VM](docs/worker-vm.md) and a default-deny firewall. The repository also includes a same-host `compose.hermes.yml` pilot topology; its example settings are **not** the production configuration. Workers can request exact pinned public Python wheels and Rust crates through the [controlled dependency broker](docs/dependency-access.md). They cannot browse the host or make arbitrary outbound connections.

See [Hermes operations](deploy/HERMES.md) for service boundaries, [project workspaces](docs/project-workspaces.md) for durable file scope, [operations and retention](docs/ops-and-retention.md) for diagnostics and backups, and [CI and live verification](docs/ci-and-live-verification.md) for the test gates.

## Configuration

[`config.json`](config.json) is a tracked example, not the P910 production file. It contains the persona, model settings, Discord IDs, file paths, logging, and legacy bounded-tool limits. [`deploy/hermes.example.json`](deploy/hermes.example.json) contains example guild, officer, channel, runner, and task settings. Set real guild and role IDs, listen and control channels, model endpoint, and network addresses in protected deployment copies. Empty example allowlists must not be mistaken for a production access policy.

Keep secrets outside Git. `DISCORD_TOKEN` is required. `LLAMA_CPP_API_KEY` is the optional model API bearer token; `PETERBOT_RUNNER_TOKEN` authenticates trusted gateway/runner operations. The gateway reads `PETERBOT_CONFIG_FILE` and `PETERBOT_HERMES_CONFIG` when supplied. See [`.env.example`](.env.example) for the basic bot environment and [`compose.hermes.yml`](compose.hermes.yml) for the three-service pilot variables.

Club facts live in [`club-knowledge.md`](club-knowledge.md) plus versioned officer updates. The configured knowledge file must exist. Optional channel tone profiles can be supplied through `paths.channel_profiles_file`. Generated data and snapshots belong on persistent private storage, outside the source tree when deployed.

## Running and deploying

For local development, use Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
# Set DISCORD_TOKEN and a reachable inference.base_url in a local config copy.
.venv/bin/python bot.py
```

The local run uses the configured model and legacy bounded tools. A complete Hermes task path also needs the trusted runner, pinned worker image, protected Hermes configuration, and its network controls. Follow [Hermes operations](deploy/HERMES.md) and the [P910 cutover runbook](docs/p910-cutover.md); keep exactly one gateway connected to the Discord token.

For the older bundled `llama.cpp` mode, place a compatible GGUF at `./models/peterbot.gguf`, set `DISCORD_TOKEN` in `.env`, and run `docker compose up --build`. [`compose.bundled.yml`](compose.bundled.yml) is the explicit equivalent; [`compose.sidecar.yml`](compose.sidecar.yml) runs a separate `llama.cpp` container. [`compose.remote.yml`](compose.remote.yml) is the bot-only remote-model variant. These compatibility modes do not provide the deployed VM-backed Hermes workflow by themselves.

## Verification

```bash
.venv/bin/python -m json.tool config.json >/dev/null
.venv/bin/python -m json.tool deploy/hermes.example.json >/dev/null
.venv/bin/python -m compileall -q peterbot deploy tests
.venv/bin/python -m pytest -q
```

GitHub Actions also builds the gateway, runner, and pinned worker images, runs the Hermes adapter fixture inside the worker image, and checks worker isolation. The ordinary Python suite skips the optional runtime fixture when upstream Hermes is absent locally; that skip alone does not validate the adapter. A live release additionally needs the VM, model, Discord, and delivery checks in the [cutover runbook](docs/p910-cutover.md).

If Peter shows a `Debug ID: ERR-…`, search the protected gateway logs for that ID. For operational health, `/health` checks the process and Discord connection; the authenticated `/diagnostics` route also checks model, runner, and queue readiness. Details and backup procedures are in [operations and retention](docs/ops-and-retention.md).
