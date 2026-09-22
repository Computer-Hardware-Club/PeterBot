# Operating Hermes-backed Peter

Hermes is an immutable upstream source dependency, not a fork or submodule. The worker Dockerfile installs the pinned revision in `requirements-hermes.txt` using the upstream-required editable installation. Runtime root files remain read-only. Hermes streaming is explicitly disabled because Peter's capability proxy returns non-streamed completions; reasoning is disabled for the served Flash Next model so bounded Discord tasks receive usable tool calls and final answers. Do not update Hermes without the adapter tests and a real-model smoke test.

## Services and authority

- `peterbot`: existing Discord bot plus trusted task/memory/model gateway, port 8770 on private Docker networks only.
- `peterbot-hermes-runner`: trusted Docker supervisor, port 8780 on `peterbot_control` only. It has the Docker socket; never attach it to `peterbot_workers`.
- `peterbot-peterbot-<task UUID>`: disposable UID10000 worker, fixed resources, no host mounts/credentials/socket, internal `peterbot_workers` network only.
- Worker network: `192.168.240.0/24`; trusted gateway fixed at `.2`. Check for subnet conflicts on a new host.
- `peterbot-worker-firewall.service` blocks this worker subnet from host services. Install/start it before launching workers. It must remain active after reboot. Do not relax it to work around a failed task.

The worker only receives a short-lived capability for its job. The gateway verifies the requester's current Discord roles/channel access on every model/tool request. Personal memory is isolated by guild and user; club memory is public and officer-writable. Record versions and immutable revisions support audit. Authority never comes from memory. No private officer knowledge store is enabled in this pilot.

## Discord pilot

Ordinary mentions are conversational: Peter answers in the original channel without creating a thread or showing task IDs/status messages. A short model turn decides whether tools are needed. If so, work runs quietly in the existing sandbox and the useful answer/files are returned as a reply to the original message. Shared-channel workers receive public club memory and same-requester context only; personal memory is unavailable, including ID-based updates/deletes. Social context from other speakers stays in the conversational turn and is not sent to sandbox tools.

`officer_only: true` preserves the current tool pilot for configured officer role IDs. Member mentions and `/ask` keep the existing bounded conversational path. Explicit `/task` remains an optional private workspace, never the default for pings.

- `/task prompt [attachment]`: start work in a new private, non-invitable task thread.
- `/ask` stays a private conversational answer. Officer mentions use the conversational/tool-routing path above.
- Private work is explicitly continued through `/continue_task`; ordinary thread messages are not automatically converted into tasks.
- `/tasks`: list your task IDs and statuses.
- `/cancel_task task_id`: revoke the task and request immediate container termination.
- `/continue_task task_id prompt`: continue a finished/interrupted task in its original private thread.
- `/memory scope query`: privately inspect personal/public club memory.
- `/forget memory_id version`: remove an authorized memory from recall. Restricted audit revisions remain.

For explicitly requested private tasks, Discord server administrators and members with Manage Threads may be able to access private threads; they are not confidential from server administration. Task ownership still prevents another user from taking over a task. The pilot accepts at most three UTF-8 text/code attachments totaling 128 KiB (one attachment in `/task`, multiple through mentions/follow-ups). Generated artifacts total at most 8 MiB. Images, Office/PDF uploads, arbitrary internet/package access, outbound messaging tools, server administration, native global memory/session search, cron and subagents are not exposed yet.

One agent task runs at a time. At most two tasks per user and 20 globally may be pending. Default limits are 20 minutes per task, 30 Hermes iterations, 8192 tokens per response, and a total allocated output budget of 131072 tokens. The trusted model proxy disables thinking for Flash Next task requests so reasoning cannot consume the entire bounded response budget before a tool call or final answer. These are independent of legacy member-chat budgets. `deploy/prepare_hermes_config.py` generates a bot config with the stable persona, thinking disabled for member chat, a 4096-token response allowance and a 120-second legacy request limit. Preserve a backup before replacing production JSON.

## Persistence and delivery

The appdata `data/hermes` directory stores `tasks.sqlite3` and `memory.sqlite3`. Back up their directory with the service stopped or SQLite's backup API. Do not copy individual live WAL database files in isolation. Queued jobs survive restart; in-flight jobs are marked interrupted and can be explicitly continued. The previous objective, final response and input files carry into a continuation; a disposable filesystem and full tool execution stack do not. Agent steps are not transparently replayed after a crash.

A delivery cursor avoids repeating already-recorded Discord sends on ordinary retries. A crash after Discord accepts a send but before SQLite records it can still cause a duplicate. Results are withheld after access revocation. State/audit currently requires operator retention management; deletion from recall does not purge audit or historical task records.

## Deployment and rollback

Production paths on p910:

- Source checkout: `/mnt/NVME/docker/appdata/peterbot/repo`
- Deployment: `/mnt/NVME/docker/compose/peterbot/compose.yaml` (after cutover)
- Protected deployment environment and `hermes.production.json` stay outside Git.
- Image build staging: `/mnt/NVME/docker/appdata/peterbot/builds/hermes-pilot`
- Previous deployment snapshot: appdata `backups/pre-hermes-*`.

Build the gateway target `bot` in `Dockerfile`, runner in `docker/Dockerfile.hermes-runner`, and worker in `docker/Dockerfile.hermes-worker`. Set all three image variables in the deployment `.env`. Start the runner, wait for its health, run `deploy/smoke_hermes.py` in a disposable gateway container with synthetic identity and temporary state, and run `deploy/check_hermes_isolation.py` inside an actual restricted worker. Do not run two Discord gateways with the same token.

For rollback, stop/remove only the new `peterbot` container, restore the saved Compose file, `.env`, and `config.production.json`, then recreate `peterbot` from `peterbot:agent-58d8935`. Stop the new runner after active workers are gone. Preserve new SQLite state for diagnosis or later reuse. The dedicated worker firewall may safely remain installed.

This Docker pilot shares p910's kernel. Move execution to a dedicated VM before widening to general member access, arbitrary network/package downloads, or more privileged capabilities. Command allowlists and model instructions are not substitutes for OS/network isolation.
