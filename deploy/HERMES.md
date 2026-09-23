# Operating Hermes-backed Peter

Current P910 deployment (September 23, 2026) uses the [dedicated worker VM](../docs/worker-vm.md),
member work access, and private officer controls in `#officers` and `#testing`.
See the [live release record](../docs/release-evidence.md) and
[cutover runbook](../docs/p910-cutover.md) for current limits, image revisions,
backups, and rollback. The pilot defaults below document the earlier Docker
stage and must not be used as the live P910 configuration.

Hermes is an immutable upstream source dependency, not a fork or submodule. The worker Dockerfile installs the pinned revision in `requirements-hermes.txt` using the upstream-required editable installation. Runtime root files remain read-only. Hermes streaming is explicitly disabled because Peter's capability proxy returns non-streamed completions; reasoning remains enabled. Do not update Hermes without the adapter tests and a real-model smoke test.

## Services and authority

- `peterbot`: existing Discord bot plus trusted task/memory/model gateway, port 8770 on private Docker networks only.
- `peterbot-hermes-runner`: trusted Docker supervisor, port 8780 on `peterbot_control` only. It has the Docker socket; never attach it to `peterbot_workers`.
- `peterbot-peterbot-<task UUID>`: disposable UID10000 worker, fixed resources, no host mounts/credentials/socket, internal `peterbot_workers` network only.
- Worker network: `192.168.240.0/24`; trusted gateway fixed at `.2`. Check for subnet conflicts on a new host.
- `peterbot-worker-firewall.service` blocks this worker subnet from host services. Install/start it before launching workers. It must remain active after reboot. Do not relax it to work around a failed task.

The worker only receives a short-lived capability for its job. The gateway verifies the requester's current Discord roles/channel access on every model/tool request. Personal memory is isolated by guild and user; club memory is public and officer-writable. Record versions and immutable revisions support audit. Authority never comes from memory. No private officer knowledge store is enabled in this pilot.

## Discord member workflow

The current P910 configuration has member work enabled in its configured listen
channels. Peter responds when named, mentioned, replied to, or addressed through
a recent scoped follow-up. Bare greetings are answered locally. Ordinary
questions get a conversational model turn; requests that need tools or promised
files go to a disposable worker and return to the original message. The
foreground scheduler queues competing turns and gives an honest wait notice.

`officer_only` in `deploy/hermes.example.json` is an earlier pilot default, not
the current protected production value. Authorization still comes from current
Discord identity, channel, and roles. Shared-channel work receives public club
context and the requester's scoped context; private personal memory and task
history do not flow into a public work request.

- `/task prompt [attachment]` starts explicit work in a private task thread.
- `/tasks`, `/continue_task`, and `/cancel_task` inspect, resume, or stop owned
  work. A cancellation preserves valid files collected before teardown.
- `/memory` and `/forget` inspect and remove authorized recall entries; audit
  revisions remain.
- `/ask`, `/recap`, `/suggest`, and `/remindme` remain available.

Private task threads can still be visible to Discord server administrators and
members with Manage Threads; task ownership prevents another member from
continuing or cancelling one. The task accepts at most three UTF-8 text/code
attachments totaling 128 KiB. Generated artifacts total at most 8 MiB. The
Hermes worker does not receive general browser/network, Discord administration,
cron, or subagent authority. Exact pinned PyPI wheels and crates.io crates are
available only through the authenticated [dependency broker](../docs/dependency-access.md).

One worker task runs at a time. At most two tasks per user and 20 globally may
be pending. Defaults include a 20-minute task deadline, 30 Hermes iterations,
8192 output tokens per model response, and a 131072-token task output budget.
For the current conversation routing, tier budgets, retries, and live Qwen
measurements, see [model latency](../docs/model-latency.md). Club facts come
from `club-knowledge.md` and versioned officer updates; missing configured
knowledge fails startup rather than making Peter guess.

## Presence: one message that becomes the answer

`presence.py` owns at most one member-facing message per turn. Discord's typing indicator refreshes itself for as long as the `typing()` context is open (discord.py re-sends it every five seconds), but it carries no information and nothing holds it open while a sandbox task runs. Presence fills that gap:

- A turn that finishes quickly posts nothing, so ordinary banter never flashes a placeholder. The status message appears only after `STATUS_AFTER_SECONDS` (6), and edits are throttled to `MIN_EDIT_SECONDS` (3).
- The status message is edited in place as work continues and finally *becomes* the answer, so the channel shows one message rather than a stale placeholder above a reply. Answers longer than one message edit the first chunk in and send the rest.
- A handoff announces itself immediately, because the member is about to wait minutes for another process, and the message id is stored on the job (`status_message_id`) so delivery still finds it after a gateway restart.
- While a task runs, `report_progress` refreshes the line every `PROGRESS_EVERY_SECONDS` (20) with elapsed time and stage. The worker does not stream progress, so only those two honest facts are reported — there is deliberately no percentage.
- Every presence failure is logged and swallowed: a deleted message, a lost permission or a failed post must never fail the turn or lose the answer. If the status message cannot be used, delivery falls back to an ordinary reply.

Verification: `deploy/smoke_presence.py` runs the real `run_job` and `deliver` against a recording channel with a real model, runner and worker, and prints the send/edit transcript a member would see.

## Persistence and delivery

The appdata `data/hermes` directory stores `tasks.sqlite3` and `memory.sqlite3`. Back up their directory with the service stopped or SQLite's backup API. Do not copy individual live WAL database files in isolation. Queued jobs survive restart; in-flight jobs are marked interrupted and can be explicitly continued. The previous objective, final response and input files carry into a continuation; a disposable filesystem and full tool execution stack do not. Agent steps are not transparently replayed after a crash.

A delivery cursor avoids repeating acknowledged Discord chunks. If the gateway restarts during an unacknowledged send, the delivery is marked `unknown` and held for operator reconciliation; it is not blindly replayed. A verified Discord message ID can confirm one chunk, while an operator can authorize a retry after checking that no message landed. This does not guarantee exactly-once delivery across an unknown outcome. Results are withheld after access revocation. State/audit currently requires operator retention management; deletion from recall does not purge audit or historical task records.

## Deployment and rollback

Production paths on p910:

- Source checkout: `/mnt/NVME/docker/appdata/peterbot/repo`
- Deployment: `/mnt/NVME/docker/compose/peterbot/compose.yaml` (after cutover)
- Protected deployment environment and `hermes.production.json` stay outside Git.
- Image build staging: `/mnt/NVME/docker/appdata/peterbot/builds/hermes-pilot`
- Previous deployment snapshot: appdata `backups/pre-hermes-*`.

Build the gateway target `bot` in `Dockerfile`, runner in `docker/Dockerfile.hermes-runner`, and worker in `docker/Dockerfile.hermes-worker`. Set all three image variables in the deployment `.env`. Start the runner, wait for its health, run `deploy/smoke_hermes.py` in a disposable gateway container with synthetic identity and temporary state, and run `deploy/check_hermes_isolation.py` inside an actual restricted worker. Do not run two Discord gateways with the same token.

Three smoke scripts, in increasing distance from the sandbox:

- `deploy/smoke_hermes.py`: task path. Needs the runner and a disposable gateway container on both networks, alias `gateway`, worker address `192.168.240.2`. Stop the production gateway while it holds that address.
- `deploy/smoke_conversation.py`: same, with `PETERBOT_SMOKE_CONVERSATION=1` for the public conversation delivery mode.
- `deploy/smoke_conversation_turn.py`: the fast conversational turn against the live model, no runner or worker needed (control network only). Six prompts covering banter, club facts, arithmetic, tool-needing requests and a file request. Every case must return reply text or a deliberate handoff, with no raised exception and no internal string in the reply. Run this after any change to `conversation.py`, the persona, or the knowledge file.
- `deploy/smoke_presence.py`: presence and delivery. Drives the real `run_job` and `deliver` with a recording channel: one status message must appear, be edited while the task runs, finally hold the answer, and artifacts must arrive as their own messages. Needs the runner and the worker address, like `smoke_hermes.py`.
- `deploy/run_sandbox_task.py`: reproduction tool, not a pass/fail smoke. Runs one prompt through the runner and prints every model call's status, duration, reasoning size and tool names plus the final outcome and artifacts.

Note that a handoff for "who are the current club officers?" is correct: that answer needs the live roster tool, not the static knowledge file.

For a later image switch or rollback, follow the
[cutover runbook](../docs/p910-cutover.md). It requires a fresh verified state
snapshot, protected copies of deployment config, one Discord gateway at a
time, and reconciliation of uncertain delivery before any replay. The old
same-host Docker pilot is a historical topology; current member work runs in
the dedicated VM boundary.
