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

One sandbox agent task runs at a time. At most two tasks per user and 20 globally may be pending. Default limits are 20 minutes per task, 30 Hermes iterations, 8192 tokens per response, and a total allocated output budget of 131072 tokens. Thinking is enabled by the trusted model proxy regardless of caller flags. These are independent of legacy member-chat budgets. `deploy/prepare_hermes_config.py` generates a bot config with the stable persona, thinking enabled for member chat too, a 4096-token response allowance and a 240-second legacy request limit. Preserve a backup before replacing production JSON.

## Fast conversational turn

The first model turn on a mention decides whether to answer or hand the request to the sandbox. That turn runs with thinking enabled, because the deployed reasoning model does not emit tool calls reliably without it, and its completion budget (4096, and never below that) leaves room for thinking as well as the answer. Thinking is billed against the same budget, so a 2k allowance truncates mid-thought and returns an empty answer.

Reliability rules for that turn, all enforced in `conversation.py`:

- A blank answer is retried once with thinking disabled, which is the reliably non-empty path, plus an instruction to answer plainly.
- Two blank answers return a short human line. Members never see an internal error string from a model wobble.
- A blank answer never starts sandbox work by itself: only an explicit handoff does.
- A tool name or argument shape the fast model invented is treated as a handoff, not an error. The sandbox re-checks authority and honours only its own allowlist, so failing toward doing the work is the safe direction.
- Only wall-clock that is actually left is spent: the turn honours `inference.timeout_seconds`, and the retry shares the remaining budget instead of getting a fresh one.
- The deadline is sized for a reasoning model (420 seconds deployed). A hard question can spend minutes thinking before it emits a byte, and a shorter deadline turns that into a member-facing failure.
- The thinking attempt is capped at `TOTAL_ATTEMPT_SECONDS` and holds back `RETRY_RESERVE_SECONDS` for the cheap retry, keeping at least half of what is left if the deadline is short. Attempts log their budget, so a slow turn is distinguishable from a dead one.
- Requests stream, and a stream that goes quiet for `STREAM_IDLE_SECONDS` is treated as dead. Non-streamed, vLLM sends nothing until the whole completion is finished, so a total deadline cannot tell "still thinking" from "server gone" and always loses to a long turn.
- The fast turn is told to decide promptly and hand off rather than attempt real work itself.

Club facts come from `club-knowledge.md`, baked into the gateway image and loaded through `paths.knowledge_file`. The file must exist: a missing one fails config load rather than silently letting Peter answer club questions from guesses. Both the conversational turn and the sandbox persona receive the same excerpt.

Sandbox model calls get their own deadline (up to 600 seconds, bounded by `job_timeout`). The session-wide client deadline is far too short for a reasoning model writing thousands of tokens.

The worker sets an explicit `HERMES_API_CALL_STALE_TIMEOUT` (600 seconds) before it builds the agent. The trusted proxy answers non-streamed, and Hermes abandons a non-streamed call it has heard nothing from: the upstream floor for this model family is 180 seconds, while a 4k-token reasoning turn needs up to about 250 seconds before its first byte. Without the override, long tasks die as `model_failed` after the stale retry collides with the proxy's one-call-per-task lock. Setting it explicitly also prevents the run-budget calculation from halving it mid-job.

A heavy task can still exceed the 20-minute job budget, because the deployed model decodes at roughly 15-20 tokens per second and a coding task spends minutes reasoning. That ends as an honest `timeout` with the artifacts collected, not as a model error. The same limit decides whether a member's long request should hand off early rather than be attempted in the conversational turn.

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

For rollback, stop/remove only the new `peterbot` container, restore the saved Compose file, `.env`, and `config.production.json`, then recreate `peterbot` from the previous gateway image (currently `peterbot-hermes-gateway:088c670-flashnext-v2`). Stop the new runner after active workers are gone. Preserve new SQLite state for diagnosis or later reuse. The dedicated worker firewall may safely remain installed.

This Docker pilot shares p910's kernel. Move execution to a dedicated VM before widening to general member access, arbitrary network/package downloads, or more privileged capabilities. Command allowlists and model instructions are not substitutes for OS/network isolation.
