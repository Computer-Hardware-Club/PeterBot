# Peter redesign release evidence

Status: September 23, 2026. The `PETER-xx` keys map to the supplied local
backlog, not published GitHub issues. The implementation is the draft stacked
[PR #3](https://github.com/Computer-Hardware-Club/PeterBot/pull/3), based on
`feat/hermes-peter` (`332d366`). Gateway code revision `2800d97` is deployed
on P910 with VM runner/worker revision `08c8968`; the gateway-only hotfix adds
the exact typoed model-correction wording from the user's screenshot. It has
not been merged. The hosted [push CI run](https://github.com/Computer-Hardware-Club/PeterBot/actions/runs/35922403596)
and [PR CI run](https://github.com/Computer-Hardware-Club/PeterBot/actions/runs/35922408559)
both passed for `2800d97`.

| Backlog | Evidence | Remaining limit |
| --- | --- | --- |
| PETER-01 | The [pre-cutover baseline](p910-baseline-2026-09-22.md) separates Discord, model, runner, queue, state, and firewall health. The live authenticated `/diagnostics` endpoint later reported each dependency ready. A real `#testing` greeting, brokered work, and delivered files passed. | The original reported outage was not reproduced in the baseline. |
| PETER-02 | PR #1 changes were reconciled into foundation PR #2; PR #3 is stacked on it. CI runs the ordinary suite, compile/config checks, pinned Hermes fixture, and all three image builds without production secrets or automatic deployment. | Human review and merge remain separate. |
| PETER-03 | Preparing-job race, idempotent ingress, terminal delivery cursors, unknown receipts, cancel/complete races, and restart recovery have tests. An old four-row P910 jobs snapshot migrated without replay. Live `/cancel_task` stopped a sleeping worker, edited its status to cancelled, and left no active slot. | No claim of exactly-once Discord delivery under arbitrary outages. |
| PETER-04 | A durable global foreground scheduler covers chat, `/ask`, `/recap`, and worker work. During live `#testing`, a second request received a truthful one-ahead acknowledgement and answered after the first worker completed. | A live two-human-user race was not available; deterministic tests cover separate identities. |
| PETER-05 | Tiered model budgets, a shared turn deadline, safe rescue, and explicit-work handoff postcondition pass tests. Five warm-idle repetitions per case on the actual Qwen/vLLM server are in [model-latency.md](model-latency.md). Explicit coding handoff improved from 44.173 s to 2.462 s p95 in model-only probes, with 5/5 valid routes. The previous gateway revision routed a real coding request in 3.562 s, then delivered its compiled `ready.rs` file. | Previous model-routed greeting p50 was 2.007 s against a proposed 2 s target; bare greetings now skip that model call. End-to-end timing includes Discord and worker time. |
| PETER-06 | Name, mention, reply, and short unpinged follow-up routing pass tests. Bare greetings return one or two words with no model call. The awareness filter no longer drops a follow-up starting `Nah,` as another person's name, recognizes a final-line `Peter`, and renews a five-minute scoped conversation lease on each accepted follow-up. Live `Nah, Qwen under the hood` got a reply without another name mention. | Unrelated chatter remains intentionally ignored. |
| PETER-07 | Conversation turns persist by guild, requester, channel, and audience, independent of Discord transport. The private task thread continued its saved work after a gateway restart. Bounded saved public club notes now reach later chat below current typed facts; personal memory stays out of shared chat. | Private/public context is not merged into one transcript. |
| PETER-08 | One editable presence/status message, real stages, and owner cancellation are wired. The voice update renders verified worker stages as short playful text plus elapsed time. Live `*letting the compiler judge me* (40s)` edited the existing status; that timed task completed and delivered `voice-check.txt`. Earlier live cancellation finalized its original status and stopped the worker. | Discord may show separate attachment messages for files. |
| PETER-09 | Private officer controls bind intent to the current Discord source and fresh member roles. `#testing` joined `#officers` as a configured private control channel; live officer commands there worked. A member in testing and a public-channel command remain denied in integration tests. | We did not use a second nonofficer Discord account for a live denial. |
| PETER-10 | Typed facts/roster, effective state, revisions, and undo pass tests. A new synthetic officer fact set in `#testing` appeared in the next answer, then undo removed it; the database confirmed zero active test/model-name fact rows. A model-identity fact attempt was acknowledged from the trusted Qwen runtime setting and made no club-state revision. | The earlier incorrect Claude conversation remains in historical Discord messages; new answers use runtime truth. |
| PETER-11 | Bounded, versioned style changes reach all reply paths. A private `be brief` request applied and then undid; a two-dial request asked for clarification. | Concision of one capacitor reply was weaker than desired. |
| PETER-12 | A dedicated Debian 12 worker VM has a host-only runner control link and default-deny worker egress. Persistent P910 and guest firewall rules allow only the authenticated broker path. The restricted real worker passed 31/31 Rust, firewall, isolation, and pinned Hermes checks, including broker 401. | Host/guest operator access remains privileged by design. |
| PETER-13 | Controlled wheel/crate broker passed 90 dedicated tests. The deployed `08c8968` P910 restricted worker passed 6 offline package/cache checks; real broker reachability and rejection were included in the 31-check VM smoke against gateway `2800d97`. | Failed remote package fetches may repeat before the call cap; quotas bound the effect. |
| PETER-14 | Scoped, versioned project manifests/blobs and partial recovery pass tests. Live public Rust `main.rs` and `README.md` attachments were downloaded and independently compiled/tested inside a fresh restricted worker (6 checks). A private `counter.py` project was edited, run, attached, and continued after restart. | Files are scoped to their requester/audience. |
| PETER-15 | Source-bound announcement outbox, nonce, rate limit, receipt, and reconciliation pass tests. One explicit private-officer instruction sent once to the configured `#testing` destination; the outbox shows `sent: 1`, no unknown receipt. | No release-test post was sent to public `#announcements`. |
| PETER-16 | Online state backup, manifest verification, separate staging restore/diagnosis, and read-only health/retention diagnostics ran successfully on P910. The same private housekeeping command is scheduled weekly. The dry run selected zero current items for deletion. | The first scheduled cron execution has not occurred; retention apply remains a deliberate operator action. |
| PETER-17 | Current Python suite: **1,115 passed, 2 declared optional skips**, one third-party `audioop` deprecation warning. Hosted unit and all image checks passed. The deployed gateway/worker passed **31/31** Rust/isolation and **6/6** package checks. Live Discord checks now include the screenshot's Qwen correction, a natural `remember that` request that created a real club-memory row, and cleanup of that synthetic note. | A separate nonofficer Discord identity was unavailable; an earlier current-source answer cited the official blog domain rather than the exact article URL. |

## Live service and rollout

- P910 gateway image: `peterbot-hermes-gateway:2800d97`, image ID
  `sha256:1d572916ee68183b01a366658961814994cb68b0372e909e4a6de49a86f111c6`.
  The gateway-only typo hotfix followed code revision `08c8968`; the guest runner
  remains `peterbot-hermes-runner:08c8968` (image ID
  `sha256:7feb2d9b31c9f381925de0175a3d7501d614946ff63dfbc22364fe77cf93f82d`)
  and worker remains `peterbot-hermes-worker:08c8968` (image ID
  `sha256:d49928e91a50eebc05c7bad67f52a7d98b7108d3a6e0d539844ccb78c4e43027`).
  All revision labels and transferred guest image IDs were verified.
- `officer_only=false` and `member_work_enabled=true` after the isolated VM
  gate. Natural addressing is configured in `#testing`, `#officers`, `#general`,
  `#pc-help`, `#off-topic`, `#projects`, and `#meeting-plans`. The private
  control channels are `#officers` and `#testing`, with fresh officer
  role checks on every mutation. The member chat allowance is 20 requests per
  minute and the guild allowance is 120; announcement destinations remain
  `#testing` and `#announcements`.
- The protected former image/config set is at
  `/mnt/NVME/docker/appdata/peterbot/backups/cutover-config-20260923T1030Z`.
  Fresh online weekly snapshots were verified and separately restored before
  member rollout and after the Qwen/memory follow-up. The latest is
  `/mnt/NVME/docker/appdata/peterbot/backups/weekly-20260923T213656Z`.
  Later image switches require another fresh snapshot.
- [Public Rust source attachment](https://discord.com/channels/1306793423256420352/1308190621084946494/1552298031590940748),
  [usage note](https://discord.com/channels/1306793423256420352/1308190621084946494/1552298028218449970),
  and [private continuation thread](https://discord.com/channels/1306793423256420352/1552302889278382204)
  are live test receipts. The public files were verified by SHA-256 and execution
  independently of Peter's own completion claim.

## Final gate record

After the final hotfix, the authenticated diagnostic reported revision `2800d97`,
Discord/model/runner/queue all `ready`, and no queued, running, or unknown-cleanup
work. Live `Peter, you're Qwen bro, not Claude` got a casual correct Qwen reply,
and the unpinged `Nah, Qwen under the hood` also got a reply. Replaying the
screenshot's two-message memory exchange, including `the model the powers you`,
produced `Qwen3.8 Flash Next under the hood` without a stale memory write.
An officer set, recalled, and undid a synthetic club fact in `#testing`.
A separate natural memory request used the worker, created one verified club
memory row, and its disposable note was soft-deleted with the audit revision
retained. The restricted VM smoke returned 31 pass/0 fail/0 skip; the package
smoke returned 6 pass/0 fail.
The [cutover runbook](p910-cutover.md)
documents single-gateway deployment and rollback.
