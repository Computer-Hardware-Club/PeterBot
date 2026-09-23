# Peter redesign release evidence

Status: September 23, 2026. The `PETER-xx` keys map to the supplied local
backlog, not published GitHub issues. The implementation is the draft stacked
[PR #3](https://github.com/Computer-Hardware-Club/PeterBot/pull/3), based on
`feat/hermes-peter` (`332d366`). Code revision `62e2b9f` is deployed on P910
for testing; it has not been merged. The hosted [push CI run](https://github.com/Computer-Hardware-Club/PeterBot/actions/runs/35882617172)
passed; the [PR CI run](https://github.com/Computer-Hardware-Club/PeterBot/actions/runs/35882623041)
passed on its second attempt after an unrelated queue-test timing race in the
first attempt. The test now waits for actual queue events.

| Backlog | Evidence | Remaining limit |
| --- | --- | --- |
| PETER-01 | The [pre-cutover baseline](p910-baseline-2026-09-22.md) separates Discord, model, runner, queue, state, and firewall health. The live authenticated `/diagnostics` endpoint later reported each dependency ready. A real `#testing` greeting, brokered work, and delivered files passed. | The original reported outage was not reproduced in the baseline. |
| PETER-02 | PR #1 changes were reconciled into foundation PR #2; PR #3 is stacked on it. CI runs the ordinary suite, compile/config checks, pinned Hermes fixture, and all three image builds without production secrets or automatic deployment. | Human review and merge remain separate. |
| PETER-03 | Preparing-job race, idempotent ingress, terminal delivery cursors, unknown receipts, cancel/complete races, and restart recovery have tests. An old four-row P910 jobs snapshot migrated without replay. Live `/cancel_task` stopped a sleeping worker, edited its status to cancelled, and left no active slot. | No claim of exactly-once Discord delivery under arbitrary outages. |
| PETER-04 | A durable global foreground scheduler covers chat, `/ask`, `/recap`, and worker work. During live `#testing`, a second request received a truthful one-ahead acknowledgement and answered after the first worker completed. | A live two-human-user race was not available; deterministic tests cover separate identities. |
| PETER-05 | Tiered model budgets, a shared turn deadline, safe rescue, and explicit-work handoff postcondition pass tests. Five warm-idle repetitions per case on the actual Qwen/vLLM server are in [model-latency.md](model-latency.md). Explicit coding handoff improved from 44.173 s to 2.462 s p95 in model-only probes, with 5/5 valid routes. The previous gateway revision routed a real coding request in 3.562 s, then delivered its compiled `ready.rs` file. | Previous model-routed greeting p50 was 2.007 s against a proposed 2 s target; bare greetings now skip that model call. End-to-end timing includes Discord and worker time. |
| PETER-06 | Name, mention, reply, and short unpinged follow-up routing pass tests. Bare greetings now return one or two words with no model call. Live `hey peter` replied `yo`, and a later `Yo Peter` replied `whats good`. Unpinged follow-ups still work. | Unrelated chatter remains intentionally ignored. |
| PETER-07 | Conversation turns persist by guild, requester, channel, and audience, independent of Discord transport. The private task thread continued its saved work after a gateway restart. | Private/public context is not merged into one transcript. |
| PETER-08 | One editable presence/status message, real stages, and owner cancellation are wired. The voice update renders verified worker stages as short playful text plus elapsed time. Live `*letting the compiler judge me* (40s)` edited the existing status; that timed task completed and delivered `voice-check.txt`. Earlier live cancellation finalized its original status and stopped the worker. | Discord may show separate attachment messages for files. |
| PETER-09 | Private officer controls bind intent to the current Discord source and fresh member roles. A real `#officers` fact/style/announcement request worked; public text and nonofficer role paths are rejected in tests. | We did not use a second nonofficer Discord account for a live denial. |
| PETER-10 | Typed facts/roster, effective state, revisions, and undo pass tests. A synthetic officer fact appeared in the next `#testing` answer; undo removed it. | One later model answer inaccurately described the earlier, then-valid fact as a mistake; current state was correct. |
| PETER-11 | Bounded, versioned style changes reach all reply paths. A private `be brief` request applied and then undid; a two-dial request asked for clarification. | Concision of one capacitor reply was weaker than desired. |
| PETER-12 | A dedicated Debian 12 worker VM has a host-only runner control link and default-deny worker egress. Persistent P910 and guest firewall rules allow only the authenticated broker path. The restricted real worker passed 31/31 Rust, firewall, isolation, and pinned Hermes checks, including broker 401. | Host/guest operator access remains privileged by design. |
| PETER-13 | Controlled wheel/crate broker passed 90 dedicated tests. The `62e2b9f` P910 restricted worker passed 6 offline package/cache checks; real broker reachability and rejection were included in the 31-check VM smoke. | Failed remote package fetches may repeat before the call cap; quotas bound the effect. |
| PETER-14 | Scoped, versioned project manifests/blobs and partial recovery pass tests. Live public Rust `main.rs` and `README.md` attachments were downloaded and independently compiled/tested inside a fresh restricted worker (6 checks). A private `counter.py` project was edited, run, attached, and continued after restart. | Files are scoped to their requester/audience. |
| PETER-15 | Source-bound announcement outbox, nonce, rate limit, receipt, and reconciliation pass tests. One explicit private-officer instruction sent once to the configured `#testing` destination; the outbox shows `sent: 1`, no unknown receipt. | No release-test post was sent to public `#announcements`. |
| PETER-16 | Online state backup, manifest verification, separate staging restore/diagnosis, and read-only health/retention diagnostics ran successfully on P910. The same private housekeeping command is scheduled weekly. The dry run selected zero current items for deletion. | The first scheduled cron execution has not occurred; retention apply remains a deliberate operator action. |
| PETER-17 | Voice-update Python suite: **1,087 passed, 2 declared optional skips**, one third-party `audioop` deprecation warning. Native AMD64 gateway/runner/worker images carry revision `62e2b9f`; the restricted worker passed **31/31** Rust/isolation and **6/6** package checks. Live Discord greetings, current-source research, Rust file delivery, project continuation, officer controls, queue, cancellation, a test announcement, and the new playful status were exercised. | A separate nonofficer Discord identity was unavailable; current-source research gave the official blog domain rather than the exact article URL. |

## Live service and rollout

- P910 gateway image: `peterbot-hermes-gateway:62e2b9f`, image ID
  `sha256:20a6f49f8b1cbaaa3d39aad3392fd17fcdd86fd0026e9ff5fac72af256c670ea`.
  Guest runner image ID is `sha256:7492b7d4560b8bc6f428258e25fd833549afd198cdf93fdd19fd7c8c7d5c3197`;
  worker ID is `sha256:0b165a1ad3dbe304b94785fa4653da5efc018934d4d8a8fc81674aae0b6cb462`.
  All three labels report revision `62e2b9f`; transferred guest image IDs match
  their host builds.
- `officer_only=false` and `member_work_enabled=true` after the isolated VM
  gate. Natural addressing is configured in `#testing`, `#officers`, `#general`,
  `#pc-help`, `#off-topic`, `#projects`, and `#meeting-plans`. The private
  control channel stays `#officers`; announcement destinations stay configured
  as `#testing` and `#announcements`.
- The protected former image/config set is at
  `/mnt/NVME/docker/appdata/peterbot/backups/cutover-config-20260923T1030Z`.
  Fresh online weekly snapshots were verified and separately restored before
  member rollout and after the voice update. The latest is
  `/mnt/NVME/docker/appdata/peterbot/backups/weekly-20260923T172437Z`.
  Later image switches require another fresh snapshot.
- [Public Rust source attachment](https://discord.com/channels/1306793423256420352/1308190621084946494/1552298031590940748),
  [usage note](https://discord.com/channels/1306793423256420352/1308190621084946494/1552298028218449970),
  and [private continuation thread](https://discord.com/channels/1306793423256420352/1552302889278382204)
  are live test receipts. The public files were verified by SHA-256 and execution
  independently of Peter's own completion claim.

## Final gate record

After the voice switch, the authenticated diagnostic reported revision `62e2b9f`,
Discord/model/runner/queue all `ready`, and no queued, running, or unknown-cleanup
work. On the voice revision, `hey peter` got `yo`, `Yo Peter` got `whats good`,
and an ordinary capacitor answer had no em dash. The timed sandbox task edited
one status line to `*letting the compiler judge me* (40s)`, then delivered
`voice-check.txt`. An earlier Rust request on revision `274c86a` produced and
attached `ready.rs`; its private routing counter was 3,562 ms. The voice-update restricted VM smoke
returned 31 pass/0 fail/0 skip; the package smoke returned 6 pass/0 fail.
The [cutover runbook](p910-cutover.md)
documents single-gateway deployment and rollback.
