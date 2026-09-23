# P910 redesign cutover runbook

The initial cutover completed September 23, 2026. The member rollout is active in seven configured channels. Use this runbook for a later image/config switch or rollback; the [release evidence](release-evidence.md) records the completed gates and limits. Keep a single Discord gateway connected to the bot token throughout.

## Before the switch

1. Record the exact Git SHA, CI run, gateway/runner/worker image digests and revision labels, served model ID, and active queue count. Wait for active work to finish or interrupt it through the verified cancellation path.
2. Verify the worker firewall and VM boundary required for the selected rollout stage. Keep ordinary member execution disabled until its isolation checks pass inside the real worker after reboot.
3. Take a fresh online appdata snapshot with `deploy/state_backup.py`, copy it outside the state directory, verify its manifest, and perform a separate restore check. The [September 22 baseline](p910-baseline-2026-09-22.md) is an earlier recovery point, not a substitute for a fresh cutover backup.
4. Save protected copies of the deployment Compose file, `.env`, `config.production.json`, and `hermes.production.json` on P910 with restrictive permissions. Do not print their contents or move secrets into Git. Record checksums for comparison.
5. Build the gateway, runner, and worker images with the candidate SHA as `PETERBOT_REVISION`. Run deterministic tests, the pinned Hermes fixture in the actual worker image, a synthetic broker/tool/artifact smoke, and isolation probes in the restricted boundary. Stage images without a second gateway connection.

## Switch and test

1. Stop the old `peterbot` gateway and verify it has disconnected. Confirm the guest runner is healthy, then start exactly one new gateway with the staged VM Compose overlay. Always specify `-p peterbot`: without it, Compose selects the `deploy` project and may leave the live gateway running. For a config-only change use `--no-deps --force-recreate peterbot`. Do not use `--remove-orphans` on later switches; it removed an unrelated stale model container at initial cutover. The old P910 host runner remains stopped.
2. Check Discord readiness, inference model identity, runner health, queue consumer/age, worker firewall, and absence of orphan workers. Use the authenticated `/diagnostics` command in [ops-and-retention.md](ops-and-retention.md) to distinguish model, runner, and queue failures; a process health check alone does not prove a complete reply.
3. Use the private `#testing` channel for a natural greeting, unpinged reply, factual club question, current research with a cited source, Rust compile/test and real file delivery, queued second requester, cancellation, project continuation, and a test-destination announcement. Record Discord message links, timings, files, and failures. Test private officer controls in the configured private channel without publishing test data to a public destination.
4. Confirm the desired audience flags and listen channels from the protected Hermes config. The September 23 rollout sets `officer_only=false` and `member_work_enabled=true` after a 31/31 restricted-VM isolation smoke. Private `#officers` and `#testing` are the configured control channels; each mutation still requires a fresh officer role and private-channel check. Run representative warm-backend latency repetitions and record the actual server/model configuration separately from queue delay.

## Rollback

1. Stop only the new `peterbot` gateway; do not allow both versions to use the token. Preserve post-cutover SQLite state and diagnostic logs separately before restoring anything.
2. Restore the protected Compose/environment/configuration snapshot and previous image tags. The initial pre-cutover config/image set is in `/mnt/NVME/docker/appdata/peterbot/backups/cutover-config-20260923T1030Z`; verified state snapshots are separate, including the weekly snapshot. If a schema migration prevents the previous image from reading new state, restore the verified pre-cutover state only after recording any accepted work since cutover for manual reconciliation.
3. Recreate one old gateway, verify Discord and inference readiness, and run the private `#testing` greeting. Reconcile uncertain delivery/outbox records instead of replaying them blindly.

This runbook does not authorize a public announcement by itself. A specific current officer request in the private control channel supplies that action's authority.
