# Operator operations, retention, and snapshot verification (PETER-16)

`deploy/housekeeping.py` is the operator CLI for the state directory. In the
live layout, `foreground.sqlite3` and reminders are under `data/`, while
`tasks.sqlite3`, `memory.sqlite3`, `club.sqlite3`, `style.sqlite3`,
`conversations.sqlite3`, `announcements.sqlite3`, `metrics.sqlite3`, and
`projects/` are under `data/hermes/`. The CLI also accepts the Hermes directory
directly for focused checks. It performs no network, model, or Discord calls; nothing in this slice
can post publicly.

## Commands

    python deploy/housekeeping.py diagnose [STATE_DIR]
    python deploy/housekeeping.py retention [STATE_DIR]           # dry-run, always
    python deploy/housekeeping.py retention [STATE_DIR] --apply \
        --backup-destination DIR [--conversations-days N] [--metrics-days N]
        [--terminal-jobs-days N] [--settled-receipts-days N] [--include-projects]
    python deploy/housekeeping.py check-snapshot SNAPSHOT STAGING

`STATE_DIR` defaults to `$PETERBOT_STATE_DIR` (`/app/peterbot-data` in the
container). All output is JSON on stdout with aggregate counts, status
histograms, ages, schema versions, and fixed reason tokens only — never
prompts, answers, memory text, Discord IDs, job IDs, tokens, filenames, or
private transcript content. `revision` reports `$PETERBOT_REVISION` (the image
label) or `unknown`.

### Component health

Each store is reported `ok`, `offline` (file absent), or `degraded` with one
fixed reason: `schema_newer` (a `user_version` above what this code writes —
usually a rolled-back deploy), `missing_tables` (a valid database that is not
a Peter store), or `corrupt_or_unreadable`. Stores are opened read-only; a
live gateway is not disturbed. `foreground.cleanup_unknown` counts workers
whose cleanup was never confirmed — the queue-holds-open condition an operator
must reconcile manually. `jobs.oldest_active_age_seconds` measures only
`preparing`/`queued`/`running` rows.

### Retention

Nothing is deleted unless `--apply` is given, and `--apply` refuses to run
without `--backup-destination`: it first writes and verifies a snapshot (see
below) and aborts before any deletion if that check fails. It also refuses if
a store it would delete from is `degraded`; `offline` stores are skipped and
listed under `skipped`. Defaults: conversations 90 days, metrics 30, terminal
jobs 90, settled announcement receipts 180.

Deleted (only when older than the TTL):

| Category | Rows |
| --- | --- |
| `conversations` | final `conversation_turns` rows past TTL |
| `metrics` | `stage_metrics` rows past TTL |
| `terminal_jobs` | terminal jobs with `delivered=1` and delivery `delivered` or `withheld` |
| `settled_receipts` | outbox rows with status `sent` |

Never deleted: jobs in `preparing`/`queued`/`running`, any job whose delivery
is `pending`, `delivering`, `unknown`, or `exhausted` (an unknown send is
operator-reconciled, never silently dropped), outbox rows that are not settled
`sent` receipts (`pending`, `preparing`, `sending`, `unknown`), and all club,
style, and memory state and audit revisions regardless of age.

Projects are pruned only with `--include-projects`, which delegates to
`ProjectStore.retention_sweep()` under its own `ProjectSettings.retention_days`
(90-day) policy — it ages non-latest versions of live projects and GCs blobs
that are no longer referenced. The dry-run plan mirrors that sweep's
eligibility without mutating; the CLI never issues ad-hoc SQL against active
project data.

### Soft-forget vs deletion

`/forget` is a soft-forget: the memory row is flagged `deleted` and previous
content remains in the append-only `memory_revisions` ledger (trigger-enforced
— no UPDATE/DELETE is possible on it). `diagnose` reports `active`,
`soft_forgotten`, and `revisions` counts separately. Retention performs no
memory deletion of any kind. The only true deletion path for forgotten memory
content is destroying a snapshot that predates it: rotate old snapshots once
they outlive your recovery needs (`rm -rf` the snapshot directory as the state
owner). Do not copy snapshots to shared or world-readable locations.

### Snapshot verification with project-blob cross-check

`deploy/state_backup.py backup` uses the SQLite online backup API, so it runs
against the live gateway; `-wal`/`-shm`/`-journal` sidecars are intentionally
not copied. `verify` checks the manifest digests *and* cross-checks every
`projects.sqlite` manifest `files` entry against the snapshot's blobs: a
manifest referencing a missing blob or a blob whose content no longer matches
its recorded SHA-256 fails with `missing or mismatched blob` (this catches a
GC interrupted mid-sweep in the live store before it becomes a useless
snapshot). `check-snapshot` runs `verify`, restores into a new staging
directory, and diagnoses the copy: queued jobs, memory, project manifests and
bytes, and outbox `unknown` receipts must survive the round-trip — and a
restore never replays an uncertain send; `unknown` stays `unknown` until a
protected operator checks Discord and explicitly reconciles the ledger.

## p910 scheduling and permissions

The gateway container runs as uid/gid 10000 with a read-only root filesystem
and the state bind-mounted at `/app/peterbot-data`
(`${PETERBOT_APPDATA}/data` on the host, e.g.
`/mnt/NVME/docker/appdata/peterbot/data`). The p910 SSH user has libvirt,
Docker, and app access but no passwordless sudo. Create the backup parent as
the operator, then run the image as the gateway UID with supplemental `apps`
group access — no host-level chowns or sudo:

    install -d -m 2770 -g apps "$PETERBOT_APPDATA/backups"
    apps_gid=$(getent group apps | cut -d: -f3)
    docker run --rm --network none --read-only --user 10000:10000 \
      --group-add "$apps_gid" \
      -v "$PETERBOT_APPDATA/data:/state:ro" \
      -v "$PETERBOT_APPDATA/backups:/backups" \
      --entrypoint python "$PETERBOT_GATEWAY_IMAGE" -I \
      /app/deploy/housekeeping.py diagnose /state

Snapshot destinations must be outside `/state` (the tool refuses nesting).
Snapshots land as `0700` directories of `0600` files owned by uid 10000.
The image now
contains `/app/deploy/housekeeping.py` and `state_backup.py`. Run the one-off
container as uid 10000 with the host `apps` group added so it can read the
gateway state and write to an operator-owned, group-writable backup directory
without opening that directory to everyone. Schedule via
the operator's own `crontab -e` (no sudo needed), e.g. weekly `diagnose` plus
`backup` + `check-snapshot` into a dated directory; write cron stdout to a
private log (`>> "$PETERBOT_APPDATA/logs/housekeeping.log" 2>&1`, the log
directory is group-private `2770`). `deploy/p910-housekeeping.sh` is the concrete
weekly command: it takes one online snapshot, verifies it, then prints private
diagnostics and a retention **dry run** using the currently running gateway
image. It uses no network and does not delete live state. After the release
image is deployed and the script is staged in the appdata `deploy/` directory,
the operator's crontab can run it, for example:

    17 7 * * 1 umask 077; /mnt/NVME/docker/appdata/peterbot/deploy/p910-housekeeping.sh >> /mnt/NVME/docker/appdata/peterbot/logs/housekeeping.log 2>&1

The report contains no secrets, but it is still operator-facing — keep it
inside appdata. Snapshot rotation remains an explicit operator action.

Do NOT schedule `retention --apply`. It is a deliberate, human-reviewed
operation: read the dry-run plan, confirm the eligible counts look right,
then run apply once with a fresh `--backup-destination`. After the plan check,
also run `check-snapshot` on a prior backup so the rollback path is known-good
before any deletion.

`diagnose` on the live state opens databases read-only (`mode=ro`) and never
mutates; `backup` uses the online backup API and is safe with the gateway
running. `check-snapshot` staging restores are throwaway: diagnose the copy,
then delete the staging directory. `restore` over a live state requires the
gateway stopped and an empty (or absent) destination — never restore in place.

## Rollback

1. `retention --apply` already left a verified snapshot at
   `--backup-destination` (taken moments before deletion, blobs
   cross-checked). Treat it as the immediate rollback point.
2. To confirm it: `check-snapshot SNAPSHOT STAGING` (gateway may stay up;
   staging is a copy). It must report the pre-deletion counts and a healthy
   component set.
3. To roll back for real: stop the gateway container, move the current state
   directory aside (do not delete it — it may contain accepted work since the
   snapshot), `restore SNAPSHOT /path/to/new/data`, then start the gateway.
   Restore refuses a non-empty destination, so the aside-move is enforced by
   the tool.
4. After a restore, reconcile rather than replay: outbox records that come
   back `unknown` (and jobs with `unknown`/`exhausted` delivery) still require
   officer reconciliation in the private control channel before anything is
   re-sent. A restored `unknown` receipt must never trigger an automatic
   resend.
5. If deletion was a mistake but nothing else regressed, the lighter option is
   to leave the newer state running and keep the snapshot archived; retention
   removals are only rows already past their TTL, and nothing outside
   `conversations`, `metrics`, terminal delivered/withheld jobs, and settled
   receipts is ever removed.
