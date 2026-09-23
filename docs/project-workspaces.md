# Project workspaces (PETER-14) — trusted persistent project/file store

Module: `peterbot/project_store.py`. Tests: `tests/test_project_store.py`.
The gateway now creates, saves, and restores these stores for project tasks.
The integration details below describe the current path.

## What it is

A durable, audience-scoped store that lets a task save bounded source/result
files with a manifest, and lets a later authorized continuation retrieve the
exact bytes into a fresh disposable worker after a gateway restart. Design
points:

- **Manifest** lives in SQLite (`<root>/projects.sqlite`, WAL): projects,
  explicit grants, versions, per-file entries (name, size, sha256), task→project
  bindings, and an append-style event log.
- **Blobs** live in `<root>/blobs/<hh>/<sha256>`, content-addressed, written
  with `O_CREAT|O_EXCL|O_NOFOLLOW` at mode `0600`, fsynced, then atomically
  `os.replace`d. Directories are `0700`.
- **Trust boundary**: only the trusted gateway opens a `ProjectStore`. Workers
  hand bytes to the gateway/runner; the store never executes, imports, or
  deserializes task artifacts on the host.
- **Adapter shape** is a validated filename→bytes mapping. **Archives are not
  accepted at all** — no tar/zip path exists in the store, so untrusted tar
  members and decompression bombs have no entry point. This is a deliberate,
  honest interface limitation, not a TODO.

## Audience and ownership model

Every method takes a gateway-constructed `Principal` (from `agent_policy`).
There is no owner/guild/channel override argument.

- Default `private`: exact **guild + owning user + channel** must match.
  No implicit club-wide or same-guild sharing. Other users, other channels,
  and other guilds get the same `ProjectDenied` as a nonexistent id (no
  existence oracle).
- `shared` requires an explicit `share()` grant per user, still bound to the
  project's guild and channel. Only the owner can grant/revoke.
- `relocate()` moves the single audience channel (e.g. a new continuation
  thread). The old channel loses access immediately.
- A task id is bound to exactly one project (`task_projects`), enforced under
  the unique index and across processes. Replaying a task id against a
  different project — restore or save — raises `ProjectDenied`. This blocks
  cross-task manifest forgery.
- `check_access(principal, project_id)` is the standalone revocation check.
  Nothing caches authorization; grants, revocations, relocation, and deletes
  take effect on the next call.

## Adapter API

```python
from peterbot.project_store import ProjectStore, ProjectSettings

store = ProjectStore(state_dir / "projects")   # one instance in the gateway

store.create_project(p, name="edigits", task_id=job_id) -> dict
store.save(p, project_id, task_id=..., files={"src/main.rs": b"..."},
           provenance="task <id> ...", verified=True,
           dependency_instructions="cargo --offline build",
           best_effort=False) -> dict    # one full-snapshot version
store.list_projects(p) / store.describe(p, project_id) -> dict
store.check_access(p, project_id) -> dict          # revocation gate
store.list_files(p, project_id, version=None) -> dict
store.read_file(p, project_id, name, version=None) -> bytes
store.verify(p, project_id, version=None) -> dict  # re-hash everything
store.restore(p, project_id, task_id=..., version=None) -> dict  # exact bytes
store.worker_payload(p, project_id, task_id=...) -> dict  # JSON-safe, b64 files
store.share(p, project_id, user_ids=[...]) / revoke(p, project_id, user_id=...)
store.relocate(p, project_id, channel_id=...)
store.delete_project(p, project_id) -> dict        # owner purge
store.retention_sweep() -> dict                    # operator/gateway housekeeping
```

`save()` records a version as `verified` (completed task) or `partial`
(timeout/cancellation salvage; caller passes `verified=False`). With
`best_effort=True`, individually invalid entries are skipped and reported under
`rejected` so already-collected valid files survive teardown; strict mode fails
closed. Each version is a full snapshot of the project's user-relevant files.

`restore()`/`worker_payload()` return the manifest's `state`, `provenance`,
and `dependency_instructions` so the continuation prompt can honestly say
"these are partial/unverified files from a timed-out task, built with X".

## Limits (defaults in `ProjectSettings`)

| Limit | Default |
| --- | --- |
| File size | 2 MiB |
| Files per version | 64 |
| Version bytes | 8 MiB (matches runner `MAX_ARTIFACT_BYTES`) |
| Project bytes | 32 MiB |
| Per-(guild,user) bytes | 128 MiB |
| Projects per user | 50 |
| Versions per project | 8 (oldest evicted + blobs GCed) |
| Retention | 90 days |

Callers cannot widen these per save; the operator constructs the store with a
`ProjectSettings` instance once.

## Rejected inputs

- Absolute paths, `..`/dot segments, `//`, `.\`, `:`, trailing dot/space
  segments, control/format/unassigned/surrogate characters, non-NFC names,
  Windows device names, >240 chars, >255-byte segments, >16 path parts.
- Duplicate or case-folded-colliding names in one save; a name that is also a
  directory prefix of another (`a` and `a/b`).
- Anything that isn't `bytes`; strings rejected (no implicit encoding).
- Archive/compressed inputs by extension (`.zip .tar .gz .zst .7z .rar .deb
  .rpm .jar .iso …`) and by magic prefix (zip/gzip/bzip2/xz/lzma/7z/rar/zstd/
  ar/rpm, plus the ustar header at offset 257). A renamed bomb trips the magic
  check; nothing is ever decompressed.
- Oversized files, versions, projects, and per-user totals; version-count and
  project-count caps.
- Non-principal callers; wrong guild/user/channel; forged task references.

On restore/read, each blob must be a regular non-symlink file with the exact
manifest size and sha256; otherwise `ProjectIntegrityError` and no bytes are
returned. Symlinked or swapped blobs fail the `O_NOFOLLOW`/`fstat`/hash checks.

## What is *not* promised

- **Abrupt host loss** between blob writes and the manifest commit loses that
  version. The bytes stay behind as unreferenced blobs; `retention_sweep()`
  reclaims unreferenced regular blobs (and stale `*.tmp.*` files) only past a
  one-hour grace window, so a concurrent in-flight save's fresh blob always
  survives the race, and non-hex or symlinked entries are never touched. A
  timeout/cancellation only preserves files the runner actually collected
  before teardown, marked `partial`; unsnapshotted work is gone. This matches
  the PETER-14 contract.
- No cross-process file lock: use **one** `ProjectStore` per root (the gateway
  process). The task-bind unique index still makes cross-instance forgeries
  fail closed.
- No de-dup across users is leaked: identical bytes share a blob, but a blob
  is only readable through a manifest row the principal can see, and GC only
  runs when *no* manifest references it (tested).

## Gateway and runner integration

The trusted gateway in `peterbot/hermes_gateway.py` owns `ProjectStore`. It
creates or links a project to a job, saves verified worker files as a new
version, and can preserve valid partial files after a timeout or cancellation.
Discord attachment delivery is separate from this durable copy.

For a continuation, the gateway checks the requester's current access, binds
the new task to the saved project, and passes `worker_payload()` files to a
fresh disposable worker. A missing or revoked project fails the continuation
instead of silently starting with an empty workspace. The worker cannot supply
its own `Principal` or grant access by mentioning a project ID in model output.

Operator retention is a separate, explicit path: see
[operations and retention](ops-and-retention.md). It never runs as part of an
ordinary member turn.

## Security invariants (tested)

- Visibility is enforced in SQL before row content enters Python; bytes are
  re-verified against sha256/size at read time.
- Revocation is immediately effective; callers re-check before staging.
- Restrictive permissions and atomic writes; no `chmod` of foreign paths.
- The store holds no secrets, capabilities, caches, or worker credentials —
  only task files and their manifest.

## Rollback / migration

- First start creates `<root>/projects/` (`projects.sqlite`, `blobs/`);
  nothing existing reads them. To roll back code, leave this directory dormant
  so accepted member files remain recoverable; archive it before any deletion.
  No other table, config, or file is touched.
- The SQLite `user_version` is 1. A newer schema is refused on open instead of
  being rewritten by older gateway code. Future migrations must preserve a
  verified backup and advance this version with the schema change.
- Backup: the SQLite file supports the backup API (PETER-16); copy blobs only
  from a quiet moment or re-verify with `verify()` after restore, because the
  manifest is the authority on which blobs matter.
