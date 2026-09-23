"""Trusted, durable, audience-scoped project/file store (PETER-14).

Only the trusted gateway opens this store. Workers hand file *bytes* to the
gateway/runner; the store never executes, imports, or deserializes task
artifacts on the host. Blobs are content-addressed (sha256), written atomically
with owner-only permissions, and referenced by a SQLite manifest that binds
every project to a guild, an owning user, one channel audience, and the task
that created it.

Access model (enforced in SQL before any row or blob content is read, and
re-checked by ``check_access()`` immediately before a caller stages files into
a worker):

- Default audience is ``private``: exact guild + user + channel must match.
  No implicit club-wide or same-guild sharing.
- ``shared`` requires an explicit grant row for the requesting user, still
  inside the project's guild and channel.
- A task id may be bound to at most one project. A forged continuation that
  replays another task's id against a different project is denied, which also
  blocks cross-task manifest forgery.

Integrity: a stored file is restored only if the on-disk blob is a regular
non-symlink file whose size and sha256 match the manifest. Archive/compressed
inputs are rejected outright (extension and magic), so decompression bombs and
untrusted tar members have no entry point; the adapter API is a validated
filename→bytes mapping, not an archive.

Known limits stated honestly: an abrupt host crash between writing a version's
blobs and committing its manifest rows loses that version (orphan blobs are
garbage-collected); a timeout/cancellation can save already-collected valid
files as a ``partial`` version, but work not yet collected is gone. Rollback of
this module is data-safe: drop the table set and blob tree; nothing else
references them.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sqlite3
import stat
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from .agent_policy import Principal, _valid_id

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_PROJECT_ID = re.compile(r"\A[0-9a-f]{32}\Z")

# Extensions that can only be an archive/container: rejected before hashing.
_DENIED_SUFFIXES = frozenset({
    ".zip", ".tar", ".tgz", ".tbz2", ".txz", ".gz", ".bz2", ".xz", ".lz",
    ".lz4", ".zst", ".7z", ".rar", ".jar", ".war", ".apk", ".whl", ".egg",
    ".deb", ".rpm", ".cab", ".iso", ".lzh", ".arj", ".z", ".sz", ".cpio",
})
# Magic prefixes for archive/compressed streams. Never executed, never
# extracted; rejecting them removes the bomb surface entirely.
_MAGICS = (
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",       # zip
    b"\x1f\x8b",                                        # gzip
    b"BZh",                                             # bzip2
    b"\xfd7zXZ\x00", b"\x04\x22\x6d\x18\x02\x00",       # xz, lzma
    b"7z\xbc\xaf\x27\x1c",                              # 7-zip
    b"Rar!\x1a\x07",                                    # rar
    b"\x28\xb5\x2f\xfd",                                # zstd
    b"!<arch>\n",                                       # ar (deb/rpm container)
    b"\xed\xab\xee\xdb",                                # rpm
)
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)


class ProjectError(ValueError):
    """Base class for project-store rejections."""


class ProjectViolation(ProjectError):
    """A filename, byte payload, or manifest field failed validation."""


class ProjectQuota(ProjectError):
    """A per-file, per-version, per-project, or per-user quota was exceeded."""


class ProjectIntegrityError(ProjectError):
    """A stored blob no longer matches its manifest (tamper or corruption)."""


class ProjectDenied(PermissionError):
    """The trusted principal may not see or use this project at all."""


@dataclass(frozen=True)
class ProjectSettings:
    """Fixed ceilings. Callers cannot widen them per save."""
    max_file_bytes: int = 2 * 1024 * 1024
    max_files_per_version: int = 64
    max_version_bytes: int = 8 * 1024 * 1024      # matches runner MAX_ARTIFACT_BYTES
    max_project_bytes: int = 32 * 1024 * 1024
    max_user_bytes: int = 128 * 1024 * 1024
    max_projects_per_user: int = 50
    max_versions_per_project: int = 8
    max_name_chars: int = 240
    max_path_parts: int = 16
    retention_days: int = 90
    max_provenance_chars: int = 4000
    max_dependency_chars: int = 2000
    max_project_name_chars: int = 120
    max_task_id_chars: int = 64


def _clean_text(value: object, limit: int, label: str, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ProjectViolation(f"{label} must be a string")
    for ch in value:
        if ord(ch) < 32 or ord(ch) == 127 or unicodedata.category(ch) in {"Cc", "Cf"}:
            raise ProjectViolation(f"{label} contains control or format characters")
    text = value.strip()
    if required and not text:
        raise ProjectViolation(f"{label} is required")
    if len(text) > limit:
        raise ProjectViolation(f"{label} exceeds {limit} characters")
    return text


def _validate_filename(raw: object, settings: ProjectSettings) -> tuple[str, str]:
    """Return (canonical, casefold-key) for a safe relative project path.

    Rejects absolute paths, traversal, backslashes, control/format characters,
    lone surrogates, dot-only segments, trailing dots/spaces, Windows device
    names, and overlong names. NFC stability is required so two visually
    identical encodings cannot collide.
    """
    if type(raw) is not str or not raw or len(raw) > settings.max_name_chars:
        raise ProjectViolation("filename must be a non-empty bounded string")
    if unicodedata.normalize("NFC", raw) != raw:
        raise ProjectViolation("filename is not NFC-normalized")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError:
        raise ProjectViolation("filename contains unpaired surrogates") from None
    if "\\" in raw or ":" in raw:
        raise ProjectViolation("filename contains a forbidden separator")
    for ch in raw:
        if ord(ch) < 32 or ord(ch) == 127:
            raise ProjectViolation("filename contains control characters")
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise ProjectViolation("filename contains invisible or unassigned characters")
    parts = raw.split("/")
    if raw.startswith("/") or len(parts) > settings.max_path_parts:
        raise ProjectViolation("filename must be a bounded relative path")
    for part in parts:
        # empty == '//'/leading/trailing slash; dots-or-spaces-only == '.', '..';
        # rstrip catches trailing dot/space segments.
        if (not part or set(part) <= {".", " "} or part != part.rstrip(". ")
                or len(part.encode("utf-8")) > 255
                or part.split(".")[0].upper() in _WINDOWS_DEVICES):
            raise ProjectViolation("filename contains an unsafe path segment")
    return "/".join(parts), raw.casefold()


class ProjectStore:
    """Durable manifest + content-addressed blob store. Gateway-trusted only.

    Every public method takes a freshly gateway-constructed ``Principal``;
    there is intentionally no owner/guild/channel override argument. The
    gateway must re-check ``check_access()`` immediately before handing
    restored bytes to a worker and again before any delivery.
    """

    def __init__(self, root: str | Path, *, settings: ProjectSettings | None = None,
                 clock: Callable[[], int] | None = None) -> None:
        self.settings = settings or ProjectSettings()
        self._clock = clock or (lambda: int(time.time()))
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.blob_dir = self.root / "blobs"
        self.blob_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.blob_dir, 0o700)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.root / "projects.sqlite",
                                     isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            schema_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if schema_version > 1:
                self._conn.close()
                raise ProjectIntegrityError("Project database is newer than this gateway")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects(
                    id TEXT PRIMARY KEY,
                    guild_id INTEGER NOT NULL,
                    owner_user_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    audience TEXT NOT NULL CHECK(audience IN ('private','shared')),
                    origin_task_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS projects_access
                    ON projects(guild_id, channel_id, owner_user_id, deleted);
                CREATE TABLE IF NOT EXISTS grants(
                    project_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    granted_by INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(project_id, user_id));
                CREATE TABLE IF NOT EXISTS versions(
                    project_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('verified','partial')),
                    provenance TEXT NOT NULL,
                    dependency_instructions TEXT NOT NULL,
                    file_count INTEGER NOT NULL,
                    total_bytes INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY(project_id, version));
                CREATE TABLE IF NOT EXISTS files(
                    project_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY(project_id, version, name));
                CREATE TABLE IF NOT EXISTS task_projects(
                    task_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    actor_user_id INTEGER NOT NULL,
                    detail TEXT NOT NULL DEFAULT '');
            """)
            if schema_version == 0:
                self._conn.execute("PRAGMA user_version=1")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- plumbing

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    @staticmethod
    def _require_principal(principal: Principal) -> None:
        if not isinstance(principal, Principal):
            raise ProjectDenied("A gateway-verified Principal is required")
        if principal.guild_id is None or not _valid_id(principal.guild_id):
            raise ProjectDenied("Projects are guild-bound")

    @staticmethod
    def _validate_project_id(project_id: object) -> str:
        if type(project_id) is not str or not _PROJECT_ID.match(project_id):
            raise ProjectViolation("project_id must be a canonical 32-hex id")
        return project_id

    @staticmethod
    def _validate_task_id(task_id: object, limit: int) -> str:
        return _clean_text(task_id, limit, "task_id", required=True)

    def _blob_path(self, digest: str) -> Path:
        if not _HEX64.match(digest):
            raise ProjectIntegrityError("Invalid blob reference")
        return self.blob_dir / digest[:2] / digest

    def _event(self, conn: sqlite3.Connection, project_id: str, kind: str,
               actor_user_id: int, detail: str = "") -> None:
        conn.execute("INSERT INTO events(ts,project_id,kind,actor_user_id,detail) "
                     "VALUES (?,?,?,?,?)",
                     (self._clock(), project_id, kind, actor_user_id, detail[:240]))

    # ------------------------------------------------------------ access checks

    _ACCESS_SQL = """SELECT p.* FROM projects p
        WHERE p.id=? AND p.deleted=0 AND p.guild_id=? AND p.channel_id=?
          AND (p.owner_user_id=?
               OR (p.audience='shared' AND EXISTS
                   (SELECT 1 FROM grants g WHERE g.project_id=p.id AND g.user_id=?)))"""

    def _project_row(self, conn: sqlite3.Connection, principal: Principal,
                     project_id: str) -> sqlite3.Row:
        # Visibility in SQL, before any project content enters Python. One
        # denial message for "missing" and "not yours": no existence oracle.
        row = conn.execute(self._ACCESS_SQL,
                           (project_id, principal.guild_id, principal.channel_id,
                            principal.user_id, principal.user_id)).fetchone()
        if row is None:
            raise ProjectDenied("This project does not exist or is not shared with you.")
        return row

    def _bind_task(self, conn: sqlite3.Connection, project_id: str, task_id: str) -> None:
        existing = conn.execute("SELECT project_id FROM task_projects WHERE task_id=?",
                                (task_id,)).fetchone()
        if existing is None:
            try:
                conn.execute("INSERT INTO task_projects(task_id,project_id) VALUES (?,?)",
                             (task_id, project_id))
            except sqlite3.IntegrityError:
                # Lost a bind race to another thread/instance: re-read under
                # this transaction's view and deny unless it bound the same
                # project. A silent success here would merge two projects.
                existing = conn.execute("SELECT project_id FROM task_projects WHERE task_id=?",
                                        (task_id,)).fetchone()
                if existing is None or existing["project_id"] != project_id:
                    raise ProjectDenied("This task is bound to a different project.") from None
        elif existing["project_id"] != project_id:
            raise ProjectDenied("This task is bound to a different project.")

    def check_access(self, principal: Principal, project_id: str) -> dict:
        """Standalone revocation check. Callers MUST invoke this immediately
        before staging files into a worker and before any delivery; grants and
        audience changes take effect on the next call, no cache exists here."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        with self._read() as conn:
            return dict(self._project_row(conn, principal, project_id))

    # ---------------------------------------------------------------- projects

    def create_project(self, principal: Principal, *, name: str, task_id: str) -> dict:
        self._require_principal(principal)
        task_id = self._validate_task_id(task_id, self.settings.max_task_id_chars)
        name = _clean_text(name, self.settings.max_project_name_chars, "name", required=True)
        project_id = os.urandom(16).hex()
        ts = self._clock()
        with self._write() as conn:
            own = conn.execute(
                "SELECT COUNT(*) c FROM projects WHERE guild_id=? AND owner_user_id=? AND deleted=0",
                (principal.guild_id, principal.user_id)).fetchone()["c"]
            if own >= self.settings.max_projects_per_user:
                raise ProjectQuota("This user has too many stored projects.")
            self._bind_task(conn, project_id, task_id)
            conn.execute(
                "INSERT INTO projects(id,guild_id,owner_user_id,channel_id,name,audience,"
                "origin_task_id,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (project_id, principal.guild_id, principal.user_id, principal.channel_id,
                 name, "private", task_id, ts, ts))
            self._event(conn, project_id, "create", principal.user_id)
        return self.check_access(principal, project_id)

    def describe(self, principal: Principal, project_id: str) -> dict:
        self._require_principal(principal)
        self._validate_project_id(project_id)
        with self._read() as conn:
            return self._summary(conn, self._project_row(conn, principal, project_id))

    def list_projects(self, principal: Principal) -> list[dict]:
        self._require_principal(principal)
        with self._read() as conn:
            rows = conn.execute(
                """SELECT p.* FROM projects p
                   WHERE p.deleted=0 AND p.guild_id=? AND p.channel_id=?
                     AND (p.owner_user_id=? OR (p.audience='shared' AND EXISTS
                          (SELECT 1 FROM grants g WHERE g.project_id=p.id AND g.user_id=?)))
                   ORDER BY p.updated_at DESC""",
                (principal.guild_id, principal.channel_id, principal.user_id,
                 principal.user_id)).fetchall()
            return [self._summary(conn, row) for row in rows]

    def _summary(self, conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
        latest = conn.execute(
            """SELECT v.* FROM versions v WHERE v.project_id=?
               ORDER BY v.version DESC LIMIT 1""", (row["id"],)).fetchone()
        return {
            "id": row["id"], "name": row["name"], "guild_id": row["guild_id"],
            "owner_user_id": row["owner_user_id"], "channel_id": row["channel_id"],
            "audience": row["audience"], "origin_task_id": row["origin_task_id"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "latest_version": None if latest is None else latest["version"],
            "state": None if latest is None else latest["state"],
            "file_count": 0 if latest is None else latest["file_count"],
            "total_bytes": 0 if latest is None else latest["total_bytes"],
        }

    def share(self, principal: Principal, project_id: str, *, user_ids: Sequence[int]) -> dict:
        """Owner-only explicit collaboration. No implicit club-wide audience."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        users = list(user_ids)
        if not users or len(users) > 20:
            raise ProjectViolation("Share with between 1 and 20 explicit users")
        for user_id in users:
            if not _valid_id(user_id) or user_id == principal.user_id:
                raise ProjectViolation("Invalid grantee")
        with self._write() as conn:
            row = self._project_row(conn, principal, project_id)
            if row["owner_user_id"] != principal.user_id:
                raise ProjectDenied("Only the project owner can share it.")
            ts = self._clock()
            conn.execute("UPDATE projects SET audience='shared', updated_at=? WHERE id=?",
                         (ts, project_id))
            for user_id in users:
                conn.execute("INSERT OR IGNORE INTO grants(project_id,user_id,granted_by,created_at) "
                             "VALUES (?,?,?,?)", (project_id, user_id, principal.user_id, ts))
            self._event(conn, project_id, "grant", principal.user_id,
                        ",".join(str(u) for u in users)[:240])
        return self.check_access(principal, project_id)

    def revoke(self, principal: Principal, project_id: str, *, user_id: int) -> dict:
        self._require_principal(principal)
        self._validate_project_id(project_id)
        if not _valid_id(user_id):
            raise ProjectViolation("Invalid grantee")
        with self._write() as conn:
            row = self._project_row(conn, principal, project_id)
            if row["owner_user_id"] != principal.user_id:
                raise ProjectDenied("Only the project owner can revoke access.")
            conn.execute("DELETE FROM grants WHERE project_id=? AND user_id=?",
                         (project_id, user_id))
            self._event(conn, project_id, "revoke", principal.user_id, str(user_id))
        return self.check_access(principal, project_id)

    def relocate(self, principal: Principal, project_id: str, *, channel_id: int) -> dict:
        """Owner re-authorizes the project's single audience channel (e.g. a new
        continuation thread). Access in the old channel stops immediately."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        if not _valid_id(channel_id):
            raise ProjectViolation("channel_id must be a Discord channel ID")
        with self._write() as conn:
            row = self._project_row(conn, principal, project_id)
            if row["owner_user_id"] != principal.user_id:
                raise ProjectDenied("Only the project owner can move it.")
            ts = self._clock()
            conn.execute("UPDATE projects SET channel_id=?, updated_at=? WHERE id=?",
                         (channel_id, ts, project_id))
            self._event(conn, project_id, "relocate", principal.user_id, str(channel_id))
            # check_access(principal) would now deny: the caller's own channel
            # is the OLD audience. Re-read the moved row (owner verified above).
            moved = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            return self._summary(conn, moved)

    def delete_project(self, principal: Principal, project_id: str) -> dict:
        """Owner purge. Manifest rows and blobs go now; the row tombstone keeps
        the append-only event history addressable."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        with self._write() as conn:
            row = self._project_row(conn, principal, project_id)
            if row["owner_user_id"] != principal.user_id:
                raise ProjectDenied("Only the project owner can delete it.")
            hashes = [r["sha256"] for r in conn.execute(
                "SELECT sha256 FROM files WHERE project_id=?", (project_id,))]
            conn.execute("DELETE FROM files WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM versions WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM grants WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM task_projects WHERE project_id=?", (project_id,))
            conn.execute("UPDATE projects SET deleted=1, updated_at=? WHERE id=?",
                         (self._clock(), project_id))
            self._event(conn, project_id, "delete", principal.user_id)
        self._gc_blobs(hashes)
        return {"deleted": True, "project_id": project_id}

    # ------------------------------------------------------------------- saves

    @staticmethod
    def _pairs(files: Mapping[str, bytes] | Iterable[tuple[str, bytes]]
               ) -> list[tuple[str, bytes]]:
        if isinstance(files, Mapping):
            return list(files.items())
        if isinstance(files, Iterable) and not isinstance(files, (str, bytes)):
            pairs = []
            for item in files:
                if (isinstance(item, tuple) and len(item) == 2):
                    pairs.append(item)
                else:
                    raise ProjectViolation("files must be a mapping or (name, bytes) pairs")
            return pairs
        raise ProjectViolation("files must be a mapping of filename to bytes")

    def _validate_file(self, name: object, data: object, seen: dict[str, str]
                       ) -> tuple[str, str, bytes, str]:
        canonical, key = _validate_filename(name, self.settings)
        if key in seen:
            raise ProjectViolation("Duplicate or case-colliding filename")
        if isinstance(data, (bytes, bytearray, memoryview)):
            blob = bytes(data)
        else:
            raise ProjectViolation("file content must be bytes")
        if len(blob) > self.settings.max_file_bytes:
            raise ProjectQuota(f"File exceeds {self.settings.max_file_bytes} bytes")
        suffix = Path(canonical).suffix.lower()
        if suffix in _DENIED_SUFFIXES:
            raise ProjectViolation("Archive and compressed inputs are not accepted")
        if any(blob.startswith(magic) for magic in _MAGICS):
            raise ProjectViolation("Archive and compressed payloads are rejected")
        if len(blob) > 257 and blob[257:262] == b"ustar":  # tar bomb path
            raise ProjectViolation("Archive and compressed payloads are rejected")
        digest = hashlib.sha256(blob).hexdigest()
        seen[key] = canonical
        return canonical, key, blob, digest

    @staticmethod
    def _check_prefix_conflicts(names: list[str]) -> None:
        for outer in names:
            prefix = outer + "/"
            for inner in names:
                if inner != outer and inner.startswith(prefix):
                    raise ProjectViolation("A file path is also used as a directory")

    def save(self, principal: Principal, project_id: str, *, task_id: str,
             files: Mapping[str, bytes] | Iterable[tuple[str, bytes]],
             provenance: str, verified: bool = True,
             dependency_instructions: str | None = None,
             best_effort: bool = False) -> dict:
        """Persist one full-snapshot version of a project.

        ``verified=False`` records an unverified/partial checkpoint (e.g. files
        salvaged after a timeout or cancellation). With ``best_effort=True``
        individually invalid entries are skipped and reported under
        ``rejected`` instead of failing the save, so already-collected valid
        files survive teardown; nothing claims that an abrupt host loss is
        recoverable.
        """
        self._require_principal(principal)
        self._validate_project_id(project_id)
        task_id = self._validate_task_id(task_id, self.settings.max_task_id_chars)
        if not isinstance(verified, bool):
            raise ProjectViolation("verified must be a bool")
        provenance = _clean_text(provenance, self.settings.max_provenance_chars,
                                 "provenance", required=True)
        deps = "" if dependency_instructions is None else _clean_text(
            dependency_instructions, self.settings.max_dependency_chars,
            "dependency_instructions")
        settings = self.settings

        accepted: list[tuple[str, bytes, str]] = []
        rejected: list[dict] = []
        seen: dict[str, str] = {}
        total = 0
        for name, data in self._pairs(files):
            try:
                canonical, _, blob, digest = self._validate_file(name, data, seen)
            except ProjectError as exc:
                if not best_effort:
                    raise
                label = name if type(name) is str and 0 < len(name) <= settings.max_name_chars else "<invalid>"
                rejected.append({"name": label, "reason": str(exc)})
                continue
            if len(accepted) >= settings.max_files_per_version:
                if not best_effort:
                    raise ProjectQuota(f"At most {settings.max_files_per_version} files per version")
                rejected.append({"name": canonical, "reason": "file count limit"})
                continue
            if total + len(blob) > settings.max_version_bytes:
                if not best_effort:
                    raise ProjectQuota(f"A version holds at most {settings.max_version_bytes} bytes")
                rejected.append({"name": canonical, "reason": "version byte limit"})
                continue
            total += len(blob)
            accepted.append((canonical, blob, digest))
        if not accepted and not best_effort:
            raise ProjectViolation("A version must contain at least one file")
        self._check_prefix_conflicts([name for name, _, _ in accepted])
        if not accepted:
            raise ProjectViolation("No valid file remained for a partial version")

        evicted: list[str] = []
        ts = self._clock()
        with self._write() as conn:
            row = self._project_row(conn, principal, project_id)
            self._bind_task(conn, project_id, task_id)
            counts = conn.execute(
                "SELECT COUNT(*) c, COALESCE(MAX(version),0) m FROM versions WHERE project_id=?",
                (project_id,)).fetchone()
            version = counts["m"] + 1
            evict_count = counts["c"] + 1 - settings.max_versions_per_project
            if evict_count > 0:
                victims = [r["version"] for r in conn.execute(
                    "SELECT version FROM versions WHERE project_id=? ORDER BY version ASC LIMIT ?",
                    (project_id, evict_count))]
                marks = ",".join("?" * len(victims))
                evicted = [r["sha256"] for r in conn.execute(
                    f"SELECT DISTINCT sha256 FROM files WHERE project_id=? AND version IN ({marks})",
                    (project_id, *victims))]
                conn.execute(
                    f"DELETE FROM files WHERE project_id=? AND version IN ({marks})",
                    (project_id, *victims))
                conn.execute(
                    f"DELETE FROM versions WHERE project_id=? AND version IN ({marks})",
                    (project_id, *victims))
            # Quota is measured after version eviction so an evicting save is
            # charged only for the bytes it actually retains.
            held = conn.execute(
                "SELECT COALESCE(SUM(total_bytes),0) b FROM versions WHERE project_id=?",
                (project_id,)).fetchone()["b"]
            if held + total > settings.max_project_bytes:
                raise ProjectQuota("This project has reached its storage quota.")
            owned = conn.execute(
                """SELECT COALESCE(SUM(v.total_bytes),0) b FROM versions v
                   JOIN projects p ON p.id=v.project_id
                   WHERE p.guild_id=? AND p.owner_user_id=? AND p.deleted=0""",
                (principal.guild_id, row["owner_user_id"])).fetchone()["b"]
            if owned + total > settings.max_user_bytes:
                raise ProjectQuota("This user has reached the total project storage quota.")
            for _, blob, digest in accepted:
                self._write_blob(digest, blob)
            conn.execute(
                "INSERT INTO versions(project_id,version,task_id,state,provenance,"
                "dependency_instructions,file_count,total_bytes,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (project_id, version, task_id, "verified" if verified else "partial",
                 provenance, deps, len(accepted), total, ts))
            for name, blob, digest in accepted:
                conn.execute(
                    "INSERT INTO files(project_id,version,name,size,sha256) VALUES (?,?,?,?,?)",
                    (project_id, version, name, len(blob), digest))
            conn.execute("UPDATE projects SET updated_at=? WHERE id=?", (ts, project_id))
            self._event(conn, project_id, "save", principal.user_id,
                        f"v{version} {'verified' if verified else 'partial'} {len(accepted)}f")
        self._gc_blobs(evicted)
        result = {"project_id": project_id, "version": version,
                  "state": "verified" if verified else "partial",
                  "file_count": len(accepted), "total_bytes": total, "created_at": ts}
        if rejected:
            result["rejected"] = rejected
        return result

    # ------------------------------------------------------------------ reads

    def list_files(self, principal: Principal, project_id: str, *,
                   version: int | None = None) -> dict:
        self._require_principal(principal)
        self._validate_project_id(project_id)
        with self._read() as conn:
            row = self._project_row(conn, principal, project_id)
            header = self._version_row(conn, project_id, version)
            files = [{"name": f["name"], "size": f["size"], "sha256": f["sha256"]}
                     for f in conn.execute(
                         "SELECT name,size,sha256 FROM files WHERE project_id=? AND version=? "
                         "ORDER BY name", (project_id, header["version"]))]
            return {"project_id": row["id"], "version": header["version"],
                    "state": header["state"], "provenance": header["provenance"],
                    "dependency_instructions": header["dependency_instructions"],
                    "files": files}

    def read_file(self, principal: Principal, project_id: str, name: str, *,
                  version: int | None = None) -> bytes:
        """Exact bytes for one manifest entry, integrity-checked."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        canonical, _ = _validate_filename(name, self.settings)
        with self._read() as conn:
            self._project_row(conn, principal, project_id)
            header = self._version_row(conn, project_id, version)
            entry = conn.execute(
                "SELECT size,sha256 FROM files WHERE project_id=? AND version=? AND name=?",
                (project_id, header["version"], canonical)).fetchone()
            if entry is None:
                raise ProjectViolation("No such file in this project version")
            return self._read_blob(entry["sha256"], entry["size"])

    def verify(self, principal: Principal, project_id: str, *,
               version: int | None = None) -> dict:
        """Re-hash every stored blob. Raises ProjectIntegrityError on the first
        mismatch; never returns tampered bytes."""
        listing = self.list_files(principal, project_id, version=version)
        for entry in listing["files"]:
            self._read_blob(entry["sha256"], entry["size"])
        return {"project_id": project_id, "version": listing["version"],
                "files": len(listing["files"]), "ok": True}

    def restore(self, principal: Principal, project_id: str, *, task_id: str,
                version: int | None = None) -> dict:
        """Full bytes for a continuation. Performs the final authority check
        itself; the caller still repeats ``check_access()`` after any gap
        (worker boot, approval wait) before staging."""
        self._require_principal(principal)
        self._validate_project_id(project_id)
        task_id = self._validate_task_id(task_id, self.settings.max_task_id_chars)
        new_binding = False
        with self._read() as conn:
            row = self._project_row(conn, principal, project_id)
            existing = conn.execute("SELECT project_id FROM task_projects WHERE task_id=?",
                                    (task_id,)).fetchone()
            if existing is not None and existing["project_id"] != project_id:
                raise ProjectDenied("This task is bound to a different project.")
            header = self._version_row(conn, project_id, version)
            entries = conn.execute(
                "SELECT name,size,sha256 FROM files WHERE project_id=? AND version=? ORDER BY name",
                (project_id, header["version"])).fetchall()
            files = []
            for entry in entries:
                data = self._read_blob(entry["sha256"], entry["size"])
                files.append((entry["name"], data))
            new_binding = existing is None
        # A brand-new continuation task locks onto exactly this project from
        # now on. Separate transaction: the read lock is not reentrant.
        if new_binding:
            with self._write() as conn:
                self._bind_task(conn, project_id, task_id)
                self._event(conn, project_id, "restore", principal.user_id,
                            f"v{header['version']} by task")
        return {"project_id": project_id, "name": row["name"],
                "version": header["version"], "state": header["state"],
                "provenance": header["provenance"],
                "dependency_instructions": header["dependency_instructions"],
                "files": files}

    def worker_payload(self, principal: Principal, project_id: str, *, task_id: str,
                       version: int | None = None) -> dict:
        """JSON-safe shape for the runner: files as {name, data_base64}, the
        same convention as ``jobs.input_files``. Bytes are never interpreted
        here; the worker stages them into its disposable workspace."""
        restored = self.restore(principal, project_id, task_id=task_id, version=version)
        return {"project_id": restored["project_id"], "name": restored["name"],
                "version": restored["version"], "state": restored["state"],
                "provenance": restored["provenance"],
                "dependency_instructions": restored["dependency_instructions"],
                "files": [{"name": name, "data_base64": base64.b64encode(data).decode("ascii"),
                           "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in restored["files"]]}

    def _version_row(self, conn: sqlite3.Connection, project_id: str,
                     version: int | None) -> sqlite3.Row:
        if version is None:
            row = conn.execute(
                "SELECT * FROM versions WHERE project_id=? ORDER BY version DESC LIMIT 1",
                (project_id,)).fetchone()
        else:
            if type(version) is not int or version <= 0:
                raise ProjectViolation("version must be a positive integer")
            row = conn.execute(
                "SELECT * FROM versions WHERE project_id=? AND version=?",
                (project_id, version)).fetchone()
        if row is None:
            raise ProjectViolation("This project has no stored version yet")
        return row

    # ------------------------------------------------------------------ blobs

    def _write_blob(self, digest: str, blob: bytes) -> None:
        target = self._blob_path(digest)
        directory = target.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        try:
            existing = os.stat(target, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode) or existing.st_size != len(blob):
                raise ProjectIntegrityError("Blob store is inconsistent")
            return
        tmp = directory / f"{digest}.tmp.{os.urandom(6).hex()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _read_blob(self, digest: str, size: int) -> bytes:
        path = self._blob_path(digest)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError as exc:
            raise ProjectIntegrityError("A stored file is missing or not a regular file") from exc
        with os.fdopen(fd, "rb") as handle:
            meta = os.fstat(handle.fileno())
            if not stat.S_ISREG(meta.st_mode):
                raise ProjectIntegrityError("A stored file is not a regular file")
            if meta.st_size != size:
                raise ProjectIntegrityError("A stored file no longer matches its manifest size")
            data = handle.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ProjectIntegrityError("A stored file failed its hash check")
        return data

    def _gc_blobs(self, digests: Iterable[str]) -> int:
        """Delete candidate blobs no longer referenced by any manifest row.
        Never follows links; anything unexpected is left alone."""
        candidates = {d for d in digests if _HEX64.match(d)}
        if not candidates:
            return 0
        with self._read() as conn:
            referenced = {r["sha256"] for r in conn.execute("SELECT DISTINCT sha256 FROM files")}
        removed = 0
        for digest in candidates - referenced:
            path = self._blob_path(digest)
            try:
                meta = os.lstat(path)
                if stat.S_ISREG(meta.st_mode):
                    os.unlink(path)
                    removed += 1
            except OSError:
                continue
        return removed

    # --------------------------------------------------------------- retention

    def retention_sweep(self, *, now: int | None = None) -> dict:
        """Operator/gateway maintenance (no Principal: retention is policy, not
        a user capability). Deletes versions past ``retention_days``, always
        keeping the latest version unless the whole project has aged out;
        garbage-collects orphan blobs and stale temp files."""
        ts = self._clock() if now is None else now
        cutoff = ts - self.settings.retention_days * 86400
        orphan: list[str] = []
        versions_removed = projects_removed = 0
        with self._write() as conn:
            latest = {r["project_id"]: r["version"] for r in conn.execute(
                "SELECT project_id, MAX(version) version FROM versions GROUP BY project_id")}
            stale = conn.execute(
                "SELECT project_id, version, created_at FROM versions WHERE created_at<? "
                "ORDER BY project_id, version", (cutoff,)).fetchall()
            aged_projects = {r["id"] for r in conn.execute(
                "SELECT id FROM projects WHERE created_at<? AND updated_at<?", (cutoff, cutoff))}
            for row in stale:
                if latest.get(row["project_id"]) == row["version"] and row["project_id"] not in aged_projects:
                    continue
                orphan.extend(r["sha256"] for r in conn.execute(
                    "SELECT sha256 FROM files WHERE project_id=? AND version=?",
                    (row["project_id"], row["version"])))
                conn.execute("DELETE FROM files WHERE project_id=? AND version=?",
                             (row["project_id"], row["version"]))
                conn.execute("DELETE FROM versions WHERE project_id=? AND version=?",
                             (row["project_id"], row["version"]))
                versions_removed += 1
            for project_id in aged_projects:
                still = conn.execute(
                    "SELECT COUNT(*) c FROM versions WHERE project_id=?",
                    (project_id,)).fetchone()["c"]
                if still:
                    continue  # its newest version was kept; project stays live
                conn.execute("UPDATE projects SET deleted=1, updated_at=? WHERE id=? AND deleted=0",
                             (ts, project_id))
                conn.execute("DELETE FROM grants WHERE project_id=?", (project_id,))
                conn.execute("DELETE FROM task_projects WHERE project_id=?", (project_id,))
                projects_removed += 1
            conn.execute("DELETE FROM events WHERE ts<?", (cutoff,))
        blobs_removed = self._gc_blobs(orphan)
        # Full orphan scan: blobs left by a rolled-back/interrupted save (bytes
        # written, manifest rows never committed) are not in any candidate
        # list. Reclaim unreferenced regular files past a grace period; a
        # concurrent in-flight save's fresh blob must survive the race.
        referenced = set()
        with self._read() as conn:
            referenced = {r["sha256"] for r in conn.execute(
                "SELECT DISTINCT sha256 FROM files")}
        orphan_blobs = 0
        stale_tmp = 0
        try:
            for sub in self.blob_dir.iterdir():
                if not sub.is_dir():
                    continue
                for entry in sub.iterdir():
                    try:
                        meta = os.lstat(entry)
                        if not stat.S_ISREG(meta.st_mode) or ts - meta.st_mtime <= 3600:
                            continue
                        if ".tmp." in entry.name:
                            os.unlink(entry)
                            stale_tmp += 1
                        elif entry.name not in referenced and _HEX64.match(entry.name):
                            os.unlink(entry)
                            orphan_blobs += 1
                    except OSError:
                        continue
        except OSError:
            pass
        return {"versions_removed": versions_removed,
                "projects_removed": projects_removed,
                "blobs_removed": blobs_removed,
                "orphan_blobs_removed": orphan_blobs,
                "temp_files_removed": stale_tmp}
