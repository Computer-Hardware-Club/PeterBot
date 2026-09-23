"""Consistent, private snapshots of Peter's local state directory.

Run ``backup`` while Peter is live; SQLite files use the backup API. Run
``restore`` only into a new, empty directory with the gateway stopped.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import tempfile


DATABASE_SUFFIXES = (".sqlite3", ".db")
TRANSIENT_SUFFIXES = ("-wal", "-shm", "-journal")


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _safe_name(name: str) -> Path:
    posix = PurePosixPath(name)
    if not name or posix.is_absolute() or posix.as_posix() != name or any(part == ".." for part in posix.parts) or "\\" in name:
        raise ValueError("Unsafe backup path")
    return Path(*posix.parts)


def _private_file(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def _sqlite_backup(source: Path, target: Path) -> None:
    _private_file(target)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(target)) as snapshot:
            original.backup(snapshot)
            if snapshot.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"SQLite integrity check failed: {source.name}")


def _source_files(source: Path):
    for path in sorted(source.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"State contains a link or special file: {path.relative_to(source)}")
        if path.name.endswith(TRANSIENT_SUFFIXES):
            continue
        yield path


def backup(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if not source.is_dir() or os.path.lexists(destination) or source == destination or source in destination.parents:
        raise ValueError("Source must be a directory and destination must not exist inside it")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    temporary.chmod(0o700)
    try:
        entries = []
        for path in _source_files(source):
            name = path.relative_to(source).as_posix()
            target = temporary / "files" / _safe_name(name)
            kind = "sqlite" if path.name.endswith(DATABASE_SUFFIXES) else "file"
            if kind == "sqlite":
                _sqlite_backup(path, target)
            else:
                _private_file(target)
                with path.open("rb") as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file, 1024 * 1024)
            size, digest = _digest(target)
            entries.append({"path": name, "kind": kind, "bytes": size, "sha256": digest})
        manifest = temporary / "manifest.json"
        _private_file(manifest)
        manifest.write_text(json.dumps({"version": 1, "files": entries}, indent=2) + "\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def verify(snapshot: Path) -> list[dict]:
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ValueError("Invalid snapshot directory")
    manifest_path = snapshot / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("Invalid manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported snapshot manifest")
    entries = manifest["files"]
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "kind", "bytes", "sha256"}:
            raise ValueError("Invalid snapshot entry")
        name = entry["path"]
        relative = _safe_name(name)
        if name in seen or entry["kind"] not in {"sqlite", "file"}:
            raise ValueError("Duplicate or invalid snapshot entry")
        seen.add(name)
        path = snapshot / "files" / relative
        component = path
        while component != snapshot:
            if component.is_symlink():
                raise ValueError("Snapshot contains a symlink")
            component = component.parent
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Snapshot contains a link or special file")
        if (entry["bytes"], entry["sha256"]) != _digest(path):
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        if entry["kind"] == "sqlite":
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError(f"Snapshot database damaged: {name}")
    actual = set()
    for path in (snapshot / "files").rglob("*"):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("Snapshot contains a link or special file")
        actual.add(path.relative_to(snapshot / "files").as_posix())
    if actual != seen:
        raise ValueError("Snapshot has unexpected or missing files")
    return entries


def restore(snapshot: Path, destination: Path) -> None:
    entries = verify(snapshot)
    if os.path.lexists(destination):
        raise ValueError("Restore destination must not exist")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    temporary.chmod(0o700)
    try:
        for entry in entries:
            target = temporary / _safe_name(entry["path"])
            _private_file(target)
            with (snapshot / "files" / entry["path"]).open("rb") as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, 1024 * 1024)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("backup", "verify", "restore"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    args = parser.parse_args()
    if args.operation == "backup" and args.destination:
        backup(args.source, args.destination)
    elif args.operation == "verify" and not args.destination:
        verify(args.source)
    elif args.operation == "restore" and args.destination:
        restore(args.source, args.destination)
    else:
        parser.error("backup and restore need a destination; verify does not")
    print(f"{args.operation} completed")


if __name__ == "__main__":
    main()
