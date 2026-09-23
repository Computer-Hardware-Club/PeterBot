"""Operator CLI for private diagnostics, retention, and snapshot checks (PETER-16).

Subcommands:

    diagnose STATE_DIR                 read-only aggregate health report (JSON)
    retention STATE_DIR [--apply ...]  retention plan (dry-run by default)
    check-snapshot SNAPSHOT STAGING    verify snapshot, restore to staging, diagnose

Nothing here touches the network, the model, or Discord. `diagnose` and the
dry-run plan only read; `--apply` first writes a verified snapshot (including
the project-blob cross-check) and aborts before any deletion if that check
fails. Output carries aggregate counts and fixed reason tokens only: never
prompts, answers, memory text, Discord IDs, job IDs, tokens, or paths.

Requires the repository root on sys.path (the container runs from /app; the
bootstrap below also allows direct `python deploy/housekeeping.py`).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.state_backup import backup, restore, verify  # noqa: E402
from peterbot.operator_ops import (  # noqa: E402
    RetentionConfig, diagnose, retention_apply, retention_plan,
)


def _emit(report: object) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def _config(args: argparse.Namespace) -> RetentionConfig:
    return RetentionConfig(
        conversations_days=args.conversations_days,
        metrics_days=args.metrics_days,
        terminal_jobs_days=args.terminal_jobs_days,
        settled_receipts_days=args.settled_receipts_days,
        include_projects=args.include_projects,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="read-only aggregate health report")
    diag.add_argument("state_dir", type=Path,
                      default=Path(os.environ.get("PETERBOT_STATE_DIR", "peterbot-data")),
                      nargs="?")

    ret = sub.add_parser("retention", help="retention plan (dry-run) or explicit apply")
    ret.add_argument("state_dir", type=Path,
                     default=Path(os.environ.get("PETERBOT_STATE_DIR", "peterbot-data")),
                     nargs="?")
    ret.add_argument("--apply", action="store_true",
                     help="delete eligible rows; requires --backup-destination")
    ret.add_argument("--backup-destination", type=Path,
                     help="snapshot written and verified before any deletion")
    ret.add_argument("--conversations-days", type=int, default=90)
    ret.add_argument("--metrics-days", type=int, default=30)
    ret.add_argument("--terminal-jobs-days", type=int, default=90)
    ret.add_argument("--settled-receipts-days", type=int, default=180)
    ret.add_argument("--include-projects", action="store_true",
                     help="also run ProjectStore.retention_sweep under its own policy")

    snap = sub.add_parser("check-snapshot",
                          help="verify a snapshot, restore to staging, diagnose the copy")
    snap.add_argument("snapshot", type=Path)
    snap.add_argument("staging", type=Path)

    args = parser.parse_args(argv)
    if args.command == "diagnose":
        _emit(diagnose(args.state_dir))
    elif args.command == "retention":
        config = _config(args)
        if not args.apply:
            _emit(retention_plan(args.state_dir, config))
            return 0
        if args.backup_destination is None:
            parser.error("--apply requires --backup-destination")
        backup(args.state_dir, args.backup_destination)
        verify(args.backup_destination)  # redundant with restore; explicit gate
        _emit(retention_apply(args.state_dir, config))
    else:
        verify(args.snapshot)
        restore(args.snapshot, args.staging)
        _emit(diagnose(args.staging))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
