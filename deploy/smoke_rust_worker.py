#!/usr/bin/env python3
"""Compile and run the e-digits Rust fixture inside the ACTUAL restricted worker.

Run on the trusted worker host/VM (the machine holding the Docker socket), from
the repository checkout, with `aiohttp` importable (same environment as the
runner, e.g. inside the runner image with the repo mounted). It creates one
disposable worker with the production `sandbox_runner.worker_args` (same UID,
profiles, caps, tmpfs, internal network), then proves inside that container:

  1. pinned rustc/cargo compile the dependency-free fixture offline,
  2. output matches the independent stdlib-decimal reference at 12/200/2000 digits,
  3. unreasonable N and malformed arguments exit 2,
  4. deploy/check_hermes_isolation.py passes. The broker 401 probe runs against
     --gateway-host (VM: the broker alias) when given; otherwise it is a SKIP —
     reported honestly, exit-nonzero unless --expect-no-gateway declares it,
  5. runner control addresses (--runner-probe HOST:PORT, e.g. the runner's
     container IP on ctl0 and the published 192.168.241.2:8780) are unreachable
     from the worker,
  6. container resource limits match the selected operator profile.

Usage: PETERBOT_SMOKE_TOKEN=<32+ chars> python3 deploy/smoke_rust_worker.py \
           --image peterbot-hermes-worker:REV [--profile build] \
           [--gateway-host 192.168.240.2] [--runner-probe HOST:PORT ...] \
           [--expect-no-gateway]
Exit 0 only when every check passes or is a declared SKIP. Cleans up always.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peterbot import sandbox_runner as sr  # noqa: E402
from deploy.check_hermes_isolation import BLOCKED_ENDPOINTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/edigits"
DIGITS = (12, 200, 2000)
BUILD_TIMEOUT = 600
RUN_TIMEOUT = 240
REQUIRED_ISOLATION = frozenset({
    "worker_uid", "container_execution", "no_credential_environment",
    "no_host_credentials_or_docker_socket", "root_write_denied",
    "toolchain_write_denied", "workspace_write_allowed",
    "no_effective_capabilities", "no_new_privileges",
    "gateway_requires_capability",
}) | frozenset(name + "_blocked" for name, *_ in BLOCKED_ENDPOINTS)


class SmokeError(RuntimeError):
    pass


def isolation_report_complete(report: object) -> bool:
    """Never let an empty, truncated, or malformed isolation run count as smoke."""
    if not isinstance(report, dict) or not isinstance(report.get("passed"), bool):
        return False
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return False
    if any(not isinstance(item, dict) or not isinstance(item.get("check"), str)
           or type(item.get("passed")) is not bool for item in checks):
        return False
    names = [item["check"] for item in checks]
    return len(names) == len(set(names)) and REQUIRED_ISOLATION <= set(names)


async def docker(*args: str, stdin: bytes | None = None, timeout: int = 60) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout)
        return process.returncode, out, err
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise SmokeError(f"docker {args[0]} timed out")


def fixture_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(FIXTURE, arcname="edigits", recursive=True)
    return buffer.getvalue()


async def container_exec(name: str, *argv: str, timeout: int = RUN_TIMEOUT) -> tuple[int, str]:
    code, out, err = await docker("exec", "--user", "10000:10000", name, *argv, timeout=timeout)
    return code, (out + err).decode(errors="replace")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--profile", default="build", choices=sorted(sr.RESOURCE_PROFILES))
    parser.add_argument("--network", default="peterbot_workers")
    parser.add_argument("--gateway-host", default="",
                        help="broker address reachable FROM the worker (VM: the .240.2 alias)")
    parser.add_argument("--expect-no-gateway", action="store_true",
                        help="declare that no live broker is bound; exempts the 401 probe as SKIP")
    parser.add_argument("--runner-probe", action="append", default=[], metavar="HOST:PORT",
                        help="runner control address that MUST be unreachable from the worker")
    args = parser.parse_args()
    for spec in args.runner_probe:
        host, _, port = spec.rpartition(":")
        if not host or not port.isdigit():
            parser.error(f"--runner-probe wants HOST:PORT, got {spec!r}")
    token = os.environ.get("PETERBOT_SMOKE_TOKEN", "")
    settings = sr.Settings(token or ("smoke-" + "x" * 30), args.image,
                           args.network, resource_profile=args.profile)
    name = f"peterbot-smoke-{uuid.uuid4().hex[:16]}"
    results: list[dict] = []

    def record(check: str, passed, **detail):
        status = "PASS" if passed else "FAIL"
        results.append({"check": check, "status": status, "passed": bool(passed), **detail})
        print(f"{status} {check} {json.dumps(detail, default=str)[:400]}")

    def skip(check: str, declared: bool, **detail):
        # SKIP never counts as a pass; finish() tolerates it only when declared.
        results.append({"check": check, "status": "SKIP", "passed": None,
                        "declared": declared, **detail})
        print(f"SKIP {check} {json.dumps(detail, default=str)[:200]}")

    created = False
    try:
        code, out, err = await docker(*sr.worker_args(settings, name))
        if code != 0:
            raise SmokeError("worker creation failed: " + err.decode(errors="replace")[-200:])
        created = True

        # Inspection: profile ceilings are the production ones, read-only, no caps.
        code, out, _ = await docker("inspect", name, "--format",
                                    '{{.HostConfig.Memory}}|{{.HostConfig.NanoCpus}}|{{.HostConfig.PidsLimit}}|{{.HostConfig.CapDrop}}|{{.HostConfig.ReadonlyRootfs}}|{{.HostConfig.NetworkMode}}')
        fields = out.decode().strip().split("|")
        assert settings.profile.memory.endswith("g")  # profiles are declared in GiB
        expected_memory = int(float(settings.profile.memory[:-1]) * 2 ** 30)
        record("profile_limits_match", fields[0] == str(expected_memory)
               and int(fields[2]) == settings.profile.pids and fields[4] == "true"
               and args.network in fields[5], inspect=fields, profile=settings.profile.name)

        # Toolchain pin and offline compile inside the worker.
        code, text = await container_exec(name, "rustc", "--version")
        record("pinned_rustc", code == 0 and "rustc 1.98.1" in text, observed=text.strip())
        code, text = await container_exec(name, "cargo", "--version")
        record("pinned_cargo", code == 0 and "cargo 1.98.1" in text, observed=text.strip())

        # docker cp is refused against a read-only rootfs (even onto tmpfs
        # mounts); stream the fixture through exec stdin, same as the runner.
        code, _, err = await docker("exec", "-i", "--user", "10000:10000", name,
                                    "tar", "-x", "-C", "/workspace", stdin=fixture_tar())
        if code != 0:
            raise SmokeError("fixture copy failed: " + err.decode()[-200:])
        code, text = await container_exec(name, "cargo", "build", "--release", "--offline",
                                          "--manifest-path", "/workspace/edigits/Cargo.toml",
                                          "--target-dir", "/workspace/target", timeout=BUILD_TIMEOUT)
        record("offline_build", code == 0, tail=text[-400:])
        if code != 0:
            return finish(results)

        # Correctness against the independent in-container decimal reference.
        for digits in DIGITS:
            code, rust = await container_exec(name, "/workspace/target/release/edigits", str(digits))
            ref_code, reference = await container_exec(name, "python3", "-I",
                                                       "/workspace/edigits/reference_e.py", str(digits))
            match = (code == 0 and ref_code == 0
                     and rust.strip().splitlines()[-1] == reference.strip())
            record(f"edigits_{digits}_matches_reference", match,
                   rust_head=rust.strip()[:40], reference_head=reference.strip()[:40])

        # Unreasonable input contract inside the worker.
        for bad in ("0", "10001", "-3", "x"):
            code, text = await container_exec(name, "/workspace/target/release/edigits", bad)
            record(f"rejects_{bad!r}", code == 2, output=text.strip()[:120])

        # Isolation probes inside this exact container (stream in; no docker cp).
        # Point the gateway probe at a live broker when one is reachable.
        iso_env = []
        if args.gateway_host:
            iso_env = ["-e", f"PETERBOT_ISOLATION_GATEWAY_HOST={args.gateway_host}"]
        if args.runner_probe:
            # Probe the REAL runner address, not the single-host default.
            iso_env += ["-e", "PETERBOT_ISOLATION_RUNNER_HOST="
                      + args.runner_probe[0].rpartition(":")[0]]
        iso = (ROOT / "deploy/check_hermes_isolation.py").read_bytes()
        await docker("exec", "-i", "--user", "10000:10000", name,
                     "sh", "-c", "cat > /tmp/check_isolation.py", stdin=iso)
        code, out, _ = await docker("exec", *(["--user", "10000:10000"] + iso_env), name,
                                    "python3", "-I", "/tmp/check_isolation.py")
        try:
            report = json.loads(out)
        except ValueError:
            report = None
        complete = isolation_report_complete(report)
        record("isolation_report_complete", complete and (code == 0 if args.gateway_host else code in (0, 1)),
               exit_code=code)
        if complete:
            for check in report["checks"]:
                if check["check"] == "gateway_requires_capability" and not args.gateway_host:
                    skip("isolation:gateway_requires_capability", args.expect_no_gateway,
                         reason="no --gateway-host given; pass one or declare --expect-no-gateway")
                    continue
                record("isolation:" + check["check"], check["passed"],
                       **{k: v for k, v in check.items() if k not in ("check", "passed")})

        # Network wall: egress attempts must fail from the worker.
        code, text = await container_exec(name, "python3", "-I", "-c",
            "import socket,sys\n"
            "for host,port in [('1.1.1.1',443),('8.8.8.8',53)]:\n"
            "    try:\n        socket.create_connection((host,port),timeout=2); sys.exit('EGRESS to '+host)\n"
            "    except OSError: pass\nprint('no-egress')")
        record("network_egress_denied", code == 0 and "no-egress" in text, tail=text[-200:])

        # Runner control must be unreachable from the worker (separate bridge +
        # the FORWARD wall). Addresses come from the operator so this works on
        # the VM (runner container IP on ctl0 AND the published address) and on
        # the single-host layout (runner service address).
        for spec in args.runner_probe:
            host, _, port = spec.rpartition(":")
            code, text = await container_exec(name, "python3", "-I", "-c",
                f"import socket,sys\n"
                f"try:\n    socket.create_connection(('{host}',{port}),timeout=2)\n"
                f"    sys.exit('REACHABLE')\nexcept OSError as e:\n    print('blocked', type(e).__name__)")
            record(f"runner_unreachable_{spec}", code == 0 and "blocked" in text, tail=text[-160:])

        # Pinned Hermes runtime fixture inside this exact container: the image
        # ships the pinned hermes-agent + /app/peterbot, so run the unittest.
        fixture = (ROOT / "tests/test_hermes_runtime_integration.py").read_bytes()
        await docker("exec", "-i", "--user", "10000:10000", name,
                     "sh", "-c", "cat > /tmp/hermes_fixture.py", stdin=fixture)
        code, out, err = await docker("exec", "--user", "10000:10000", name,
                                      "python3", "-I", "/tmp/hermes_fixture.py", "-v",
                                      timeout=300)
        record("hermes_runtime_fixture", code == 0, tail=(out + err)[-400:])
    finally:
        if created:
            await docker("rm", "--force", name, timeout=30)
    return finish(results)


def finish(results: list[dict]) -> int:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for item in results:
        counts[item["status"]] += 1
    undeclared = [i["check"] for i in results
                  if i["status"] == "SKIP" and not i.get("declared")]
    print(json.dumps({"total": len(results), "pass": counts["PASS"], "fail": counts["FAIL"],
                      "skip": counts["SKIP"],
                      "failures": [i["check"] for i in results if i["status"] == "FAIL"],
                      "undeclared_skips": undeclared}))
    return 1 if counts["FAIL"] or undeclared or not results else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
