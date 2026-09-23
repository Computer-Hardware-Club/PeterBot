"""Verify a disposable Peter worker from INSIDE its production sandbox.

Feed this file to `docker exec -i WORKER python -I -` after creating a worker
with sandbox_runner.worker_args. This does not launch Docker or call an LLM.
The operator must supply PETERBOT_ISOLATION_RUNNER_HOST if its current control
network IP differs from the default. Other endpoints also accept explicit env
values below. Do not inject production tokens into this verification worker.
JSON goes to stdout; exit 1 means a failed check. Network checks take <= 7 sec.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import uuid

TIMEOUT = 1.0
BLOCKED_ENDPOINTS = (
    ("host_gateway_ssh", "PETERBOT_ISOLATION_HOST_GATEWAY", "192.168.240.1", 22),
    ("p910_ssh", "PETERBOT_ISOLATION_P910_HOST", "100.99.6.59", 22),
    ("direct_inference", "PETERBOT_ISOLATION_INFERENCE_HOST", "100.73.210.66", 8000),
    ("public_internet", "PETERBOT_ISOLATION_PUBLIC_HOST", "1.1.1.1", 443),
    ("runner_control", "PETERBOT_ISOLATION_RUNNER_HOST", "192.168.64.2", 8780),
)
FORBIDDEN_PATHS = (
    "/var/run/docker.sock", "/run/docker.sock", "/.env", "/app/.env",
    "/app/config.production.json", "/root/.ssh", "/root/.aws",
    "/root/.config", "/home/peterbot/.ssh", "/run/secrets",
)


def secret_environment_names(environ):
    """Return names only, never values, including credentials beyond Discord."""
    exact = {"DISCORD_TOKEN", "PETERBOT_RUNNER_TOKEN", "RUNNER_TOKEN", "SSH_AUTH_SOCK",
             "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
             "DOCKER_HOST", "DOCKER_CERT_PATH", "KUBECONFIG"}
    suffixes = ("_TOKEN", "_API_KEY", "_PASSWORD", "_SECRET", "_PRIVATE_KEY")
    return sorted(name for name, value in environ.items()
                  if value and (name.upper() in exact or name.upper().endswith(suffixes)))


def accessible(path):
    """Unreadable parent directories count as inaccessible, without escalation."""
    try:
        candidate = Path(path)
        if candidate.is_dir():
            next(candidate.iterdir(), None)
        else:
            with candidate.open("rb") as handle:
                handle.read(1)
        return True
    except (PermissionError, FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        # Socket and other special files may be mounted but cannot be read as files.
        return Path(path).exists()


def probe_write(directory):
    target = Path(directory) / (".peter-isolation-" + uuid.uuid4().hex)
    try:
        with target.open("xb") as output:
            output.write(b"isolation-check\n")
        return True
    except OSError:
        return False
    finally:
        try:
            target.unlink()
        except (FileNotFoundError, PermissionError, OSError):
            pass


def tcp_blocked(host, port):
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False  # A malformed operator target must never produce a pass.
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return False
    except (OSError, TimeoutError):
        return True


def gateway_status(host, port):
    connection = http.client.HTTPConnection(host, port, timeout=TIMEOUT)
    try:
        connection.request("GET", "/v1/models", headers={"Connection": "close"})
        # No capability, no cookies, no auth header, and no environment proxy use.
        return connection.getresponse().status
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()


def run_checks(environ=None):
    environ = os.environ if environ is None else environ
    checks = []

    def record(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    uid = os.geteuid()
    record("worker_uid", uid == 10000, observed=uid)
    inside_container = Path("/.dockerenv").exists()
    record("container_execution", inside_container)
    if uid != 10000 or not inside_container:
        # Prevent accidental execution of the write/network probes on the host.
        return {"passed": False, "checks": checks, "error": "Run inside the UID 10000 worker container"}

    names = secret_environment_names(environ)
    record("no_credential_environment", not names, unexpected_names=names)
    exposed = [path for path in FORBIDDEN_PATHS if accessible(path)]
    record("no_host_credentials_or_docker_socket", not exposed, accessible_paths=exposed)
    record("root_write_denied", not probe_write("/"))
    # The pinned rustc/cargo/node toolchain is trusted image content: read-only like /app.
    record("toolchain_write_denied", not probe_write("/usr/local/bin"))
    record("workspace_write_allowed", probe_write("/workspace"))

    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        record("no_effective_capabilities", int(status.get("CapEff", "-1").strip(), 16) == 0)
        record("no_new_privileges", status.get("NoNewPrivs", "").strip() == "1")
    except (OSError, ValueError):
        record("kernel_security_flags_readable", False)

    for name, env_name, default, port in BLOCKED_ENDPOINTS:
        host = environ.get(env_name, default)
        record(name + "_blocked", tcp_blocked(host, port), host=host, port=port)
    host = environ.get("PETERBOT_ISOLATION_GATEWAY_HOST", "192.168.240.2")
    response_status = gateway_status(host, 8770)
    record("gateway_requires_capability", response_status == 401,
           host=host, port=8770, http_status=response_status)
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
