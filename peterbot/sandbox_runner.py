"""Trusted, narrow Docker supervisor. Never expose this service to task containers."""
from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import re
import tarfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from aiohttp import web

LOG = logging.getLogger(__name__)
MAX_STDERR = 32 * 1024
MAX_REQUEST = 256 * 1024
MAX_RESULT = 128 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 9 * 1024 * 1024
MAX_FILES = 256
WORKER_LABEL = "io.peterbot.worker=hermes"
WORKER_ERROR_CODES = frozenset({
    "invalid_request", "initialization_failed", "model_failed", "execution_failed",
    "result_invalid", "unknown",
})


# Docker cp cannot collect live tmpfs mounts. Read through trusted in-container
# executables instead, without a shell, links, special files, or unbounded reads.
RESULT_READ_SOURCE = """import os, stat, sys
fd = os.open('/workspace/.peter-result.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
with os.fdopen(fd, 'rb') as source:
    metadata = os.fstat(source.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 131072:
        raise ValueError('Invalid result file')
    data = source.read(131073)
    if len(data) > 131072:
        raise ValueError('Result too large')
    sys.stdout.buffer.write(data)
"""

@dataclass(frozen=True)
class Settings:
    token: str
    image: str
    network: str
    scope: str = "peterbot"
    concurrency: int = 2
    timeout: int = 1800

    def __post_init__(self):
        if len(self.token) < 24 or not self.token.isascii() or any(c.isspace() for c in self.token):
            raise ValueError("Runner token must contain at least 24 ASCII characters")
        if not self.image or self.image.startswith("-") or any(c.isspace() for c in self.image):
            raise ValueError("Invalid worker image")
        for value in (self.network, self.scope):
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}", value):
                raise ValueError("Invalid runner network/scope")
        if not 1 <= self.concurrency <= 2 or not 1 <= self.timeout <= 1800:
            raise ValueError("Invalid runner limits")

    @classmethod
    def from_env(cls):
        return cls(
            token=os.environ["PETERBOT_RUNNER_TOKEN"],
            image=os.environ["PETERBOT_WORKER_IMAGE"],
            network=os.environ["PETERBOT_WORKER_NETWORK"],
            scope=os.getenv("PETERBOT_RUNNER_SCOPE", "peterbot"),
            concurrency=int(os.getenv("PETERBOT_RUNNER_CONCURRENCY", "2")),
            timeout=int(os.getenv("PETERBOT_RUNNER_TIMEOUT", "1800")),
        )


def authorized(header: str, token: str) -> bool:
    return hmac.compare_digest(header.encode("utf-8"), ("Bearer " + token).encode("ascii"))


def normalize_job_id(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F-]{32,36}", value):
        raise ValueError("Invalid job_id")
    parsed = uuid.UUID(value)
    if value.lower() not in (parsed.hex, str(parsed)):
        raise ValueError("Invalid job_id")
    return parsed.hex


def worker_args(settings: Settings, name: str) -> list[str]:
    """All container capabilities are fixed here, never taken from a request."""
    return [
        "run", "--detach", "--name", name,
        "--label", WORKER_LABEL, "--label", f"io.peterbot.runner={settings.scope}",
        "--user", "10000:10000", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true", "--read-only",
        "--pids-limit", "128", "--cpus", "2", "--memory", "2g", "--memory-swap", "2g",
        "--network", settings.network, "--dns", "127.0.0.1",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,uid=10000,gid=10000,mode=1777",
        "--tmpfs", "/workspace:rw,nosuid,nodev,size=512m,uid=10000,gid=10000,mode=700",
        "--workdir", "/workspace", "--env", "HOME=/workspace", "--env", "TMPDIR=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "HERMES_HOME=/tmp/hermes",
        "--log-driver", "none", "--entrypoint", "python", settings.image,
        "-c", "import time; time.sleep(1900)",
    ]


class RunnerError(Exception):
    """Internal errors deliberately carry no subprocess output."""


def safe_stderr_diagnostics(data: bytes) -> list[str]:
    """Keep only Python frame metadata under trusted code roots and exception types."""
    diagnostics = []
    frame_pattern = re.compile(
        r'  File "(?P<path>/[a-zA-Z0-9_./-]{1,240}|<frozen [a-zA-Z0-9_.]{1,60}>)", '
        r'line (?P<line>[0-9]{1,7}), in (?P<function>[a-zA-Z_][a-zA-Z0-9_]{0,79}|<module>)'
    )
    exception_pattern = re.compile(r"([A-Za-z][A-Za-z0-9_.]{0,79}(?:Error|Exception)):")
    trusted_roots = ("/app/peterbot/", "/opt/upstream/", "/usr/local/lib/python", "/usr/lib/python")
    exception_type = None
    for line in data[:MAX_STDERR].decode("utf-8", errors="replace").splitlines():
        frame = frame_pattern.fullmatch(line)
        if frame:
            path = frame.group("path")
            if ".." not in PurePosixPath(path).parts and (path.startswith(trusted_roots) or path.startswith("<frozen ")):
                diagnostics.append(line)
                diagnostics = diagnostics[-12:]
        exception = exception_pattern.match(line)
        if exception:
            exception_type = exception.group(1)
    if exception_type:
        diagnostics.append("exception_type=" + exception_type)
    return diagnostics


async def docker(args: list[str], *, input_data: bytes | None = None,
                 max_bytes: int = MAX_RESULT, timeout: int = 30, check: bool = True) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async def communicate():
        async def feed():
            if input_data is not None:
                try:
                    proc.stdin.write(input_data)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    proc.stdin.close()
        async def drain_stderr():
            captured = bytearray()
            while chunk := await proc.stderr.read(65536):
                remaining = MAX_STDERR - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])
            return bytes(captured)

        feeder = asyncio.create_task(feed())
        stderr_reader = asyncio.create_task(drain_stderr())
        try:
            output = bytearray()
            while chunk := await proc.stdout.read(65536):
                output.extend(chunk)
                if len(output) > max_bytes:
                    LOG.warning("Docker output limit operation=%s limit=%d", args[0], max_bytes)
                    raise RunnerError("Output limit exceeded")
            await feeder
            stderr = await stderr_reader
            returncode = await proc.wait()
            if returncode != 0:
                LOG.warning("Docker operation failed operation=%s exit_code=%d", args[0], returncode)
                for diagnostic in safe_stderr_diagnostics(stderr):
                    LOG.warning("Docker traceback operation=%s %s", args[0], diagnostic)
                if check:
                    raise RunnerError("Container command failed")
            return bytes(output)
        finally:
            for task in (feeder, stderr_reader):
                if not task.done():
                    task.cancel()
            await asyncio.gather(feeder, stderr_reader, return_exceptions=True)

    try:
        return await asyncio.wait_for(communicate(), timeout)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


def safe_tar_files(data: bytes, *, byte_limit=MAX_ARTIFACT_BYTES, file_limit=MAX_FILES) -> dict[str, bytes]:
    """Inspect tar streams in memory; never extract paths onto the supervisor."""
    if len(data) > MAX_ARCHIVE_BYTES:
        raise RunnerError("Archive too large")
    files = {}
    total = 0
    entries = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for member in archive:
                entries += 1
                if entries > MAX_FILES * 4:
                    raise RunnerError("Too many archive entries")
                raw = member.name
                path = PurePosixPath(raw)
                if (path.is_absolute() or ".." in path.parts or "\\" in raw
                        or ":" in raw or any(ord(c) < 32 or ord(c) == 127 for c in raw) or len(raw) > 240):
                    raise RunnerError("Unsafe archive path")
                if member.isdir():
                    continue
                if not member.isfile() or member.issparse():
                    raise RunnerError("Unsafe archive member")
                name = path.as_posix().removeprefix("artifacts/")
                if not name or name == "." or name in files:
                    raise RunnerError("Invalid archive member")
                total += member.size
                if member.size < 0 or total > byte_limit or len(files) >= file_limit:
                    raise RunnerError("Artifact limit exceeded")
                stream = archive.extractfile(member)
                if stream is None:
                    raise RunnerError("Unreadable artifact")
                content = stream.read(member.size + 1)
                if len(content) != member.size:
                    raise RunnerError("Truncated artifact")
                files[name] = content
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise RunnerError("Invalid archive") from exc
    return files


def encode_artifacts(files: dict[str, bytes]) -> list[dict[str, str]]:
    if not files:
        return []
    if len(files) > 3:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, content in files.items():
                bundle.writestr(name, content)
        content = output.getvalue()
        if len(content) > MAX_ARTIFACT_BYTES:
            raise RunnerError("Artifact bundle limit exceeded")
        files = {"peter-artifacts.zip": content}
    if sum(map(len, files.values())) > MAX_ARTIFACT_BYTES:
        raise RunnerError("Artifact limit exceeded")
    return [{"name": name, "data_base64": base64.b64encode(data).decode("ascii")}
            for name, data in files.items()]


@dataclass
class Job:
    name: str
    task: asyncio.Task
    phase: str = "queued"


class Runner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, Job] = {}
        self.cleanup_failed = False

    async def remove(self, name):
        try:
            await docker(["rm", "--force", name], max_bytes=4096)
            return True
        except (RunnerError, asyncio.TimeoutError, OSError):
            LOG.warning("Worker cleanup failed; new work disabled until restart")
            self.cleanup_failed = True
            return False

    async def startup(self, app):
        containers = await docker([
            "ps", "--all", "--quiet", "--filter", f"label={WORKER_LABEL}",
            "--filter", f"label=io.peterbot.runner={self.settings.scope}",
        ], max_bytes=65536)
        for container in containers.decode("ascii").split():
            if not re.fullmatch(r"[a-f0-9]{12,64}", container):
                raise RunnerError("Unexpected container identifier")
            if not await self.remove(container):
                raise RunnerError("Stale worker cleanup failed")

    async def shutdown(self, app):
        tasks = [job.task for job in self.jobs.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def execute(self, job: Job, payload: dict):
        job.phase = "create"
        await docker(worker_args(self.settings, job.name), max_bytes=4096)
        job.phase = "exec"
        await docker(["exec", "-i", job.name, "/usr/local/bin/python", "-I", "-m", "peterbot.hermes_worker"],
                     input_data=json.dumps(payload).encode(), max_bytes=MAX_RESULT,
                     timeout=self.settings.timeout, check=False)
        job.phase = "result_copy"
        result_bytes = await docker(["exec", job.name, "/usr/local/bin/python", "-I", "-c", RESULT_READ_SOURCE],
                                    max_bytes=MAX_RESULT)
        job.phase = "result_parse"
        result = json.loads(result_bytes)
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise RunnerError("Invalid worker result")
        status = result.get("status", "completed")
        if status not in ("completed", "failed"):
            status = "failed"
        error_code = result.get("error_code")
        if not isinstance(error_code, str) or error_code not in WORKER_ERROR_CODES:
            error_code = None
        if status == "failed":
            LOG.warning("Worker reported failure phase=worker_result error_code=%s", error_code or "unspecified")
        # Missing artifacts is normal; malformed or excessive artifacts fail closed.
        job.phase = "artifacts_copy"
        try:
            archive = await docker(["exec", job.name, "/bin/tar", "-C", "/workspace", "-cf", "-", "--", "artifacts"],
                                   max_bytes=MAX_ARCHIVE_BYTES)
        except RunnerError as exc:
            LOG.warning("Worker artifacts unavailable phase=%s error_type=%s", job.phase, type(exc).__name__)
            artifacts = []
        else:
            job.phase = "artifacts_parse"
            artifacts = encode_artifacts(safe_tar_files(archive))
        response = {"status": status, "answer": result["answer"][:65536], "artifacts": artifacts}
        if status == "failed" and error_code is not None:
            response["error_code"] = error_code
        job.phase = "complete"
        return response

    async def run(self, request):
        body = await read_body(request)
        if set(body) != {"job_id", "request"} or not isinstance(body["request"], dict):
            raise web.HTTPBadRequest(text="Expected job_id and request object")
        job_id = parse_job_id(body.get("job_id"))
        if self.cleanup_failed:
            raise web.HTTPServiceUnavailable(text="Worker cleanup requires recovery")
        if job_id in self.jobs:
            raise web.HTTPConflict(text="Job already running")
        if len(self.jobs) >= self.settings.concurrency:
            raise web.HTTPTooManyRequests(text="Workers busy")
        job = Job(f"peterbot-{self.settings.scope}-{job_id}", asyncio.current_task())
        self.jobs[job_id] = job
        try:
            result = await asyncio.wait_for(self.execute(job, body["request"]), self.settings.timeout)
        except asyncio.TimeoutError:
            LOG.warning("Worker task timed out phase=%s", job.phase)
            result = {"status": "timeout", "answer": "Task reached its execution limit.", "artifacts": []}
        except asyncio.CancelledError:
            result = {"status": "cancelled", "answer": "Task cancelled.", "artifacts": []}
        except Exception as exc:
            # Container output, request data, and exception strings may contain secrets.
            LOG.warning("Worker task failed phase=%s error_type=%s", job.phase, type(exc).__name__)
            result = {"status": "failed", "answer": "The isolated worker could not complete this task.", "artifacts": []}
        finally:
            cleanup = asyncio.create_task(self.remove(job.name))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
            self.jobs.pop(job_id, None)
        return web.json_response(result)

    async def cancel(self, request):
        body = await read_body(request)
        if set(body) != {"job_id"}:
            raise web.HTTPBadRequest(text="Expected job_id")
        job_id = parse_job_id(body.get("job_id"))
        job = self.jobs.get(job_id)
        if job is not None:
            job.task.cancel()
        return web.json_response({"cancelled": job is not None})


async def read_body(request):
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError):
        raise web.HTTPBadRequest(text="Invalid JSON") from None
    if not isinstance(value, dict):
        raise web.HTTPBadRequest(text="Expected object")
    return value


def parse_job_id(value):
    try:
        return normalize_job_id(value)
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid job_id") from None


def create_app(settings: Settings, *, cleanup_on_start=True):
    runner = Runner(settings)

    @web.middleware
    async def auth(request, handler):
        if request.path != "/health" and not authorized(request.headers.get("Authorization", ""), settings.token):
            raise web.HTTPUnauthorized(text="Unauthorized")
        return await handler(request)

    async def health(request):
        return web.json_response({"status": "degraded" if runner.cleanup_failed else "ok"},
                                 status=503 if runner.cleanup_failed else 200)

    app = web.Application(middlewares=[auth], client_max_size=MAX_REQUEST)
    app[web.AppKey("runner", Runner)] = runner
    app.router.add_get("/health", health)
    app.router.add_post("/run", runner.run)
    app.router.add_post("/cancel", runner.cancel)
    if cleanup_on_start:
        app.on_startup.append(runner.startup)
    app.on_shutdown.append(runner.shutdown)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(create_app(Settings.from_env()), host="0.0.0.0", port=8780,
                access_log=None, handler_cancellation=True)
