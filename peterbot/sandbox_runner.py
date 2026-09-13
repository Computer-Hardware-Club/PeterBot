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
MAX_REQUEST = 256 * 1024
MAX_RESULT = 128 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_BYTES = 9 * 1024 * 1024
MAX_FILES = 256
WORKER_LABEL = "io.peterbot.worker=hermes"


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


async def docker(args: list[str], *, input_data: bytes | None = None,
                 max_bytes: int = MAX_RESULT, timeout: int = 30, check: bool = True) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdin=asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
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
        feeder = asyncio.create_task(feed())
        try:
            output = bytearray()
            while chunk := await proc.stdout.read(65536):
                output.extend(chunk)
                if len(output) > max_bytes:
                    raise RunnerError("Output limit exceeded")
            await feeder
            if await proc.wait() != 0 and check:
                raise RunnerError("Container command failed")
            return bytes(output)
        finally:
            if not feeder.done():
                feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

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
        await docker(worker_args(self.settings, job.name), max_bytes=4096)
        await docker(["exec", "-i", job.name, "python", "-I", "-m", "peterbot.hermes_worker"],
                     input_data=json.dumps(payload).encode(), max_bytes=MAX_RESULT,
                     timeout=self.settings.timeout, check=False)
        result_tar = await docker(["cp", f"{job.name}:/workspace/.peter-result.json", "-"],
                                  max_bytes=MAX_RESULT + 16384)
        result_files = safe_tar_files(result_tar, byte_limit=MAX_RESULT, file_limit=1)
        result = json.loads(result_files[".peter-result.json"])
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise RunnerError("Invalid worker result")
        status = result.get("status", "completed")
        if status not in ("completed", "failed"):
            status = "failed"
        # Missing artifacts is normal; malformed or excessive artifacts fail closed.
        try:
            archive = await docker(["cp", f"{job.name}:/workspace/artifacts", "-"],
                                   max_bytes=MAX_ARCHIVE_BYTES)
        except RunnerError:
            artifacts = []
        else:
            artifacts = encode_artifacts(safe_tar_files(archive))
        return {"status": status, "answer": result["answer"][:65536], "artifacts": artifacts}

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
            result = {"status": "timeout", "answer": "Task reached its execution limit.", "artifacts": []}
        except asyncio.CancelledError:
            result = {"status": "cancelled", "answer": "Task cancelled.", "artifacts": []}
        except Exception:
            # Container output, request data, and exception strings may contain secrets.
            LOG.warning("Worker task failed")
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
