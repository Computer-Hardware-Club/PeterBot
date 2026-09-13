import asyncio
import base64
import io
import json
import tarfile
import uuid
import zipfile
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from peterbot import sandbox_runner as sr

TOKEN = "test-" + "a" * 32
SETTINGS = sr.Settings(TOKEN, "peterbot-worker:fixed", "peterbot-egress")
JOB_ID = uuid.uuid4().hex


def archive(entries):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "/etc/passwd"
            if kind == tarfile.REGTYPE:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            else:
                tar.addfile(info)
    return buf.getvalue()


def test_auth_uses_constant_time_comparison():
    with patch.object(sr.hmac, "compare_digest", wraps=sr.hmac.compare_digest) as compare:
        assert sr.authorized("Bearer " + TOKEN, TOKEN)
        assert not sr.authorized("bearer " + TOKEN, TOKEN)
        assert not sr.authorized("Bearer wrong", TOKEN)
        assert not sr.authorized("Bearer \N{SNOWMAN}", TOKEN)
        assert compare.call_count == 4


@pytest.mark.parametrize("value", ["--privileged", "../escape", "a" * 33, "g" * 32,
                                  " "+JOB_ID, JOB_ID+"\n", 123, None,
                                  "a"*8+"--"+"a"*22])
def test_malicious_job_ids_rejected(value):
    with pytest.raises(ValueError):
        sr.normalize_job_id(value)


def test_uuid_canonicalization():
    parsed = uuid.UUID(JOB_ID)
    assert sr.normalize_job_id(str(parsed).upper()) == JOB_ID
    assert sr.normalize_job_id(JOB_ID.upper()) == JOB_ID


def test_worker_container_has_only_fixed_capabilities():
    args = sr.worker_args(SETTINGS, "peterbot-test-" + JOB_ID)
    assert args[0] == "run"
    assert args[args.index("--user") + 1] == "10000:10000"
    assert args[args.index("--network") + 1] == SETTINGS.network
    assert args[args.index("--dns") + 1] == "127.0.0.1"
    assert args[args.index("--memory") + 1] == "2g"
    assert args[args.index("--memory-swap") + 1] == "2g"
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges:true" in args
    assert "--read-only" in args
    assert not {"--privileged", "--volume", "-v", "--mount", "--add-host"} & set(args)
    assert TOKEN not in " ".join(args)
    assert args[-3:] == [SETTINGS.image, "-c", "import time; time.sleep(1900)"]
    assert len([a for a in args if a.startswith("/workspace:rw")]) == 1


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                                  tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_artifacts_reject_links_and_devices(kind):
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(archive([("artifacts/unsafe", b"", kind)]))


@pytest.mark.parametrize("name", ["/etc/passwd", "../escape", "artifacts/../escape",
                                  "artifacts/a\\b", "artifacts/a\nname", "x" * 241])
def test_artifacts_reject_unsafe_paths(name):
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(archive([(name, b"data", tarfile.REGTYPE)]))


def test_artifact_limits_and_duplicate_names():
    data = archive([("artifacts/one", b"abc", tarfile.REGTYPE),
                    ("artifacts/two", b"def", tarfile.REGTYPE)])
    assert sr.safe_tar_files(data) == {"one": b"abc", "two": b"def"}
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(data, byte_limit=5)
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(data, file_limit=1)
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(b"x" * (sr.MAX_ARCHIVE_BYTES + 1))
    with pytest.raises(sr.RunnerError):
        sr.safe_tar_files(archive([("artifacts/a", b"one", tarfile.REGTYPE),
                                   ("artifacts/a", b"two", tarfile.REGTYPE)]))


def test_artifacts_pack_many_files_without_extraction():
    files = {f"subdir/{i}.txt": str(i).encode() for i in range(4)}
    result = sr.encode_artifacts(files)
    assert len(result) == 1
    assert result[0]["name"] == "peter-artifacts.zip"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result[0]["data_base64"]))) as bundle:
        assert {name: bundle.read(name) for name in bundle.namelist()} == files
    assert sr.encode_artifacts({"a": b"abc"}) == [{"name": "a", "data_base64": "YWJj"}]


def test_stale_cleanup_is_scoped_to_both_labels():
    calls = []

    async def fake_docker(args, **kwargs):
        calls.append(args)
        return b"abcdef123456\n" if args[0] == "ps" else b""

    async def exercise():
        with patch.object(sr, "docker", fake_docker):
            await sr.Runner(SETTINGS).startup(None)
    asyncio.run(exercise())
    assert calls == [["ps", "--all", "--quiet", "--filter", "label=io.peterbot.worker=hermes",
                      "--filter", "label=io.peterbot.runner=peterbot"],
                     ["rm", "--force", "abcdef123456"]]


def test_http_auth_body_validation_and_health():
    async def exercise():
        async with TestClient(TestServer(sr.create_app(SETTINGS, cleanup_on_start=False))) as client:
            assert (await client.get("/health")).status == 200
            assert (await client.post("/run", json={})).status == 401
            headers = {"Authorization": "Bearer " + TOKEN}
            assert (await client.post("/run", headers=headers, json=[])).status == 400
            assert (await client.post("/run", headers=headers,
                                     json={"job_id": "--privileged", "request": {}})).status == 400
            assert (await client.post("/run", headers=headers,
                                     json={"job_id": JOB_ID, "request": {}, "image": "bad"})).status == 400
            assert (await client.post("/run", headers=headers,
                                     json={"job_id": JOB_ID, "request": {"text": "x"*sr.MAX_REQUEST}})).status == 413
            response = await client.post("/cancel", headers=headers, json={"job_id": JOB_ID})
            assert await response.json() == {"cancelled": False}
    asyncio.run(exercise())


def test_run_returns_artifacts_and_always_cleans_container():
    calls = []
    payloads = []

    async def fake_docker(args, **kwargs):
        calls.append(args)
        if "peterbot.hermes_worker" in args:
            payloads.append(json.loads(kwargs["input_data"]))
        if sr.RESULT_READ_SOURCE in args:
            return b'{"answer":"Done"}'
        if "/bin/tar" in args:
            return archive([("artifacts/report.txt", b"result", tarfile.REGTYPE)])
        return b""

    async def exercise():
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(SETTINGS, cleanup_on_start=False))) as client:
                response = await client.post("/run", headers={"Authorization": "Bearer " + TOKEN},
                                             json={"job_id": JOB_ID, "request": {"task": "hello", "image": "ignored"}})
                assert await response.json() == {"status": "completed", "answer": "Done",
                                                 "artifacts": [{"name": "report.txt", "data_base64": "cmVzdWx0"}]}
    asyncio.run(exercise())
    assert payloads == [{"task": "hello", "image": "ignored"}]
    assert all(call[0] != "cp" for call in calls)
    assert ["exec", f"peterbot-peterbot-{JOB_ID}", "/usr/local/bin/python", "-I", "-c", sr.RESULT_READ_SOURCE] in calls
    assert ["exec", f"peterbot-peterbot-{JOB_ID}", "/bin/tar", "-C", "/workspace", "-cf", "-", "--", "artifacts"] in calls
    assert calls[0][calls[0].index("--entrypoint") + 2] == SETTINGS.image
    assert calls[-1] == ["rm", "--force", f"peterbot-peterbot-{JOB_ID}"]


def test_worker_failure_sanitized_and_cleaned(caplog):
    calls = []

    async def fake_docker(args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            raise RuntimeError("DO NOT LEAK " + TOKEN)
        return b""

    async def exercise():
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(SETTINGS, cleanup_on_start=False))) as client:
                response = await client.post("/run", headers={"Authorization": "Bearer " + TOKEN},
                                             json={"job_id": JOB_ID, "request": {}})
                text = await response.text()
                assert TOKEN not in text
                assert json.loads(text)["status"] == "failed"
    asyncio.run(exercise())
    assert calls[-1][0] == "rm"
    assert "phase=exec error_type=RuntimeError" in caplog.text
    assert TOKEN not in caplog.text
    assert "DO NOT LEAK" not in caplog.text


def test_cancel_and_busy_limits():
    started = None
    calls = []

    async def fake_docker(args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            started.set()
            await asyncio.Event().wait()
        return b""

    async def exercise():
        nonlocal started
        started = asyncio.Event()
        settings = sr.Settings(TOKEN, SETTINGS.image, SETTINGS.network, concurrency=1)
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(settings, cleanup_on_start=False))) as client:
                headers = {"Authorization": "Bearer " + TOKEN}
                task = asyncio.create_task(client.post("/run", headers=headers,
                                                       json={"job_id": JOB_ID, "request": {}}))
                await asyncio.wait_for(started.wait(), 2)
                duplicate = await client.post("/run", headers=headers, json={"job_id": JOB_ID, "request": {}})
                assert duplicate.status == 409
                busy = await client.post("/run", headers=headers, json={"job_id": uuid.uuid4().hex, "request": {}})
                assert busy.status == 429
                cancel = await client.post("/cancel", headers=headers, json={"job_id": JOB_ID})
                assert await cancel.json() == {"cancelled": True}
                response = await asyncio.wait_for(task, 2)
                assert (await response.json())["status"] == "cancelled"
    asyncio.run(exercise())
    assert calls[-1] == ["rm", "--force", f"peterbot-peterbot-{JOB_ID}"]


def test_timeout_cleans_worker():
    calls = []

    async def fake_docker(args, **kwargs):
        calls.append(args)
        if args[0] == "exec":
            await asyncio.Event().wait()
        return b""

    async def exercise():
        settings = sr.Settings(TOKEN, SETTINGS.image, SETTINGS.network, timeout=1)
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(settings, cleanup_on_start=False))) as client:
                response = await client.post("/run", headers={"Authorization": "Bearer " + TOKEN},
                                             json={"job_id": JOB_ID, "request": {}})
                assert (await response.json())["status"] == "timeout"
    asyncio.run(exercise())
    assert calls[-1][0] == "rm"


def test_failed_cleanup_disables_new_jobs_and_health():
    async def fake_docker(args, **kwargs):
        raise sr.RunnerError("private subprocess error")

    async def exercise():
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(SETTINGS, cleanup_on_start=False))) as client:
                headers = {"Authorization": "Bearer " + TOKEN}
                response = await client.post("/run", headers=headers,
                                             json={"job_id": JOB_ID, "request": {}})
                assert (await response.json())["status"] == "failed"
                assert (await client.get("/health")).status == 503
                assert (await client.post("/run", headers=headers,
                                          json={"job_id": uuid.uuid4().hex, "request": {}})).status == 503
    asyncio.run(exercise())


def test_subprocess_output_bound_and_nonzero_result_collection():
    import sys
    original = asyncio.create_subprocess_exec
    children = []
    program = "import sys; sys.stdout.write('x'*100000); sys.exit(1)"

    async def spawn(*args, **kwargs):
        assert args[0] == "docker"
        child = await original(sys.executable, "-c", program, **kwargs)
        children.append(child)
        return child

    async def exercise():
        with patch.object(asyncio, "create_subprocess_exec", spawn):
            with pytest.raises(sr.RunnerError):
                await sr.docker(["fake"], max_bytes=10)
            assert children[-1].returncode is not None
            assert len(await sr.docker(["fake"], max_bytes=200000, check=False)) == 100000
            with pytest.raises(sr.RunnerError):
                await sr.docker(["fake"], max_bytes=200000)
    asyncio.run(exercise())


def test_client_disconnect_removes_worker():
    started = None
    cleaned = None

    async def fake_docker(args, **kwargs):
        if args[0] == "exec":
            started.set()
            await asyncio.Event().wait()
        if args[0] == "rm":
            cleaned.set()
        return b""

    async def exercise():
        nonlocal started, cleaned
        started, cleaned = asyncio.Event(), asyncio.Event()
        with patch.object(sr, "docker", fake_docker):
            server = TestServer(sr.create_app(SETTINGS, cleanup_on_start=False), handler_cancellation=True)
            async with TestClient(server) as client:
                task = asyncio.create_task(client.post("/run", headers={"Authorization": "Bearer " + TOKEN},
                                                       json={"job_id": JOB_ID, "request": {}}))
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.wait_for(cleaned.wait(), 2)
    asyncio.run(exercise())


@pytest.mark.parametrize("code", ["model_failed", "private-error-" + TOKEN, {"secret": TOKEN}])
def test_worker_error_codes_are_allowlisted_in_response_and_logs(caplog, code):
    async def fake_docker(args, **kwargs):
        if sr.RESULT_READ_SOURCE in args:
            result = {"status": "failed", "answer": "Task failed", "error_code": code}
            return json.dumps(result).encode()
        if "/bin/tar" in args:
            return archive([])
        return b""

    async def exercise():
        with patch.object(sr, "docker", fake_docker):
            async with TestClient(TestServer(sr.create_app(SETTINGS, cleanup_on_start=False))) as client:
                response = await client.post("/run", headers={"Authorization": "Bearer " + TOKEN},
                                             json={"job_id": JOB_ID, "request": {}})
                result = await response.json()
                if code == "model_failed":
                    assert result["error_code"] == code
                else:
                    assert "error_code" not in result
    asyncio.run(exercise())
    assert "phase=worker_result" in caplog.text
    assert TOKEN not in caplog.text


def test_stderr_diagnostics_allow_only_safe_frames_and_exception_classes():
    stderr = ('Traceback (most recent call last):\n'
              '  File "<frozen runpy>", line 198, in _run_module_as_main\n'
              '  File "/app/peterbot/hermes_worker.py", line 430, in main\n'
              '    secret = "' + TOKEN + '"\n'
              '  File "/workspace/' + TOKEN + '.py", line 1, in bad\n'
              '  File "/app/peterbot/../../' + TOKEN + '", line 1, in bad\n'
              '  File "/app/peterbot/hermes_worker.py", line 9, in bad\x1b[31m\n'
              'PermissionError: secret token = ' + TOKEN + '\n').encode()
    diagnostics = sr.safe_stderr_diagnostics(stderr)
    assert diagnostics == [
        '  File "<frozen runpy>", line 198, in _run_module_as_main',
        '  File "/app/peterbot/hermes_worker.py", line 430, in main',
        'exception_type=PermissionError',
    ]
    assert TOKEN not in str(diagnostics)


def test_nonzero_subprocess_drains_stderr_without_logging_secrets(caplog):
    import sys
    original = asyncio.create_subprocess_exec
    initial = ('  File "/app/peterbot/hermes_worker.py", line 430, in main\n'
               'PermissionError: ' + TOKEN + '\n')
    program = ('import sys\n'
               'sys.stderr.write(' + repr(initial) + ')\n'
               'sys.stderr.write("x" * 200000)\n'
               'sys.stderr.write("\\nRuntimeError: discarded past cap\\n")\n'
               'sys.stdout.write(' + repr(TOKEN) + ')\n'
               'sys.exit(1)\n')

    async def spawn(*args, **kwargs):
        return await original(sys.executable, "-c", program, **kwargs)

    async def exercise():
        with patch.object(asyncio, "create_subprocess_exec", spawn):
            assert await sr.docker(["exec"], check=False, timeout=3) == TOKEN.encode()
    asyncio.run(exercise())
    assert "exit_code=1" in caplog.text
    assert 'hermes_worker.py", line 430, in main' in caplog.text
    assert "exception_type=PermissionError" in caplog.text
    assert "RuntimeError" not in caplog.text
    assert TOKEN not in caplog.text
    assert "x" * 100 not in caplog.text


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "oversized"])
def test_tmpfs_result_reader_is_bounded_regular_file_only(tmp_path, kind):
    import os
    import subprocess
    import sys
    path = tmp_path / "result.json"
    expected = b'{"answer":"collected from live mount"}'
    if kind == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(expected)
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(expected if kind == "regular" else b"x" * (sr.MAX_RESULT + 1))
    # Substitute only the hardcoded path for this filesystem test; the production
    # command takes no caller-controlled path, source, or environment.
    source = sr.RESULT_READ_SOURCE.replace("'/workspace/.peter-result.json'", repr(str(path)))
    result = subprocess.run([sys.executable, "-I", "-c", source], capture_output=True, timeout=3)
    if kind == "regular":
        assert result.returncode == 0
        assert result.stdout == expected
    else:
        assert result.returncode != 0
        assert result.stdout == b""
