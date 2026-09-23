import importlib.util
from pathlib import Path
import socket

import pytest

spec = importlib.util.spec_from_file_location(
    "check_hermes_isolation", Path(__file__).parents[1] / "deploy/check_hermes_isolation.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)


def test_secret_environment_reports_names_only():
    names = isolation.secret_environment_names({
        "DISCORD_TOKEN": "secret-one", "PETERBOT_RUNNER_TOKEN": "secret-two",
        "AWS_SECRET_ACCESS_KEY": "secret-three", "SOME_API_KEY": "secret-four",
        "HOME": "/workspace", "PETERBOT_ISOLATION_RUNNER_HOST": "192.168.64.2",
        "EMPTY_TOKEN": "",
    })
    assert names == ["AWS_SECRET_ACCESS_KEY", "DISCORD_TOKEN", "PETERBOT_RUNNER_TOKEN", "SOME_API_KEY"]
    assert "secret" not in str(names)


def test_host_execution_fails_before_probes(monkeypatch):
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 501)
    monkeypatch.setattr(isolation, "probe_write", lambda _: pytest.fail("Host write probe"))
    monkeypatch.setattr(isolation, "tcp_blocked", lambda *args: pytest.fail("Host network probe"))
    result = isolation.run_checks({})
    assert not result["passed"]
    assert len(result["checks"]) == 2


def test_workspace_probe_removes_its_own_file(tmp_path):
    assert isolation.probe_write(tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert not isolation.probe_write(tmp_path / "missing")


def test_tcp_probes_fail_on_success_and_pass_on_denial(monkeypatch):
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: Connection())
    assert not isolation.tcp_blocked("192.0.2.1", 22)

    def denied(*args, **kwargs):
        raise ConnectionRefusedError()
    monkeypatch.setattr(socket, "create_connection", denied)
    assert isolation.tcp_blocked("192.0.2.1", 22)
    assert not isolation.tcp_blocked("invalid-operator-ip", 22)


def test_gateway_probe_uses_no_authorization_and_requires_exact_401(monkeypatch):
    captured = []
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            captured.append((args, kwargs))
        def getresponse(self):
            return type("Response", (), {"status": 401})()
        def close(self):
            pass
    monkeypatch.setattr(isolation.http.client, "HTTPConnection", Connection)
    assert isolation.gateway_status("192.168.240.2", 8770) == 401
    assert captured == [(("GET", "/v1/models"), {"headers": {"Connection": "close"}})]


@pytest.mark.parametrize("failed_check", [None, "network", "credential", "gateway"])
def test_complete_check_report(monkeypatch, failed_check):
    monkeypatch.setattr(isolation.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(isolation.Path, "exists", lambda _: True)
    monkeypatch.setattr(isolation.Path, "read_text", lambda _: "CapEff:\t0000000000000000\nNoNewPrivs:\t1\n")
    monkeypatch.setattr(isolation, "accessible", lambda _: False)
    monkeypatch.setattr(isolation, "probe_write", lambda path: path == "/workspace")
    hosts = []
    def blocked(host, port):
        hosts.append((host, port))
        return failed_check != "network"
    monkeypatch.setattr(isolation, "tcp_blocked", blocked)
    monkeypatch.setattr(isolation, "gateway_status", lambda *args: 200 if failed_check == "gateway" else 401)
    env = {"PETERBOT_ISOLATION_RUNNER_HOST": "192.168.65.9"}
    if failed_check == "credential":
        env["DISCORD_TOKEN"] = "must-not-print"
    result = isolation.run_checks(env)
    assert result["passed"] is (failed_check is None)
    assert ("192.168.65.9", 8780) in hosts
    assert len(result["checks"]) == 15
    assert {check["check"] for check in result["checks"]} >= {"toolchain_write_denied"}
    # The toolchain probe must require denial: writable /usr/local/bin fails the run.
    paths = []
    def write_probe(path):
        paths.append(path)
        return path in ("/workspace", "/usr/local/bin")
    monkeypatch.setattr(isolation, "probe_write", write_probe)
    assert not isolation.run_checks({})["passed"]
    assert "must-not-print" not in str(result)
