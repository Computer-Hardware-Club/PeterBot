"""PETER-13 controlled dependency access: validation core, broker, worker staging.

Network tests run a local HTTPS registry double. Two deliberate seams keep it
deterministic without weakening what is under test:
  * PublicOnly replaces `_address_blocked` with an allow-list, so the broker's
    real decision path ("only these answers may be dialed") is exercised.
  * the connector resolver validates answers through the production
    `_ValidatedResolver`, then rewrites the dialed address/port to the double
    (URLs still validate as 443-only against the host allow-list).
Everything else — redirect walking, content types, budgets, hash checks,
metadata parsing — is the production code path.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import zipfile
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler

import aiohttp
import pytest

from peterbot import package_access as pa
from peterbot.hermes_worker import Broker, stage_dependency
from peterbot.package_access import (
    PACKAGE_INVENTORY, PackageBroker, PackageError, cache_versions, image_lookup,
    index_dir, validate_arguments, validate_package_request, _address_blocked,
    _safe_url, _ValidatedResolver, _WHEEL_TAG_RE)

HOSTS = ("pypi.org", "files.pythonhosted.org", "index.crates.io", "static.crates.io")
PUBLIC = ("127.0.0.1",)

WHEEL = io.BytesIO()
with zipfile.ZipFile(WHEEL, "w") as archive:
    archive.writestr("six.py", "VERSION = '1.17.0'\n")
    archive.writestr("six-1.17.0.dist-info/METADATA",
                     "Metadata-Version: 2.1\nName: six\nVersion: 1.17.0\n")
    archive.writestr("six-1.17.0.dist-info/WHEEL",
                     "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py2-none-any\nTag: py3-none-any\n")
    archive.writestr("six-1.17.0.dist-info/RECORD", "")
WHEEL_BYTES = WHEEL.getvalue()
WHEEL_SHA = hashlib.sha256(WHEEL_BYTES).hexdigest()
CRATE_BYTES = gzip.compress(b"crate payload")
CRATE_SHA = hashlib.sha256(CRATE_BYTES).hexdigest()

CERT_DIR = Path(tempfile.mkdtemp(prefix="p13-reg-"))
_ORIGINAL_SAFE_URL = pa._safe_url


def _ensure_cert() -> tuple[Path, Path] | None:
    key, crt = CERT_DIR / "k.pem", CERT_DIR / "c.pem"
    if key.exists() and crt.exists():
        return key, crt
    if shutil.which("openssl") is None:
        return None
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(crt), "-days", "2",
                    "-subj", "/CN=peterbot-test",
                    "-addext", "subjectAltName=" + ",".join(f"DNS:{h}" for h in HOSTS)],
                   check=True, capture_output=True)
    return key, crt


def wheel_doc(name="distro", version="1.9.0", *, path="/file", sha=None, yanked=False):
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return {"info": {"name": name, "version": version, "yanked": yanked},
            "urls": [{"packagetype": "bdist_wheel", "filename": filename,
                      "url": f"https://files.pythonhosted.org{path}",
                      "digests": {"sha256": sha or WHEEL_SHA},
                      "size": len(WHEEL_BYTES)}]}


def crate_line(name="itoa", version="1.0.15", **over):
    entry = {"name": name, "vers": version,
             "deps": [{"name": "serde", "req": "^1", "optional": True, "default_features": True,
                       "features": [], "target": None, "kind": "normal"}],
             "cksum": CRATE_SHA, "features": {}, "yanked": False, "size": len(CRATE_BYTES)}
    entry.update(over)
    return json.dumps(entry, separators=(",", ":"))


def make_registry():
    from aiohttp import web

    state = {"doc": wheel_doc(), "raw": None, "index": (crate_line() + "\n").encode(),
             "redirect_to": "", "crate": CRATE_BYTES}

    async def catch(request):
        path = request.path
        if path.startswith("/pypi/") and path.endswith("/json") or path == "/json":
            if state["raw"] is not None:
                return web.Response(body=state["raw"], content_type="application/json")
            if state["doc"] is None:
                return web.json_response({"message": "no such release"}, status=404)
            return web.json_response(state["doc"])
        if path == "/redirect":
            raise web.HTTPFound(state["redirect_to"])
        if path == "/file":
            return web.Response(body=WHEEL_BYTES, content_type="application/octet-stream")
        if path == "/boom":
            return web.json_response({"error": "provider down"}, status=500)
        if path == "/html":
            return web.Response(body=b"<html>json?</html>", content_type="text/html")
        if path == "/huge":
            response = web.StreamResponse()
            await response.prepare(request)
            for _ in range(8):
                await response.write(b"x" * 16384)
            await response.write_eof()
            return response
        if path.startswith("/crates/"):
            return web.Response(body=state["crate"], content_type="application/octet-stream")
        if path.startswith("/it/oa/itoa"):
            return web.Response(body=state["index"], content_type="text/plain")
        return web.json_response({"message": "no route"}, status=404)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", catch)
    return app, state


class PublicOnly:
    """Seam: only the listed addresses are 'public'; everything else is blocked."""

    def __init__(self, allowed=PUBLIC):
        self.allowed = {str(ipaddress.ip_address(text)) for text in allowed}

    def __enter__(self):
        self._saved = pa._address_blocked
        pa._address_blocked = lambda ip: str(ip) not in self.allowed
        return self

    def __exit__(self, *exc):
        pa._address_blocked = self._saved
        return False


def _safe_url_allow_port(port: int):
    def wrapper(url, hosts):
        try:
            return _ORIGINAL_SAFE_URL(url, hosts)
        except PackageError:
            stripped = url.replace(f":{port}", "", 1) if f":{port}" in url else None
            if stripped is None:
                raise
            _ORIGINAL_SAFE_URL(stripped, hosts)  # every non-port rule still applies
            return url
    return wrapper


class _LocalDialResolver(_ValidatedResolver):
    """Production validation first; then dial the local double, never the answer."""

    def __init__(self, real_port: int):
        self.real_port = real_port

    async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
        answers = await super().resolve(host, port, family)
        for answer in answers:
            answer["host"] = "127.0.0.1"
            answer["port"] = self.real_port
            answer["family"] = socket.AF_INET
        return answers


def with_registry(action, dns=None, *, public=PUBLIC):
    """Serve the registry double, pin DNS, run action(broker, port, state)."""
    certs = _ensure_cert()
    if certs is None:
        pytest.skip("openssl unavailable for local registry TLS")
    key, crt = certs
    server_ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ssl.load_cert_chain(crt, key)
    dns = dict(dns or {})

    async def scenario():
        from aiohttp import web
        app, state = make_registry()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=server_ssl)
        await site.start()
        port = int(runner.addresses[0][1])

        answers = {host: ([value] if isinstance(value, str) else list(value))
                   for host, value in [(host, "127.0.0.1") for host in HOSTS] + list(dns.items())}

        async def fake_getaddrinfo(host, p, family=0, type=0, proto=0, flags=0):
            seq = answers.setdefault(host, ["127.0.0.1"])
            text = seq.pop(0) if len(seq) > 1 else seq[0]
            fam = socket.AF_INET6 if ":" in text else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, proto or socket.IPPROTO_TCP, "",
                     (text, p) if ":" not in text else (text, p, 0, 0))]

        asyncio.get_running_loop().getaddrinfo = fake_getaddrinfo
        real_init = aiohttp.TCPConnector.__init__
        client_ssl = ssl.create_default_context(cafile=crt)

        def init(self, *args, resolver=None, **kwargs):
            kwargs["ssl"] = client_ssl
            real_init(self, *args, resolver=_LocalDialResolver(port), **kwargs)

        aiohttp.TCPConnector.__init__ = init
        pa._safe_url = _safe_url_allow_port(port)
        try:
            with PublicOnly(public):
                broker = PackageBroker(cache_dir=CERT_DIR / "no-cache-here")
                return await action(broker, port, state)
        finally:
            aiohttp.TCPConnector.__init__ = real_init
            pa._safe_url = _ORIGINAL_SAFE_URL
            await runner.cleanup()

    return asyncio.run(scenario())


async def must_raise(coro):
    try:
        await coro
    except PackageError as exc:
        return exc
    pytest.fail("expected PackageError")


def codes(func, *args, **kwargs):
    with pytest.raises(PackageError) as excinfo:
        func(*args, **kwargs)
    return excinfo.value


# --- pure validation ----------------------------------------------------------

def test_address_policy_blocks_every_private_or_trick_form():
    blocked = ["10.1.2.3", "172.16.0.1", "192.168.240.2", "127.0.0.1", "169.254.169.254",
               "100.64.0.1", "100.100.100.100", "0.0.0.0", "224.0.0.1", "240.0.0.1",
               "198.18.0.1", "192.0.2.1", "198.51.100.1", "203.0.113.9",
               "::1", "::", "fe80::1", "fc00::1", "fd12:3456::7", "ff02::1",
               "2002:0a00:0001::", "::ffff:10.0.0.1", "::ffff:127.0.0.1",
               "64:ff9b::0a00:0001", "64:ff9b::"]
    for text in blocked:
        assert _address_blocked(ipaddress.ip_address(text)), text
    for text in ("8.8.8.8", "2606:4700:4700::1111", "151.101.0.223", "1.1.1.1"):
        assert not _address_blocked(ipaddress.ip_address(text)), text


@pytest.mark.parametrize("url", [
    "http://files.pythonhosted.org/a/x.whl",
    "https://pypi.org.evil.example/a/x.whl",
    "https://user:***@files.pythonhosted.org/a/x.whl",
    "https://user@files.pythonhosted.org/a/x.whl",
    "https://127.0.0.1/a/x.whl",
    "https://[::1]/a/x.whl",
    "https://169.254.169.254/latest/meta-data",
    "https://files.pythonhosted.org/a/../x.whl",
    "https://files.pythonhosted.org/%2e%2e/x.whl",
    "https://files.pythonhosted.org:8443/a/x.whl",
    "https://files.pythonhosted.org/a?redirect=1",
    "https://files.pythonhosted.org/a#frag",
    "https://files.pythonhosted.org/a b/x.whl",
    "https://files.pythonhosted.org/a\\x01",
    "ftp://files.pythonhosted.org/a/x.whl",
    "https://index.crates.io" + "/a" * 700,
])
def test_safe_url_rejects(url):
    codes(_safe_url, url, pa.ALL_PACKAGE_HOSTS)


def test_safe_url_accepts_registry_shapes():
    for url, hosts in [
        ("https://pypi.org/pypi/six/1.17.0/json", pa.PYPI_METADATA_HOSTS),
        ("https://files.pythonhosted.org/packages/b7/ce/x.whl", pa.PYPI_FILE_HOSTS),
        ("https://index.crates.io/it/oa/itoa", pa.CARGO_HOSTS),
        ("https://static.crates.io/crates/itoa/itoa-1.0.15.crate", pa.CARGO_HOSTS),
    ]:
        assert _safe_url(url, hosts) == url


@pytest.mark.parametrize("registry,name,version", [
    ("npm", "six", "1.0"), ("PYPY", "six", "1.0"), ("pypi", "", "1.0"), ("pypi", "six", ""),
    ("pypi", "six/../x", "1.0"), ("pypi", "_six", "1.0"), ("pypi", "-six", "1.0"),
    ("pypi", "s" * 101, "1.0"), ("pypi", "six", "1.0\n"), ("pypi", "six", "ü"),
    ("pypi", "六", "1.0"), ("pypi", "six", "?" * 65), ("pypi", "si x", "1.0"),
    ("cratesio", "Itio", "1.0"), ("cratesio", "itoa", "v1.0"), ("cratesio", "itoa", "1.0/../"),
    ("cratesio", "itoa", "1.0:9999"), ("cratesio", "1toa", "1.0"),
])
def test_malformed_names_versions_rejected(registry, name, version):
    codes(validate_package_request, registry, name, version)


def test_valid_names_normalize_and_arguments_validate():
    assert validate_package_request("pypi", "Python_DateUtil", "2.9.0.post0") == \
        ("pypi", "python-dateutil", "2.9.0.post0")
    assert validate_package_request("cratesio", "itoa", "1.0.15") == ("cratesio", "itoa", "1.0.15")
    assert validate_arguments({"registry": "pypi", "name": "six", "version": "1.17.0"}) == \
        {"registry": "pypi", "name": "six", "version": "1.17.0"}
    codes(validate_arguments, {"registry": "pypi", "name": "six", "version": "1", "url": "x"})
    codes(validate_arguments, "pypi/six/1")


def test_wheel_tag_policy_pure_only():
    for name in ("six-1.17.0-py2.py3-none-any.whl", "idna-3.10-py3-none-any.whl",
                 "python_dateutil-2.9.0.post0-py2.py3-none-any.whl"):
        assert _WHEEL_TAG_RE.search(name), name
    for name in ("x-1.0-cp312-cp312-linux_x86_64.whl", "x-1.0-py3-none-win_amd64.whl",
                 "x-1.0.tar.gz", "x-1.0-py3-abi3-manylinux_2_17_aarch64.whl"):
        assert not _WHEEL_TAG_RE.search(name), name


def test_inventory_shape_urls_and_index_layout():
    seen = set()
    for entry in PACKAGE_INVENTORY:
        key = (entry["registry"], entry["name"], entry["version"])
        assert key not in seen
        seen.add(key)
        validate_package_request(*key)
        _safe_url(entry["url"], pa.PYPI_FILE_HOSTS if entry["registry"] == "pypi" else pa.CARGO_HOSTS)
        assert len(entry["sha256"]) == 64 and int(entry["size"]) > 0
        if entry["registry"] == "cratesio":
            line = json.loads(entry["index_line"])
            assert line["name"] == entry["name"] and line["vers"] == entry["version"]
            assert line["cksum"] == entry["sha256"] and line["yanked"] is False
    assert index_dir("a") == "1/a" and index_dir("ab") == "2/ab"
    assert index_dir("abc") == "3/a/abc" and index_dir("itoa") == "it/oa/itoa"
    assert image_lookup("pypi", "SIX", "1.17.0")["filename"] == "six-1.17.0-py2.py3-none-any.whl"
    assert image_lookup("pypi", "six", "9.9.9") is None and image_lookup("pypi", "nope", "1") is None


def test_cache_versions_rejects_poisoned_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "wheels").mkdir()
    manifest.write_text(json.dumps({"version": 1, "packages": [
        {"registry": "pypi", "name": "six", "version": "1.17.0"},
        {"registry": "pypi", "name": "si\rmew", "version": "1.0"},
        {"registry": "pypi", "name": "ok", "version": "../bad"},
        "junk"]}))
    assert cache_versions(tmp_path) == {"pypi/six": ["1.17.0"]}
    manifest.write_text("not json")
    assert cache_versions(tmp_path) == {}
    manifest.write_text(json.dumps({"version": 2, "packages": []}))
    assert cache_versions(tmp_path) == {}


# --- broker happy paths over the TLS double --------------------------------------

def test_pypi_pure_wheel_acquire_hash_verified():
    async def action(broker, port, state):
        return await broker.serve({"registry": "pypi", "name": "distro", "version": "1.9.0"})

    result = with_registry(action)
    assert result.data == WHEEL_BYTES and result.sha256 == WHEEL_SHA
    assert result.source == "pypi" and result.filename.endswith("-py3-none-any.whl")


def test_crate_acquire_verifies_cksum_and_returns_index_line():
    async def action(broker, port, state):
        saved = pa.PACKAGE_INVENTORY
        pa.PACKAGE_INVENTORY = ()  # force the public-registry path, not the image cache
        try:
            return await broker.serve({"registry": "cratesio", "name": "itoa", "version": "1.0.15"})
        finally:
            pa.PACKAGE_INVENTORY = saved

    result = with_registry(action)
    assert result.data == CRATE_BYTES and result.sha256 == CRATE_SHA
    assert result.source == "cratesio" and json.loads(result.index_line)["vers"] == "1.0.15"


def test_image_cache_serves_without_any_network(tmp_path, monkeypatch):
    entry = image_lookup("pypi", "six", "1.17.0")
    cache = tmp_path / "deps"
    (cache / "wheels").mkdir(parents=True)
    (cache / "wheels" / entry["filename"]).write_bytes(WHEEL_BYTES)
    monkeypatch.setattr(pa, "PACKAGE_INVENTORY",
                        (dict(entry, sha256=WHEEL_SHA, size=len(WHEEL_BYTES)),))

    def no_transport(*args, **kwargs):
        raise AssertionError("image cache path must not build a connector")

    monkeypatch.setattr(aiohttp, "TCPConnector", no_transport)
    broker = PackageBroker(cache)
    result = asyncio.run(broker.serve({"registry": "pypi", "name": "six", "version": "1.17.0"}))
    assert result.source == "image_cache" and result.sha256 == WHEEL_SHA


# --- broker attack paths ------------------------------------------------------------

def test_pypi_wrong_hash_raises():
    async def action(broker, port, state):
        state["doc"] = wheel_doc(sha="0" * 64)
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    assert with_registry(action).code == "hash_mismatch"


def test_crate_hash_mismatch_and_not_gzip():
    async def wrong_hash(broker, port, state):
        state["crate"] = gzip.compress(b"different bytes entirely")
        return await must_raise(broker._acquire_crate("itoa", "1.0.15", pa.MAX_PACKAGE_BYTES))

    async def not_gzip(broker, port, state):
        payload = b"#!/bin/sh\r\nnope\r\n"
        state["crate"] = payload
        state["index"] = (crate_line(cksum=hashlib.sha256(payload).hexdigest()) + "\n").encode()
        return await must_raise(broker._acquire_crate("itoa", "1.0.15", pa.MAX_PACKAGE_BYTES))

    assert with_registry(wrong_hash).code == "hash_mismatch"
    assert with_registry(not_gzip).code == "poisoned_metadata"


@pytest.mark.parametrize("location,code", [
    ("https://evil.example/file", "poisoned_metadata"),            # host allow-list
    ("https://169.254.169.254/file", "poisoned_metadata"),         # metadata IP literal
    ("http://files.pythonhosted.org/file", "poisoned_metadata"),   # scheme downgrade
    ("https://u:***@files.pythonhosted.org/file", "poisoned_metadata"),  # credentials
])
def test_redirect_to_private_or_offlist_host_rejected(location, code):
    async def action(broker, port, state):
        state["doc"] = wheel_doc(path="/redirect")
        state["redirect_to"] = location
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    assert with_registry(action).code == code


def test_redirect_resolving_to_private_address_blocked():
    async def action(broker, port, state):
        state["doc"] = wheel_doc(path="/redirect")
        state["redirect_to"] = "/file"
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    # Redirect target passes the allow-list; the resolver must still block the dial.
    assert with_registry(action, dns={"files.pythonhosted.org": ["127.0.0.1", "10.0.0.9"]}).code == \
        "blocked_address"


def test_dns_rebinding_between_preflight_and_connect_blocked():
    async def action(broker, port, state):
        state["doc"] = wheel_doc(path="/redirect")
        state["redirect_to"] = "/file"
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    # First answer (preflight) looks public; the connect-time answer is the metadata IP.
    assert with_registry(action, public=("127.0.0.1", "93.184.216.34"),
                         dns={"files.pythonhosted.org": ["93.184.216.34", "169.254.169.254"]}
                         ).code == "blocked_address"


@pytest.mark.parametrize("ip", ["10.0.0.9", "192.168.240.9", "169.254.169.254",
                                "127.0.0.1 ", "::1", "fd00::9", "::ffff:10.0.0.9",
                                "2002:0a00:0001::"])
def test_private_addresses_blocked_at_every_hop(ip):
    async def action(broker, port, state):
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    assert with_registry(action, dns={"files.pythonhosted.org": ip}).code == \
        "blocked_address"


def test_oversized_body_rejected_declared_and_streamed():
    async def streamed(broker, port, state):
        return await must_raise(broker._fetch("https://files.pythonhosted.org/huge",
                                              pa.PYPI_FILE_HOSTS, 64 * 1024, ()))

    async def declared(broker, port, state):
        state["doc"] = dict(wheel_doc(), urls=[dict(wheel_doc()["urls"][0],
                                                    size=pa.MAX_PACKAGE_BYTES + 1)])
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    async def crate_budget(broker, port, state):
        state["index"] = (crate_line(size=5_000_000) + "\n").encode()
        return await must_raise(broker._acquire_crate("itoa", "1.0.15", 1024))

    assert with_registry(streamed).code == "oversized"
    assert with_registry(declared).code == "oversized"
    assert with_registry(crate_budget).code == "oversized"


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s.update(raw=b"not json at all"), "poisoned_metadata"),
    (lambda s: s.update(doc={"info": "not-a-dict", "urls": []}), "poisoned_metadata"),
    (lambda s: s.update(doc=wheel_doc(name="other-project")), "poisoned_metadata"),
    (lambda s: s.update(doc={**wheel_doc(), "info": dict(wheel_doc()["info"], version="9.9.9")}),
     "poisoned_metadata"),
    (lambda s: s.update(doc={**wheel_doc(), "urls": "nope"}), "poisoned_metadata"),
    (lambda s: s.update(doc=wheel_doc(yanked=True)), "yanked"),
    (lambda s: s.update(doc={**wheel_doc(), "urls": [dict(wheel_doc()["urls"][0],
                                                          digests={"sha256": "zz"})]}),
     "poisoned_metadata"),
    (lambda s: s.update(doc={"info": wheel_doc()["info"],
                             "urls": [wheel_doc()["urls"][0],
                                      dict(wheel_doc()["urls"][0],
                                           filename="distro-1.9.0-py2-none-any.whl")]}),
     "no_artifact"),
    (lambda s: s.update(doc={"info": wheel_doc()["info"],
                             "urls": [dict(wheel_doc()["urls"][0], packagetype="sdist",
                                           filename="distro-1.9.0.tar.gz")]}),
     "no_artifact"),
])
def test_poisoned_pypi_metadata_rejected(mutate, code):
    async def action(broker, port, state):
        mutate(state)
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    assert with_registry(action).code == code


def test_wrong_content_type_rejected():
    async def action(broker, port, state):
        return await must_raise(broker._fetch("https://pypi.org/html", pa.PYPI_METADATA_HOSTS,
                                              4096, ("application/json",)))

    assert with_registry(action).code == "poisoned_metadata"


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s.update(index=b"garbage line\n"), "poisoned_metadata"),
    (lambda s: s.update(index=(crate_line(name="someone-else") + "\n").encode()), "poisoned_metadata"),
    (lambda s: s.update(index=(crate_line() + "\n" + crate_line() + "\n").encode()), "poisoned_metadata"),
    (lambda s: s.update(index=(crate_line(cksum="nope") + "\n").encode()), "poisoned_metadata"),
    (lambda s: s.update(index=(crate_line(yanked=True) + "\n").encode()), "yanked"),
    (lambda s: s.update(index=(crate_line(deps="x") + "\n").encode()), "poisoned_metadata"),
    (lambda s: s.update(index=b""), "no_artifact"),
    (lambda s: s.update(index=(crate_line(version="2.0.0") + "\n").encode()), "no_artifact"),
])
def test_poisoned_crate_index_rejected(mutate, code):
    async def action(broker, port, state):
        mutate(state)
        return await must_raise(broker._acquire_crate("itoa", "1.0.15", pa.MAX_PACKAGE_BYTES))

    assert with_registry(action).code == code


def test_provider_unavailable_stays_machine_readable():
    async def status(broker, port, state):
        return await must_raise(broker._fetch("https://pypi.org/boom", pa.PYPI_METADATA_HOSTS,
                                              4096, ("application/json",)))

    async def missing(broker, port, state):
        state["doc"] = None
        return await must_raise(broker._acquire_pypi("distro", "1.9.0", pa.MAX_PACKAGE_BYTES))

    assert with_registry(status).code == "provider_unavailable"
    assert with_registry(missing).code == "provider_unavailable"
    assert with_registry(status).status == 503


def test_quota_and_cache_corruption(tmp_path):
    cache = tmp_path / "deps"
    (cache / "wheels").mkdir(parents=True)
    entry = image_lookup("pypi", "six", "1.17.0")
    wheel = cache / "wheels" / entry["filename"]

    import peterbot.package_access as module
    saved = module.PACKAGE_INVENTORY
    module.PACKAGE_INVENTORY = (dict(entry, sha256=WHEEL_SHA, size=len(WHEEL_BYTES)),)
    try:
        wheel.write_bytes(WHEEL_BYTES)
        broker = PackageBroker(cache)
        args = {"registry": "pypi", "name": "six", "version": "1.17.0"}
        assert codes(lambda: asyncio.run(broker.serve(args, quota=0))).code == "over_quota"
        assert asyncio.run(broker.serve(args, quota=len(WHEEL_BYTES))).sha256 == WHEEL_SHA
        wheel.write_bytes(b"tampered")
        assert codes(lambda: asyncio.run(broker.serve(args))).code == "cache_corrupted"
        wheel.write_bytes(WHEEL_BYTES)
        assert codes(lambda: asyncio.run(broker.serve(args, quota=16))).code == "over_quota"
    finally:
        module.PACKAGE_INVENTORY = saved


# --- worker staging (offline doubles) ---------------------------------------------

def synthetic_cache(tmp_path, monkeypatch):
    cache = tmp_path / "deps"
    (cache / "wheels").mkdir(parents=True)
    (cache / "crates").mkdir(parents=True)
    entries = []
    for entry in PACKAGE_INVENTORY:
        folder = "wheels" if entry["registry"] == "pypi" else "crates"
        (cache / folder / entry["filename"]).write_bytes(WHEEL_BYTES)
        entries.append(dict(entry, sha256=WHEEL_SHA, size=len(WHEEL_BYTES)))
    (cache / "manifest.json").write_text(json.dumps({"version": 1, "packages": entries}))
    monkeypatch.setattr(pa, "PACKAGE_INVENTORY", tuple(entries))
    return cache


class FakeResponse:
    def __init__(self, data=b"", headers=None):
        self.data, self.headers = data, headers or {}

    def read(self, limit=-1):
        return self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.requests = response, error, []

    def open(self, request, timeout=None):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


def fake_broker(response=None, error=None):
    broker = Broker("http://gateway:8770", "t" * 32, job_id=None)
    broker.opener = FakeOpener(response, error)
    return broker


def broker_headers(name="distro", version="1.9.0", source="pypi", index_line=""):
    return {"X-Peterbot-Sha256": WHEEL_SHA,
            "X-Peterbot-Filename": f"{name.replace('-', '_')}_{version}-py3-none-any.whl",
            "X-Peterbot-Size": str(len(WHEEL_BYTES)),
            "X-Peterbot-Source": source,
            "X-Peterbot-Index-Line": index_line}


def test_worker_stages_pinned_wheel_from_image_cache(tmp_path, monkeypatch):
    cache = synthetic_cache(tmp_path, monkeypatch)
    workspace = tmp_path / "ws"
    result = stage_dependency({"registry": "pypi", "name": "Six", "version": "1.17.0"},
                              workspace, cache_dir=cache, cargo_home=tmp_path / "cargo")
    six = image_lookup("pypi", "six", "1.17.0")
    assert result["source"] == "image_cache" and result["status"] == "ok"
    assert result["path"] == str(workspace / "deps" / "wheels" / six["filename"])
    assert Path(result["path"]).read_bytes() == WHEEL_BYTES
    assert "pip install" in result["install"] and "--no-index" in result["install"]
    assert "base64" not in json.dumps(result).lower()


def test_worker_stages_pinned_crate_into_offline_local_registry(tmp_path, monkeypatch):
    cache = synthetic_cache(tmp_path, monkeypatch)
    cargo = tmp_path / "cargo"
    result = stage_dependency({"registry": "cratesio", "name": "itoa", "version": "1.0.15"},
                              tmp_path / "ws", cache_dir=cache, cargo_home=cargo)
    assert result["status"] == "ok" and result["source"] == "image_cache"
    crate = Path(result["path"])
    assert crate == cargo / "registry" / "itoa-1.0.15.crate" and crate.read_bytes() == WHEEL_BYTES
    index_path = Path(result["index"])
    assert index_path == cargo / "registry" / "index" / "it" / "oa" / "itoa"
    assert json.loads(index_path.read_text())["vers"] == "1.0.15"
    config = (cargo / "config.toml").read_text()
    assert 'replace-with = "peterbot-local"' in config
    assert f'local-registry = "{cargo / "registry"}"' in config
    stage_dependency({"registry": "cratesio", "name": "itoa", "version": "1.0.15"},
                     tmp_path / "ws", cache_dir=cache, cargo_home=cargo)
    assert config.count("peterbot-local") == 2  # idempotent append
    other = tmp_path / "cargo2"
    other.mkdir()
    (other / "config.toml").write_text('[source.crates-io]\nreplace-with = "mine"\n')
    stage_dependency({"registry": "cratesio", "name": "ryu", "version": "1.0.20"},
                     tmp_path / "ws2", cache_dir=cache, cargo_home=other)
    assert "peterbot-local" not in (other / "config.toml").read_text()  # task owns its config


def test_worker_falls_back_to_broker_for_unpinned_release(tmp_path):
    broker = fake_broker(FakeResponse(WHEEL_BYTES, broker_headers()))
    result = stage_dependency({"registry": "pypi", "name": "distro", "version": "1.9.0"},
                              tmp_path / "ws", broker=broker, cache_dir=tmp_path / "empty",
                              cargo_home=tmp_path / "cargo")
    staged = tmp_path / "ws" / "deps" / "wheels" / "distro_1.9.0-py3-none-any.whl"
    assert staged.read_bytes() == WHEEL_BYTES
    assert result["source"] == "pypi" and result["sha256"] == WHEEL_SHA
    request = broker.opener.requests[0]
    assert request.full_url == "http://gateway:8770/package"
    assert request.get_header("Authorization") == "Bearer " + "t" * 32


def test_worker_pinned_cache_miss_pins_inventory_not_broker_headers(tmp_path):
    tampered = b"T" * len(WHEEL_BYTES)
    headers = broker_headers(name="six", version="1.17.0")
    headers["X-Peterbot-Sha256"] = hashlib.sha256(tampered).hexdigest()
    headers["X-Peterbot-Filename"] = "six-1.17.0-py2.py3-none-any.whl"
    broker = fake_broker(FakeResponse(tampered, headers))
    with pytest.raises(ValueError):  # staging re-hashes against the inventory pin
        stage_dependency({"registry": "pypi", "name": "six", "version": "1.17.0"},
                         tmp_path / "ws", broker=broker, cache_dir=tmp_path / "empty",
                         cargo_home=tmp_path / "cargo")
    assert not (tmp_path / "ws" / "deps").exists()


def test_worker_rejects_broker_bytes_without_digest(tmp_path):
    headers = broker_headers(name="evil-pkg")
    headers["X-Peterbot-Sha256"] = "not-a-hash"
    headers["X-Peterbot-Filename"] = "evil_pkg-1.9.0-py3-none-any.whl"
    broker = fake_broker(FakeResponse(WHEEL_BYTES, headers))
    result = stage_dependency({"registry": "pypi", "name": "evil-pkg", "version": "1.9.0"},
                              tmp_path / "ws", broker=broker, cache_dir=tmp_path / "empty",
                              cargo_home=tmp_path / "cargo")
    assert result == {"status": "unavailable", "code": "hash_mismatch"}


def test_worker_rejects_oversized_broker_body(tmp_path):
    broker = fake_broker(FakeResponse(b"x" * (pa.MAX_PACKAGE_BYTES + 2048), broker_headers()))
    result = stage_dependency({"registry": "pypi", "name": "distro", "version": "1.9.0"},
                              tmp_path / "ws", broker=broker, cache_dir=tmp_path / "empty",
                              cargo_home=tmp_path / "cargo")
    assert result["status"] == "unavailable" and result["code"] == "oversized"


def test_worker_broker_failure_returns_partial_with_cached_versions(tmp_path, monkeypatch):
    cache = synthetic_cache(tmp_path, monkeypatch)
    body = json.dumps({"error": "nope", "code": "provider_unavailable"}).encode()
    error = HTTPError("http://gateway:8770/package", 503, "down", {}, io.BytesIO(body))
    result = stage_dependency({"registry": "pypi", "name": "distro", "version": "1.9.0"},
                              tmp_path / "ws", broker=fake_broker(error=error),
                              cache_dir=cache, cargo_home=tmp_path / "cargo")
    assert result["status"] == "unavailable" and result["code"] == "provider_unavailable"
    assert "pypi/six" in result["cached_versions"] and "cratesio/itoa" in result["cached_versions"]
    error = HTTPError("http://gateway:8770/package", 404, "gone", {},
                      io.BytesIO(b'{"error":"x","code":"no_artifact"}'))
    result = stage_dependency({"registry": "cratesio", "name": "serde", "version": "1.0.0"},
                              tmp_path / "ws", broker=fake_broker(error=error),
                              cache_dir=cache, cargo_home=tmp_path / "cargo")
    assert result["code"] == "no_artifact" and "cratesio/ryu" in result["cached_versions"]
    result = stage_dependency({"registry": "pypi", "name": "distro", "version": "1.9.0"},
                              tmp_path / "ws", broker=fake_broker(error=OSError("refused")),
                              cache_dir=cache, cargo_home=tmp_path / "cargo")
    assert result["code"] == "provider_unavailable" and "pypi/six" in result["cached_versions"]


def test_stage_dependency_rejects_malformed_before_touching_broker(tmp_path):
    broker = fake_broker(FakeResponse(WHEEL_BYTES, {}))
    with pytest.raises(PackageError):
        stage_dependency({"registry": "pypi", "name": "six; rm", "version": "1.17.0"},
                         tmp_path / "ws", broker=broker, cache_dir=tmp_path,
                         cargo_home=tmp_path / "cargo")
    assert broker.opener.requests == []


def test_fetch_dependency_routing_contract():
    from peterbot.hermes_worker import BROKER_TOOLS, NATIVE_TOOLS
    assert "fetch_dependency" in NATIVE_TOOLS and "fetch_dependency" not in BROKER_TOOLS


def test_broker_opener_never_follows_redirects_or_uses_proxy_env(monkeypatch):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class Direct(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"direct")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Direct)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # Set before construction: the default env-reading ProxyHandler must be suppressed.
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
        monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
        broker = Broker("http://gateway:8770", "t" * 32)
        with broker.opener.open(f"http://127.0.0.1:{server.server_port}/", timeout=5) as resp:
            assert resp.read() == b"direct"  # reached the server directly, not via proxy
        redirects = [h for h in broker.opener.handlers if isinstance(h, HTTPRedirectHandler)]
        assert len(redirects) == 1 and type(redirects[0]).__name__ == "_NoRedirect"
        assert redirects[0].redirect_request(None, None, 302, "", None, "https://x") is None
    finally:
        server.shutdown()
        server.server_close()
