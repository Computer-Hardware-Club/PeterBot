"""Controlled dependency acquisition: pure validation core plus the gateway broker.

Every hop (registry metadata and artifact download) resolves through a fresh,
cache-free connector whose resolver rejects any non-public DNS answer before the
transport ever connects, so neither a poisoned redirect nor a DNS rebinding can
aim a fetch at a loopback, LAN, tailnet, or metadata address. Bytes are
hash-verified against the registry's own digest before they are ever handed to a
worker, and the pinned image inventory is the default source. The worker side
(peterbot/hermes_worker.py) imports only the validation and inventory helpers
from this module and never needs aiohttp.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

# --- budgets ---------------------------------------------------------------
MAX_PACKAGE_BYTES = 32 * 1024 * 1024      # hard cap per acquired artifact
PACKAGE_TASK_BYTES = 64 * 1024 * 1024     # per-task broker quota
MAX_METADATA_BYTES = 4 * 1024 * 1024      # PyPI JSON / crates index body cap

REGISTRIES = ("pypi", "cratesio")

PYPI_METADATA_HOSTS = frozenset({"pypi.org"})
PYPI_FILE_HOSTS = frozenset({"files.pythonhosted.org"})
CARGO_HOSTS = frozenset({"index.crates.io", "static.crates.io"})
ALL_PACKAGE_HOSTS = PYPI_METADATA_HOSTS | PYPI_FILE_HOSTS | CARGO_HOSTS

# --- strict name/version shapes (ASCII; IDN and traversal die here) --------
_PYPI_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_PYPI_VERSION_RE = re.compile(r"[A-Za-z0-9._+!-]{1,64}")
_PYPI_WHEEL_PART_RE = re.compile(r"[A-Za-z0-9._+!-]+")
_CRATE_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_CRATE_VERSION_RE = re.compile(r"[0-9A-Za-z.\-+]{1,64}")
FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,159}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")

# Pure-Python wheels only: a platform tag implies native binary payloads that
# the broker never ships; source/binary distributions are out of scope.
# Compressed python tags ("py2.py3") are matched element by element.
_WHEEL_TAG_RE = re.compile(r"-(?:py[0-9]+(?:\.py[0-9]+)*)-none-any\.whl$")

IMAGE_DEP_CACHE = "/opt/peterbot/deps"

# --- pinned image inventory (Dockerfile mirrors these exact values) --------
PACKAGE_INVENTORY: tuple[dict, ...] = (
    {
        "registry": 'pypi', "name": 'six', "version": '1.17.0',
        "filename": 'six-1.17.0-py2.py3-none-any.whl',
        "sha256": '4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274',
        "size": 11050,
        "url": 'https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'python-dateutil', "version": '2.9.0.post0',
        "filename": 'python_dateutil-2.9.0.post0-py2.py3-none-any.whl',
        "sha256": 'a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427',
        "size": 229892,
        "url": 'https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'certifi', "version": '2025.10.5',
        "filename": 'certifi-2025.10.5-py3-none-any.whl',
        "sha256": '0f212c2744a9bb6de0c56639a6f68afe01ecd92d91f14ae897c4fe7bbeeef0de',
        "size": 163286,
        "url": 'https://files.pythonhosted.org/packages/e4/37/af0d2ef3967ac0d6113837b44a4f0bfe1328c2b9763bd5b1744520e5cfed/certifi-2025.10.5-py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'urllib3', "version": '2.5.0',
        "filename": 'urllib3-2.5.0-py3-none-any.whl',
        "sha256": 'e6b01673c0fa6a13e374b50871808eb3bf7046c4b125b216f6bf1cc604cff0dc',
        "size": 129795,
        "url": 'https://files.pythonhosted.org/packages/a7/c2/fe1e52489ae3122415c51f387e221dd0773709bad6c6cdaa599e8a2c5185/urllib3-2.5.0-py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'idna', "version": '3.10',
        "filename": 'idna-3.10-py3-none-any.whl',
        "sha256": '946d195a0d259cbba61165e88e65941f16e9b36ea6ddb97f00452bae8b1287d3',
        "size": 70442,
        "url": 'https://files.pythonhosted.org/packages/76/c6/c88e154df9c4e1a2a66ccf0005a88dfb2650c1dffb6f5ce603dfbd452ce3/idna-3.10-py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'charset-normalizer', "version": '3.4.3',
        "filename": 'charset_normalizer-3.4.3-py3-none-any.whl',
        "sha256": 'ce571ab16d890d23b5c278547ba694193a45011ff86a9162a71307ed9f86759a',
        "size": 53175,
        "url": 'https://files.pythonhosted.org/packages/8a/1f/f041989e93b001bc4e44bb1669ccdcf54d3f00e628229a85b08d330615c5/charset_normalizer-3.4.3-py3-none-any.whl',
    },
    {
        "registry": 'pypi', "name": 'requests', "version": '2.32.5',
        "filename": 'requests-2.32.5-py3-none-any.whl',
        "sha256": '2462f94637a34fd532264295e186976db0f5d453d1cdd31473c85a6a161affb6',
        "size": 64738,
        "url": 'https://files.pythonhosted.org/packages/1e/db/4254e3eabe8020b458f1a747140d32277ec7a271daf1d235b70dc0b4e6e3/requests-2.32.5-py3-none-any.whl',
    },
    {
        "registry": 'cratesio', "name": 'itoa', "version": '1.0.15',
        "filename": 'itoa-1.0.15.crate',
        "sha256": '4a5f13b858c8d314ee3e8f639011f7ccefe71f97f96e50151fb991f267928e2c',
        "size": 11231,
        "url": 'https://static.crates.io/crates/itoa/itoa-1.0.15.crate',
        "index_line": '{"name":"itoa","vers":"1.0.15","deps":[{"name":"no-panic","req":"^0.1","features":[],"optional":true,"default_features":true,"target":null,"kind":"normal"}],"cksum":"4a5f13b858c8d314ee3e8f639011f7ccefe71f97f96e50151fb991f267928e2c","features":{},"yanked":false,"rust_version":"1.36","pubtime":"2025-03-03T23:42:45Z"}',
    },
    {
        "registry": 'cratesio', "name": 'ryu', "version": '1.0.20',
        "filename": 'ryu-1.0.20.crate',
        "sha256": '28d3b2b1366ec20994f1fd18c3c594f05c5dd4bc44d8bb0c1c632c8d6829481f',
        "size": 48738,
        "url": 'https://static.crates.io/crates/ryu/ryu-1.0.20.crate',
        "index_line": '{"name":"ryu","vers":"1.0.20","deps":[{"name":"no-panic","req":"^0.1","features":[],"optional":true,"default_features":true,"target":null,"kind":"normal"},{"name":"num_cpus","req":"^1.8","features":[],"optional":false,"default_features":true,"target":null,"kind":"dev"},{"name":"rand","req":"^0.9","features":[],"optional":false,"default_features":true,"target":null,"kind":"dev"},{"name":"rand_xorshift","req":"^0.4","features":[],"optional":false,"default_features":true,"target":null,"kind":"dev"}],"cksum":"28d3b2b1366ec20994f1fd18c3c594f05c5dd4bc44d8bb0c1c632c8d6829481f","features":{"small":[]},"yanked":false,"rust_version":"1.36","pubtime":"2025-03-04T00:13:50Z"}',
    },
)


class PackageError(ValueError):
    """Rejected package request. `code` is stable machine vocabulary."""

    def __init__(self, code: str, message: str = "", status: int = 400):
        super().__init__(message or code)
        self.code = code
        self.status = status


# --- address / URL validation ------------------------------------------------
def _address_blocked(ip: ipaddress.IPvAddress) -> bool:
    if (not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_loopback
            or ip.is_link_local or ip.is_unspecified):
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.sixtofour is not None or ip.teredo is not None or ip.ipv4_mapped is not None:
            return True
    return False


def _url_public_host(host: str) -> None:
    """Literal-IP URLs are always rejected; hostnames resolve via the resolver."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return
    raise PackageError("blocked_address", "registry URL must not be a raw address")


class _ValidatedResolver:
    """Hands only validated DNS answers directly to the connecting transport."""

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_UNSPEC) -> list[dict]:
        import asyncio

        try:
            answers = await asyncio.get_running_loop().getaddrinfo(
                host, port, family=family, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise PackageError("provider_unavailable", "registry DNS failed", 503) from exc
        if not answers or len(answers) > 64:
            raise PackageError("provider_unavailable", "registry DNS returned nothing", 503)
        resolved = []
        for answer_family, _, proto, _, sockaddr in answers:
            text = sockaddr[0].split("%", 1)[0]
            try:
                ip = ipaddress.ip_address(text)
            except ValueError as exc:
                raise PackageError("blocked_address", "malformed DNS answer") from exc
            if _address_blocked(ip):
                raise PackageError("blocked_address", "registry resolved to a non-public address")
            resolved.append({
                "hostname": host, "host": text, "port": port, "family": answer_family,
                "proto": proto, "flags": socket.AI_NUMERICHOST | socket.AI_NUMERICSERV,
            })
        return resolved

    async def close(self) -> None:
        pass


def _safe_url(url: str, hosts: frozenset[str]) -> str:
    """Parse a registry URL: https, no credentials, allowlisted host, ASCII path."""
    if not isinstance(url, str) or len(url) > 600:
        raise PackageError("poisoned_metadata", "registry URL rejected")
    if any(unicodedata.category(char).startswith("C") for char in url) or "\\" in url \
            or any(char.isspace() for char in url) or not all(32 < ord(char) < 127 for char in url):
        raise PackageError("poisoned_metadata", "registry URL rejected")
    try:
        parts = urlsplit(url)
        port = parts.port  # validates malformed/out-of-range ports
    except ValueError as exc:
        raise PackageError("poisoned_metadata", "registry URL malformed") from exc
    host = (parts.hostname or "").rstrip(".").lower()
    if (parts.scheme != "https" or host not in hosts or parts.username or parts.password
            or port not in (None, 443) or parts.query or parts.fragment
            or "%" in parts.path or ".." in parts.path
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)):
        raise PackageError("poisoned_metadata", "registry URL rejected")
    _url_public_host(host)
    return url


# --- pure request validation ---------------------------------------------------
def validate_package_request(registry: object, name: object, version: object) -> tuple[str, str, str]:
    """Normalize and hard-validate (registry, name, version). Raises PackageError."""
    if not isinstance(registry, str) or registry not in REGISTRIES:
        raise PackageError("invalid_registry")
    if not isinstance(name, str) or not isinstance(version, str):
        raise PackageError("invalid_name", "name and version must be strings")
    if not name or not version or len(name) > 100 or len(version) > 64:
        raise PackageError("invalid_name", "name/version length rejected")
    if any(unicodedata.category(char).startswith("C") for char in name + version):
        raise PackageError("invalid_name", "control characters in name/version")
    try:
        name.encode("ascii")
        version.encode("ascii")
    except UnicodeError as exc:
        raise PackageError("invalid_name", "name/version must be ASCII") from exc
    if registry == "pypi":
        if not _PYPI_NAME_RE.fullmatch(name):
            raise PackageError("invalid_name")
        if not _PYPI_VERSION_RE.fullmatch(version):
            raise PackageError("invalid_version")
        name = name.lower().replace("_", "-")
    else:
        if not _CRATE_NAME_RE.fullmatch(name):
            raise PackageError("invalid_name")
        if (not _CRATE_VERSION_RE.fullmatch(version) or "://" in version or ".." in version
                or not version[0].isdigit()):
            raise PackageError("invalid_version")
    return registry, name, version


def validate_arguments(arguments: object) -> dict:
    """Validate a worker tool-call payload; returns normalized {registry,name,version}."""
    if not isinstance(arguments, dict) or set(arguments) != {"registry", "name", "version"}:
        raise PackageError("invalid_request", "expected exactly registry/name/version")
    registry, name, version = validate_package_request(
        arguments.get("registry"), arguments.get("name"), arguments.get("version"))
    return {"registry": registry, "name": name, "version": version}


def index_dir(name: str) -> str:
    lowered = name.lower()
    if len(lowered) == 1:
        return f"1/{lowered}"
    if len(lowered) == 2:
        return f"2/{lowered}"
    if len(lowered) == 3:
        return f"3/{lowered[0]}/{lowered}"
    return f"{lowered[:2]}/{lowered[2:4]}/{lowered}"


# --- image-cache inventory ------------------------------------------------------
def image_lookup(registry: str, name: str, version: str) -> dict | None:
    """Return the pinned inventory entry for a validated (registry, name, version)."""
    query = name.lower().replace("_", "-") if registry == "pypi" else name
    for entry in PACKAGE_INVENTORY:
        entry_name = (entry["name"].lower().replace("_", "-")
                      if entry["registry"] == "pypi" else entry["name"])
        if entry["registry"] == registry and entry_name == query and entry["version"] == version:
            return dict(entry)
    return None


def cache_versions(cache_dir: str | Path = IMAGE_DEP_CACHE) -> dict[str, list[str]]:
    """Versions present in an image-style cache dir, keyed by '<registry>/<name>'."""
    out: dict[str, list[str]] = {}
    manifest = Path(cache_dir) / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("packages"), list):
        return out
    for entry in data["packages"]:
        if not isinstance(entry, dict):
            continue
        try:
            registry, name, version = validate_package_request(
                entry.get("registry"), entry.get("name"), entry.get("version"))
        except PackageError:
            continue
        out.setdefault(f"{registry}/{name}", []).append(version)
    return out


@dataclass(frozen=True)
class Acquired:
    filename: str
    data: bytes
    sha256: str
    source: str          # "image_cache" | "pypi" | "cratesio"
    index_line: str = ""


class PackageBroker:
    """Gateway-side artifact broker. Image cache first; exact public releases only."""

    def __init__(self, cache_dir: str | Path, max_bytes: int = MAX_PACKAGE_BYTES):
        self.cache_dir = Path(cache_dir)
        self.max_bytes = min(int(max_bytes), MAX_PACKAGE_BYTES)

    async def serve(self, args: dict, quota: int = PACKAGE_TASK_BYTES) -> Acquired:
        """Acquire one validated artifact; `quota` is the task's remaining byte budget."""
        args = validate_arguments(args)
        if quota <= 0:
            raise PackageError("over_quota", "task dependency quota exhausted", 503)
        budget = min(quota, self.max_bytes)
        entry = image_lookup(args["registry"], args["name"], args["version"])
        if entry is not None:
            data = await _read_cached(self._cache_path(entry), entry, budget)
            return Acquired(entry["filename"], data, entry["sha256"], "image_cache",
                            entry.get("index_line", ""))
        if args["registry"] == "pypi":
            return await self._acquire_pypi(args["name"], args["version"], budget)
        return await self._acquire_crate(args["name"], args["version"], budget)

    # -- image cache -----------------------------------------------------------
    def _cache_path(self, entry: dict) -> Path:
        folder = "wheels" if entry["registry"] == "pypi" else "crates"
        filename = entry["filename"]
        if not FILENAME_RE.fullmatch(filename):
            raise PackageError("poisoned_metadata", "inventory filename rejected")
        return self.cache_dir / folder / filename

    async def cached_versions(self) -> dict[str, list[str]]:
        return await _run(cache_versions, self.cache_dir)

    # -- pypi --------------------------------------------------------------------
    async def _acquire_pypi(self, name: str, version: str, budget: int) -> Acquired:
        filename, file_url, sha256, size = await self._pypi_release(name, version)
        if size > budget:
            raise PackageError("oversized", "artifact exceeds the task byte quota")
        data = await self._fetch(file_url, PYPI_FILE_HOSTS, budget, ())
        if not data.startswith(b"PK\x03\x04"):
            raise PackageError("poisoned_metadata", "artifact is not a zip container")
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            raise PackageError("hash_mismatch", "artifact failed digest verification")
        return Acquired(filename, data, sha256, "pypi")

    async def _pypi_release(self, name: str, version: str) -> tuple[str, str, str, int]:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        _safe_url(url, PYPI_METADATA_HOSTS)
        body = await self._fetch(url, PYPI_METADATA_HOSTS, MAX_METADATA_BYTES,
                                 ("application/json",))
        try:
            doc = json.loads(body)
        except ValueError as exc:
            raise PackageError("poisoned_metadata", "PyPI JSON unparsable") from exc
        if not isinstance(doc, dict):
            raise PackageError("poisoned_metadata", "PyPI JSON shape rejected")
        info = doc.get("info")
        urls = doc.get("urls")
        if not isinstance(info, dict) or not isinstance(urls, list):
            raise PackageError("poisoned_metadata", "PyPI JSON fields rejected")
        declared = info.get("name")
        if not isinstance(declared, str):
            raise PackageError("poisoned_metadata", "PyPI project name malformed")
        if declared.lower().replace("_", "-") != name:
            raise PackageError("poisoned_metadata", "PyPI project name mismatch")
        if not isinstance(info.get("version"), str) or info["version"] != version:
            raise PackageError("poisoned_metadata", "PyPI version mismatch")
        if not isinstance(info.get("yanked"), bool) or info["yanked"]:
            raise PackageError("yanked", "requested release is yanked")
        candidates = []
        for item in urls:
            if not isinstance(item, dict):
                raise PackageError("poisoned_metadata", "PyPI file entry rejected")
            filename = item.get("filename")
            file_url = item.get("url")
            digests = item.get("digests")
            size = item.get("size")
            if not isinstance(filename, str) or not isinstance(file_url, str):
                continue
            if item.get("packagetype") != "bdist_wheel":
                continue
            sha256 = digests.get("sha256") if isinstance(digests, dict) else None
            if not isinstance(sha256, str) or not _HASH_RE.fullmatch(sha256):
                raise PackageError("poisoned_metadata", "PyPI digest malformed")
            if type(size) is not int or size < 0 or size > self.max_bytes:
                raise PackageError("oversized", "artifact exceeds the size budget")
            if not _WHEEL_TAG_RE.search(filename) or filename.count("-") < 4:
                continue  # non-pure or unusual wheel; out of broker scope
            stem = filename[:-4]
            if any(not _PYPI_WHEEL_PART_RE.fullmatch(part) for part in stem.split("-")[:2]):
                continue
            if not stem.replace("-", "_").lower().startswith(name.replace("-", "_").lower()):
                continue
            candidates.append((filename, file_url, sha256, size))
        if len(candidates) != 1:
            raise PackageError("no_artifact", "exactly one pure wheel release is required", 404)
        filename, file_url, sha256, size = candidates[0]
        _safe_url(file_url, PYPI_FILE_HOSTS)
        return filename, file_url, sha256, size

    # -- crates.io -----------------------------------------------------------------
    async def _acquire_crate(self, name: str, version: str, budget: int) -> Acquired:
        index_url = f"https://index.crates.io/{index_dir(name)}"
        _safe_url(index_url, CARGO_HOSTS)
        body = await self._fetch(index_url, CARGO_HOSTS, MAX_METADATA_BYTES,
                                 ("text/plain", "application/json"))
        line = self._parse_index(body, name, version)
        sha256 = line["cksum"]
        size = line.get("size")
        if size is not None and (type(size) is not int or size > budget):
            raise PackageError("oversized", "crate exceeds the size budget")
        file_url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
        _safe_url(file_url, CARGO_HOSTS)
        data = await self._fetch(file_url, CARGO_HOSTS, budget, ())
        if not data.startswith(b"\x1f\x8b"):
            raise PackageError("poisoned_metadata", "crate is not a gzip container")
        if hashlib.sha256(data).hexdigest() != sha256:
            raise PackageError("hash_mismatch", "crate failed cksum verification")
        return Acquired(f"{name}-{version}.crate", data, sha256, "cratesio",
                        json.dumps(line, separators=(",", ":")))

    @staticmethod
    def _parse_index(body: bytes, name: str, version: str) -> dict:
        target = None
        for raw in body.splitlines():
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except ValueError as exc:
                raise PackageError("poisoned_metadata", "crate index line unparsable") from exc
            if not isinstance(entry, dict):
                raise PackageError("poisoned_metadata", "crate index line shape rejected")
            if entry.get("name") != name:
                raise PackageError("poisoned_metadata", "crate index name mismatch")
            if entry.get("vers") != version:
                continue
            if target is not None:
                raise PackageError("poisoned_metadata", "duplicate crate index entry")
            target = entry
        if target is None:
            raise PackageError("no_artifact", "crate version not in index", 404)
        cksum = target.get("cksum")
        if not isinstance(cksum, str) or not _HASH_RE.fullmatch(cksum):
            raise PackageError("poisoned_metadata", "crate index checksum malformed")
        if target.get("yanked") is not False:
            raise PackageError("yanked", "requested crate is yanked")
        deps = target.get("deps")
        if not isinstance(deps, list) or not all(isinstance(d, dict) for d in deps):
            raise PackageError("poisoned_metadata", "crate index deps rejected")
        features = target.get("features")
        if features is not None and not isinstance(features, dict):
            raise PackageError("poisoned_metadata", "crate index features rejected")
        return target

    # -- pinned-IP, cache-free HTTP core ----------------------------------------
    async def _fetch(self, url: str, hosts: frozenset[str], limit: int,
                     content_prefixes: tuple[str, ...]) -> bytes:
        import asyncio

        try:
            async with asyncio.timeout(20):
                return await self._fetch_hops(url, hosts, limit, content_prefixes)
        except asyncio.TimeoutError as exc:
            raise PackageError("provider_unavailable", "registry fetch timed out", 503) from exc

    async def _fetch_hops(self, url: str, hosts: frozenset[str], limit: int,
                          content_prefixes: tuple[str, ...]) -> bytes:
        import aiohttp

        for hop in range(3):
            _safe_url(url, hosts)
            # A fresh connector per hop prevents cached DNS, connection reuse, or
            # cookies from crossing a redirect; the resolver validates every answer.
            connector = aiohttp.TCPConnector(resolver=_ValidatedResolver(), use_dns_cache=False)
            try:
                async with aiohttp.ClientSession(
                        connector=connector, trust_env=False,
                        cookie_jar=aiohttp.DummyCookieJar(), auto_decompress=False,
                        timeout=aiohttp.ClientTimeout(total=20),
                        headers={"User-Agent": "peterbot-package-broker/1",
                                 "Accept-Encoding": "identity"}) as session:
                    async with session.get(url, allow_redirects=False) as response:
                        if response.status in (301, 302, 303, 307, 308):
                            location = response.headers.get("Location", "")
                            if hop == 2 or not location:
                                raise PackageError("poisoned_metadata", "registry redirect rejected")
                            nxt = urljoin(url, location)
                            _safe_url(nxt, hosts)
                            # Preflight the target so a private-host redirect is a
                            # deterministic rejection instead of a connection error.
                            await _preflight(urlsplit(nxt).hostname)
                            url = nxt
                            continue
                        if response.status != 200:
                            raise PackageError("provider_unavailable",
                                               f"registry returned status {response.status}", 503)
                        ctype = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                        if content_prefixes and not ctype.startswith(content_prefixes):
                            raise PackageError("poisoned_metadata",
                                               f"registry content type rejected: {ctype}")
                        declared = response.content_length
                        if declared is not None and declared > limit:
                            raise PackageError("oversized", "registry body exceeds budget")
                        body = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            body.extend(chunk)
                            if len(body) > limit:
                                raise PackageError("oversized", "registry body exceeds budget")
                        return bytes(body)
            except PackageError:
                raise
            except (aiohttp.ClientError, OSError) as exc:
                raise PackageError("provider_unavailable", "registry connection failed", 503) from exc
        raise PackageError("poisoned_metadata", "registry redirect chain rejected")


# --- small async helpers (kept module-level so imports stay lazy) --------------
async def _preflight(host: str) -> None:
    """Resolve+validate a redirect target before the transport will touch it."""
    try:
        await _ValidatedResolver().resolve(host, 443)
    except PackageError:
        raise
    except OSError as exc:
        raise PackageError("provider_unavailable", "redirect target DNS failed", 503) from exc


async def _read_cached(path: Path, entry: dict, budget: int) -> bytes:
    import asyncio

    def read() -> bytes:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PackageError("provider_unavailable", "cached artifact missing", 503) from exc
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise PackageError("cache_corrupted", "cached artifact failed verification", 500)
        return data

    data = await _run(read)
    if len(data) > budget:
        raise PackageError("over_quota", "cached artifact exceeds the task quota", 503)
    return data


async def _run(func, *args):
    import asyncio

    return await asyncio.to_thread(func, *args)
