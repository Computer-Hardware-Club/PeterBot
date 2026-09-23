# PETER-13: Controlled research and dependency access

The trusted gateway (`peterbot/hermes_gateway.py`) is the only component with a
network path to package registries. The disposable VM worker
(`peterbot/hermes_worker.py`, `docker/Dockerfile.hermes-worker`) has no egress:
it never contacts a registry, proxy, or DNS resolver directly, and it never
holds registry, Docker, or host credentials.

## Supported workflow

A worker task acquires exactly pinned public releases with the
`fetch_dependency` tool (`registry` is `pypi` or `cratesio`; exact `name` +
`version`). The result is a JSON object with `path`, `sha256`, `size`,
`source`, and an install hint — never file bytes and never base64 in model
context.

1. Image cache first. The worker image carries an immutable read-only cache at
   `/opt/peterbot/deps` (wheels + `.crate` files + `manifest.json`) built from
   `peterbot/package_access.py PACKAGE_INVENTORY`. Every byte is re-hashed at
   build time; the build fails closed on mismatch. A pinned hit stages locally
   with zero network.
2. Authenticated broker fallback. A cache miss POSTs one request to the
   gateway's `/package` route with the task's capability token. The gateway
   broker fetches the exact release from PyPI (`pypi.org/pypi/{name}/{version}/json`
   then `files.pythonhosted.org`) or crates.io (`index.crates.io/{index_dir}`
   then `static.crates.io/crates/{name}/{name}-{version}.crate`), verifies the
   SHA256 digest, and returns raw bytes plus `X-Peterbot-Sha256/-Filename/-Size/-Source/-Index-Line`
   headers. The worker re-hashes what it receives before writing it.

Python: stage installs into the workspace with
`pip install --no-index --target /workspace/libs <path>.whl` (pure-Python
`py*-none-any` wheels only — the broker rejects platform/ABI wheels and
sdists). Rust: staged crates land in a cargo `local-registry` under
`$CARGO_HOME/registry`, with `replace-with = "peterbot-local"` written into the
task's own `$CARGO_HOME/config.toml`; build with `cargo build --offline`
(`CARGO_NET_OFFLINE=true` is baked into the image). Unpinned crate sets still
need every transitive dependency fetched explicitly — `cargo --offline` fails
closed when something is missing.

## Every-hop validation

`peterbot/package_access.py` validates at each hop, and re-validates after
every redirect:

- Registry host allowlist: `pypi.org`, `files.pythonhosted.org`,
  `index.crates.io`, `static.crates.io`. HTTPS only; no userinfo, query,
  fragment, percent-escapes, or `..` in paths; default 443 port only.
- Package name/version: strict ASCII patterns (IDN, control characters,
  traversal, and shell metacharacters die at validation).
- Artifact: declared size checked before download, streamed body hard-capped,
  declared content type checked for metadata, SHA256 from registry metadata
  verified over the received bytes (PyPI `digests.sha256`, crate index `cksum`),
  zip/gzip container magic checked, yanked releases rejected.
- DNS/IP: a custom `aiohttp` resolver validates every `getaddrinfo` answer and
  hands only validated addresses to the transport, with DNS cache disabled and
  a fresh connector per redirect hop (kills rebinding between check and dial).
  Blocked: non-global, loopback, link-local, multicast, reserved, unspecified,
  CGNAT `100.64.0.0/10`, IPv6-mapped/compat, Teredo, 6to4, and IPv4-mapped
  loopback/private equivalents; `169.254.169.254` and cloud metadata ranges
  are included. Literal-IP URLs are rejected outright. Redirect targets get a
  preflight resolve plus the same connect-time validation. Redirects are
  followed manually, max 2 hops; a redirect off-allowlist, private, or
  credential-bearing is rejected as `poisoned_metadata`.
- Transport hygiene: `trust_env=False` and an empty `ProxyHandler` (worker-side
  opener) ignore `*_proxy` environment variables; `DummyCookieJar`; no
  automatic decompression.

## Budgets and failure shape

- Per artifact: 32 MiB (`MAX_PACKAGE_BYTES`). Per task broker quota: 64 MiB
  (`PACKAGE_TASK_BYTES`), accumulated on the capability and enforced at the
  gateway before the broker runs.
- Every package fetch shares the task deadline: the gateway refuses (429) when
  fewer than 10 seconds remain and clamps the broker call to the remaining
  budget.
- Errors are machine-readable JSON `{error, code}` with a stable code
  (`invalid_request`, `invalid_registry`, `invalid_name`, `invalid_version`,
  `blocked_address`, `poisoned_metadata`, `hash_mismatch`, `oversized`,
  `over_quota`, `yanked`, `no_artifact`, `cache_corrupted`,
  `provider_unavailable`, `timeout`). An unavailable registry never destroys a
  task: the worker receives an `unavailable` result including which versions
  are in the image cache, so the model can produce a partial answer.
- Retrieved registry metadata is dependency data only. It never authorizes
  club facts, style changes, announcements, or any broader egress. Package
  contents are untrusted and execute only inside the disposable sandbox.

## Proven acquisitions

Both representative artifacts are verified in `tests/test_package_access.py`
against local protocol doubles, and the real pinned digests ship in
`PACKAGE_INVENTORY`:

- Python wheel: `six-1.17.0-py2.py3-none-any.whl`, sha256
  `4721f391…c3274`, 11050 bytes — acquired hash-verified, staged into
  `/workspace/deps/wheels`, installed with `pip --no-index`.
- Rust crate: `itoa-1.0.15.crate`, sha256
  `4a5f13b8…928e2c` (index `cksum`), 11231 bytes — acquired hash-verified with
  its index line, staged into a cargo local-registry, built with
  `cargo build --offline`.

Real-image proof (executed 2026-09-23, `docker build -f
docker/Dockerfile.hermes-worker` then a throwaway script launching the image
with production `sandbox_runner.worker_args` and `--network none`): 10/10
checks passed in the actual restricted container — wheel staged hash-verified,
installed `pip --no-index`, imported; crate staged into `$CARGO_HOME/registry`
(env pinned in the image), `cargo build --offline` compiled and the binary
ran; unpinned and unreachable-broker requests degraded to machine-readable
`unavailable` results with cached-version lists and zero egress attempts;
direct sockets to 1.1.1.1:443, 8.8.8.8:53 and pypi.org all failed;
`/opt/peterbot/deps` is read-only for the worker user.
