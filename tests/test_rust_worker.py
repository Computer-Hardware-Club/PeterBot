"""Rust e-digits fixture contract (PETER-12 acceptance).

Skipped unless a usable cargo is on PATH (developer machines or the worker image).
Execution inside the actual restricted worker container is proven by
deploy/smoke_rust_worker.py; this suite pins the fixture's algorithm contract:
correctness against an independent Python decimal reference, rejection of
unreasonable input, and offline-buildable dependency freedom.

The recipe-consistency tests at the bottom are the drift guards between the VM
deployment files (deploy/vm/*, docs/worker-vm.md, docker/Dockerfile.hermes-worker):
every one of them corresponds to a real inconsistency a reviewer found once.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "edigits"
sys.path.insert(0, str(FIXTURE))

from reference_e import e_digits  # noqa: E402  (independent decimal reference)
from deploy.smoke_rust_worker import REQUIRED_ISOLATION, isolation_report_complete  # noqa: E402

requires_cargo = pytest.mark.skipif(shutil.which("cargo") is None,
                                    reason="requires a cargo toolchain on PATH")


def test_smoke_rejects_missing_or_truncated_isolation_report():
    checks = [{"check": name, "passed": True} for name in sorted(REQUIRED_ISOLATION)]
    assert isolation_report_complete({"passed": True, "checks": checks})
    assert not isolation_report_complete({"passed": False, "checks": []})
    assert not isolation_report_complete({"passed": True, "checks": checks[:-1]})
    assert not isolation_report_complete({"passed": True, "checks": checks + checks[:1]})
    assert not isolation_report_complete({"passed": True, "checks": [{"check": "worker_uid"}]})


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("edigits-build")
    shutil.copytree(FIXTURE, build / "edigits")
    environment = dict(os.environ, CARGO_HOME=str(build / "cargo-home"))
    completed = subprocess.run(
        ["cargo", "build", "--release", "--offline", "--target-dir", str(build / "target")],
        cwd=build / "edigits", env=environment, capture_output=True, timeout=300)
    assert completed.returncode == 0, completed.stderr.decode()[-2000:]
    return build / "target" / "release" / "edigits"


def run(binary, *arguments):
    return subprocess.run([str(binary), *map(str, arguments)], capture_output=True, timeout=180)


@requires_cargo
@pytest.mark.parametrize("digits", [1, 2, 10, 50, 200, 1000])
def test_matches_independent_decimal_reference(binary, digits):
    result = run(binary, digits)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().strip() == e_digits(digits)


@requires_cargo
@pytest.mark.parametrize("argument", ["0", "-1", "10001", "99999999999999999999",
                                      "abc", "50 ", " 50", "1e3", "+10", "0x10"])
def test_unreasonable_or_malformed_input_rejected(binary, argument):
    result = run(binary, argument)
    assert result.returncode == 2, (argument, result.stdout, result.stderr)
    assert b"edigits" in result.stderr or b"usage" in result.stderr


@requires_cargo
def test_missing_or_extra_arguments_rejected(binary):
    assert run(binary).returncode == 2
    assert run(binary, "10", "extra").returncode == 2


@requires_cargo
def test_output_shape_and_known_digits(binary):
    # Known leading digits of e pin the format independent of the reference.
    assert run(binary, 12).stdout.decode().strip() == "2.718281828459"
    for digits in (1, 10, 50):
        text = run(binary, digits).stdout.decode().strip()
        # "2." prefix + exactly `digits` fractional digits: total length 1+1+digits.
        assert text.startswith("2.") and len(text) == 2 + digits, text


# ---------------------------------------------------------------------------
# Recipe consistency (VM deployment slice). These guard cross-file invariants:
# each mirrors a drift class that already occurred during review. They read
# files only — no docker, no network — so they always run.
# ---------------------------------------------------------------------------

WORKER_DOCKERFILE = (ROOT / "docker/Dockerfile.hermes-worker").read_text()
FIREWALL = (ROOT / "deploy/vm/peterbot-vm-firewall.sh").read_text()
FIREWALL_UNIT = (ROOT / "deploy/vm/peterbot-vm-firewall.service").read_text()
VERIFY = (ROOT / "deploy/vm/peterbot-worker-vm-verify.sh").read_text()
PROVISION = (ROOT / "deploy/vm/peterbot-worker-vm-provision.sh").read_text()
SETUP = (ROOT / "deploy/vm/peterbot-worker-vm-setup.sh").read_text()
TRANSFER = (ROOT / "deploy/vm/peterbot-vm-transfer.sh").read_text()
GUEST_COMPOSE = (ROOT / "deploy/vm/compose.hermes-vm.yml").read_text()
DOMAIN_XML = (ROOT / "deploy/vm/peterbot-worker-vm-domain.xml").read_text()
DOC = (ROOT / "docs/worker-vm.md").read_text()


def test_rust_tarballs_are_hash_verified_before_extraction():
    """Every downloaded tarball passes `sha256sum -c` before `tar -xzf`."""
    block = re.search(r"for item in rustc:.*?done; \\", WORKER_DOCKERFILE, re.S)
    assert block, "rust install loop not found in Dockerfile"
    body = block.group(0)
    download = body.index("curl ")
    check = body.index("sha256sum -c -")
    extract = body.index("tar -xzf")
    assert download < check < extract, "verification must sit between download and extract"


def test_rust_dist_date_consistent_across_dockerfile_and_docs():
    dockerfile_date = re.search(r"ARG RUST_DIST_DATE=([\d-]+)", WORKER_DOCKERFILE).group(1)
    assert f"dist/{dockerfile_date}" in WORKER_DOCKERFILE
    assert dockerfile_date in DOC, "worker-vm.md cites a different Rust dist date"
    # And not a stale sibling date that a previous edit left behind.
    for other in re.findall(r"dist/(\d{4}-\d{2}-\d{2})", DOC):
        assert other == dockerfile_date, f"doc cites dist/{other}, Dockerfile pins {dockerfile_date}"


def test_base_image_pinning_consistent_between_provision_and_docs():
    url_date = re.search(r"genericcloud-amd64-(\d{8}-\d{4})\.qcow2", PROVISION).group(1)
    assert re.search(r"BASE_SHA512=[0-9a-f]{128}", PROVISION)
    assert url_date in DOC, "worker-vm.md cites a different base build than provision.sh pins"


def test_broker_publish_is_host_address_only_and_consistent():
    broker = re.search(r'BROKER_REAL=([\d.]+):(\d+)', FIREWALL).group(1, 2)
    assert broker == ("192.168.241.1", "8770"), \
        "broker target must be the host-only virbr-ctl publish, matching the overlay"
    overlay = (ROOT / "deploy/vm/compose.hermes-vm-gateway.yml").read_text()
    published = re.findall(r"^\s*- '([^']+)'", overlay, re.M)
    assert published == ["192.168.241.1:8770:8770"], \
        f"overlay must publish the broker on the host-only address ONLY, got {published}"
    base = (ROOT / "compose.hermes.yml").read_text()
    assert "8770:8770" not in base, "base compose must not publish the broker without the VM overlay"
    # FORWARD/POSTROUTING match POST-DNAT state: no rule may match the alias.
    for line in FIREWALL.splitlines():
        if line.startswith("$IPT -w -A PETERBOT-WORKER-FWD") or "POSTROUTING" in line:
            assert "192.168.240.2" not in line, \
                "post-DNAT chains must match 192.168.241.1, not the pre-DNAT alias"


def test_worker_subnet_bridge_and_setup_stay_identical():
    subnet = re.search(r"WORKER_SUBNET=([\d./]+)", FIREWALL).group(1)
    bridge = re.search(r'WORKER_BRIDGE=(\S+)', FIREWALL).group(1)
    assert subnet == "192.168.240.0/24"
    # The bridge is created EXTERNALLY by setup (compose references it by name);
    # firewall and verify must use the identical name/subnet.
    assert f"--subnet {subnet}" in SETUP
    assert f"com.docker.network.bridge.name={bridge}" in SETUP
    assert f"ip link show {bridge}" in VERIFY
    # Worker net is joined via worker_args (--network), not a compose section;
    # compose's own network must be the external pre-created control bridge.
    assert "PETERBOT_WORKER_NETWORK: peterbot_workers" in GUEST_COMPOSE
    assert "external: true" in GUEST_COMPOSE
    # ICC off, no hidden masquerade, and --internal on the worker net only;
    # the control net must stay routable for the runner's published port.
    worker_net = SETUP[SETUP.index("name=pbworkers"):SETUP.index("peterbot_workers 2>/dev/null")]
    control_net = SETUP[SETUP.index("name=ctl0"):SETUP.index("peterbot_control 2>/dev/null")]
    for option in ("enable_icc=false", "enable_ip_masquerade=false"):
        assert option in worker_net
        assert option not in control_net
    assert "--internal" in worker_net
    assert "--internal" not in control_net, "published runner port breaks on internal networks"
    for option in ("enable_icc", "enable_ip_masquerade"):
        assert option in VERIFY


def test_setup_never_enables_nonlocal_bind():
    # Binding arbitrary source addresses would defeat the address-pinned wall;
    # every publish uses a real interface address, so the sysctl must stay unset
    # (setup may MENTION why in a comment, but never enable it).
    assert "net.ipv4.ip_nonlocal_bind = 1" not in SETUP
    assert not re.search(r"sysctl\s+-\w+\s+net\.ipv4\.ip_nonlocal_bind", SETUP)
    assert "ip_nonlocal_bind" in VERIFY  # verify asserts it is 0


def test_firewall_precreates_bridge_before_docker():
    assert "Before=docker.service" in FIREWALL_UNIT, \
        "egress wall must precede docker rule creation on boot"
    # A reboot with no worker yet leaves no docker-created bridge; the wall
    # must create the bare bridge itself or the broker alias has no ARP owner.
    assert 'ip link add name "$WORKER_BRIDGE" type bridge' in FIREWALL
    assert 'ip addr add 192.168.240.2/32' in FIREWALL


def test_runner_probe_addresses_match_compose_bind():
    bound = re.search(r"ports:.*?\n\s*- '([\d.]+):(\d+):\d+'", GUEST_COMPOSE, re.S)
    host, port = bound.group(1), bound.group(2)
    assert host == "192.168.241.2", "runner must publish on the control vNIC only"
    assert f"http://{host}:{port}/health" in VERIFY, "verify must probe the bound address"
    assert "127.0.0.1:8780/health" not in VERIFY, \
        "loopback health probe would pass against a drifted 0.0.0.0 bind"
    assert host in DOC and f"{host}:{port}" in DOC, "doc must cite the bound control address"


def test_verify_container_name_matches_compose():
    name = re.search(r"container_name: (\S+)", GUEST_COMPOSE).group(1)
    assert f"docker inspect {name}" in VERIFY


def test_domain_template_pins_both_macs_before_source():
    # libvirt RNG: <mac> must precede <source>; both vNICs pinned for cloud-init
    # MAC matching. The provision script renders both tokens.
    for token in ("@MAC_NAT@", "@MAC@"):
        assert token in DOMAIN_XML
        assert token in PROVISION, f"provision renderer no longer substitutes {token}"
    nat = DOMAIN_XML[DOMAIN_XML.index("network='default'") - 200:
                     DOMAIN_XML.index("network='default'")]
    assert "@MAC_NAT@" in nat, "NAT interface must pin its MAC before <source>"


def test_transfer_stages_exactly_what_the_guest_smoke_imports():
    staged = TRANSFER[TRANSFER.index("tar -C"):]
    # smoke imports peterbot.sandbox_runner (needs __init__), the isolation
    # checker, the Rust fixture, and the Hermes runtime fixture it re-runs.
    for path in ("peterbot/__init__.py", "peterbot/sandbox_runner.py",
                 "deploy/smoke_rust_worker.py", "deploy/check_hermes_isolation.py",
                 "tests/fixtures/edigits", "tests/test_hermes_runtime_integration.py"):
        assert path in staged, f"guest smoke needs {path} staged"
    assert "--image" in DOC and "smoke_rust_worker.py" in DOC


def test_setup_asserts_cloud_init_networking_not_defers_it():
    # Static addressing is provisioned by the seed; setup must FAIL LOUDLY when
    # it is absent, never print instructions for hand-patching a disposable guest.
    assert "FATAL" in SETUP and "192\\.168\\.241\\.2" in SETUP
    assert "apt-get update" in SETUP
    assert "br_netfilter" in SETUP and "modules-load.d" in SETUP


def test_seed_networking_is_a_separate_file_not_user_data():
    # NoCloud contract: early networking lives in a root-level network-config,
    # NOT in user-data (cloud-init ignores `network:` there). The seed builder
    # must emit one and pass it as its own file.
    assert "network-config" in PROVISION
    assert "-N " in PROVISION, "cloud-localds must receive network-config via -N"
    user_data = PROVISION[PROVISION.index("echo '#cloud-config'"):PROVISION.index("} > \"$SEED_DIR/user-data\"")]
    assert "network:" not in user_data, "user-data must not fake early networking"
    assert "macaddress" in PROVISION and "192.168.241.2/24" in PROVISION, \
        "network-config must match the control vNIC by pinned MAC"
    # The mkisofs fallback must also carry network-config at the volume root.
    iso_line = PROVISION[PROVISION.index("-volid cidata"):]
    assert "network-config" in iso_line[:200], "ISO fallback must include root network-config"


def test_provision_runs_unprivileged_and_never_sudo():
    # P910 grants no root SSH/passwordless sudo; libvirtd does the privileged
    # work for the operator user (libvirt/kvm/docker groups). A prose mention
    # of "no sudo" is fine; a sudo INVOCATION is not.
    assert not re.search(r"(^|[;&|]\s*)sudo\s", PROVISION, re.M)
    assert "qemu:///system" in PROVISION
    assert "/mnt/NVME/docker/appdata/peterbot/vm" in PROVISION


def test_smoke_skips_cannot_silently_pass():
    smoke = (ROOT / "deploy/smoke_rust_worker.py").read_text()
    # finish() must distinguish PASS/FAIL/SKIP and fail on undeclared skips —
    # a synthesized "passed": True for the broker probe is the bug class here.
    assert '"SKIP"' in smoke
    assert "undeclared" in smoke, "skips must fail the run unless declared"
    assert '"passed": True' not in smoke, "no check may synthesize a pass"
    assert "--expect-no-gateway" in smoke and "--runner-probe" in smoke


# Opt-in (~1 min): proves the pinned-hash enforcement in the real Dockerfile
# fails closed. Run with PETERBOT_DOCKER_BUILD_TESTS=1 where docker works.
requires_docker_build = pytest.mark.skipif(
    not (os.environ.get("PETERBOT_DOCKER_BUILD_TESTS") and shutil.which("docker")),
    reason="set PETERBOT_DOCKER_BUILD_TESTS=1 with docker available")


@requires_docker_build
def test_corrupted_rust_hash_fails_worker_build(tmp_path):
    # Corrupt BOTH arch variants: the build host's own arch decides which ARG
    # the RUN consults, and corrupting only the other one would silently pass.
    text = WORKER_DOCKERFILE
    for arch in ("AMD64", "ARM64"):
        old = re.search(rf"ARG RUSTC_SHA256_{arch}=([0-9a-f]{{4}})", text)
        assert old, f"anchor drifted for {arch}; test would silently pass"
        text = text.replace(old.group(0), f"ARG RUSTC_SHA256_{arch}=0000", 1)
    broken = tmp_path / "Dockerfile"
    broken.write_text(text)
    context = tmp_path / "ctx"
    context.mkdir()
    shutil.copytree(ROOT / "peterbot", context / "peterbot")
    shutil.copy(ROOT / "requirements-hermes.txt", context / "requirements-hermes.txt")
    completed = subprocess.run(
        ["docker", "build", "--network", "host", "-f", str(broken), "-t",
         "peterbot-hash-probe:bad", str(context)],
        capture_output=True, timeout=600,
        env=dict(os.environ, DOCKER_BUILDKIT="1"))
    log = (completed.stdout + completed.stderr).decode(errors="replace")
    assert completed.returncode != 0, "build must fail on a corrupted toolchain hash"
    assert "FAILED" in log and "sha256sum" in log, \
        "failure must be the checksum gate, not a download error"
