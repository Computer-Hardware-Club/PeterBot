#!/usr/bin/env python3
"""Prove pinned wheel and crate use inside one real restricted Peter worker.

Run on the VM host after loading the candidate image. This script uses the
production ``sandbox_runner.worker_args`` settings, the guest-local Docker
socket, and one disposable worker. It never starts a Discord gateway or sends
a model request. Broker fallback is tested separately after gateway cutover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peterbot import sandbox_runner as sr  # noqa: E402


def docker(*args: str, input_data: str | None = None, timeout: int = 120) -> str:
    result = subprocess.run(["docker", *args], input=input_data, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"docker {args[0]} failed ({result.returncode}): "
                           + (result.stderr or result.stdout)[-500:])
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--network", default="peterbot_workers")
    args = parser.parse_args()
    existing = docker("ps", "-q", "--filter", "label=io.peterbot.worker=hermes")
    if existing:
        raise RuntimeError("Refusing smoke while another worker is active")

    settings = sr.Settings("smoke-" + "x" * 30, args.image, args.network,
                           resource_profile="build")
    name = "peterbot-package-smoke-" + uuid.uuid4().hex[:12]
    checks: list[str] = []

    def passed(label: str) -> None:
        checks.append(label)
        print("PASS " + label, flush=True)

    def inside(*command: str, input_data: str | None = None, timeout: int = 120) -> str:
        flags = ("exec", "-i", "--user", "10000:10000", name) if input_data is not None else (
            "exec", "--user", "10000:10000", name)
        return docker(*flags, *command, input_data=input_data, timeout=timeout)

    created = False
    try:
        docker(*sr.worker_args(settings, name))
        created = True
        wheel = json.loads(inside("python3", "-I", "-c",
            "import json; from pathlib import Path; "
            "from peterbot.hermes_worker import stage_dependency; "
            "print(json.dumps(stage_dependency({'registry':'pypi','name':'six',"
            "'version':'1.17.0'},Path('/workspace'))))"))
        if wheel.get("status") != "ok" or wheel.get("source") != "image_cache":
            raise RuntimeError("Pinned wheel was not staged from the image cache")
        passed("pinned_wheel_staged")

        inside("python3", "-m", "pip", "install", "--no-index", "--no-deps",
               "--target", "/workspace/libs", wheel["path"], timeout=180)
        inside("python3", "-I", "-c",
               "import sys; sys.path.insert(0,'/workspace/libs'); import six; "
               "assert six.__version__ == '1.17.0'")
        passed("wheel_installed_and_imported_offline")

        crate = json.loads(inside("python3", "-I", "-c",
            "import json; from pathlib import Path; "
            "from peterbot.hermes_worker import stage_dependency; "
            "print(json.dumps(stage_dependency({'registry':'cratesio','name':'itoa',"
            "'version':'1.0.15'},Path('/workspace'))))"))
        if crate.get("status") != "ok" or crate.get("source") != "image_cache":
            raise RuntimeError("Pinned crate was not staged from the image cache")
        passed("pinned_crate_staged")

        inside("sh", "-c", "mkdir -p /workspace/crate/src && cat > /workspace/crate/Cargo.toml",
               input_data='[package]\nname="peter_package_smoke"\nversion="0.1.0"\n'
                          'edition="2021"\n[dependencies]\nitoa="=1.0.15"\n')
        inside("sh", "-c", "cat > /workspace/crate/src/main.rs",
               input_data='fn main() { let mut b = itoa::Buffer::new(); '
                          'print!("{}", b.format(42)); }\n')
        inside("cargo", "build", "--offline", "--manifest-path",
               "/workspace/crate/Cargo.toml", "--target-dir", "/workspace/crate/target",
               timeout=300)
        if inside("/workspace/crate/target/debug/peter_package_smoke") != "42":
            raise RuntimeError("Offline Rust crate produced the wrong result")
        passed("crate_compiled_and_ran_offline")

        unavailable = json.loads(inside("python3", "-I", "-c",
            "import json; from pathlib import Path; "
            "from peterbot.hermes_worker import stage_dependency; "
            "print(json.dumps(stage_dependency({'registry':'pypi','name':'not-pinned',"
            "'version':'1.0.0'},Path('/workspace'))))"))
        if unavailable.get("status") != "unavailable" or not unavailable.get("cached_versions"):
            raise RuntimeError("Unavailable dependency lost the cached-version fallback")
        passed("unpinned_request_degrades_cleanly")

        inside("python3", "-I", "-c",
               "from pathlib import Path; p=Path('/opt/peterbot/deps/manifest.json'); "
               "assert p.is_file(); assert not __import__('os').access(p, __import__('os').W_OK)")
        passed("image_cache_read_only")
        print(json.dumps({"pass": len(checks), "fail": 0}), flush=True)
        return 0
    finally:
        if created:
            docker("rm", "--force", name, timeout=30)


if __name__ == "__main__":
    raise SystemExit(main())
