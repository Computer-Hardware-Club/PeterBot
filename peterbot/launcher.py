from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import AppConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_llama_server_command(config: AppConfig, binary: str = "llama-server") -> list[str]:
    command = [
        binary,
        "--model",
        config.llama_server.model_path or "",
        "--alias",
        config.inference.model,
        "--host",
        config.llama_server.host,
        "--port",
        str(config.llama_server.port),
        "--ctx-size",
        str(config.llama_server.ctx_size),
        "--batch-size",
        str(config.llama_server.batch_size),
        "--parallel",
        str(config.llama_server.parallel),
        "--timeout",
        str(config.inference.timeout_seconds),
    ]
    if config.llama_server.threads > 0:
        command.extend(["--threads", str(config.llama_server.threads)])
    if config.llama_server.n_gpu_layers > 0:
        command.extend(["--n-gpu-layers", str(config.llama_server.n_gpu_layers)])
    if config.llama_server.continuous_batching:
        command.append("--cont-batching")
    if config.llama_server.metrics:
        command.append("--metrics")
    if config.llama_cpp_api_key:
        command.extend(["--api-key", config.llama_cpp_api_key])
    command.extend(config.llama_server.extra_args)
    return command


def wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def resolve_probe_host(host: str) -> str:
    normalized = (host or "").strip()
    if normalized in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized


def terminate_process(process: Optional[subprocess.Popen[bytes]]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()


def _wait_and_forward(bot_process: subprocess.Popen[bytes], server_process: Optional[subprocess.Popen[bytes]]) -> int:
    def handle_signal(signum: int, _frame: object) -> None:
        terminate_process(bot_process)
        terminate_process(server_process)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while True:
            bot_code = bot_process.poll()
            server_code = server_process.poll() if server_process else None

            if bot_code is not None:
                terminate_process(server_process)
                return bot_code

            if server_process and server_code is not None:
                terminate_process(bot_process)
                return server_code

            time.sleep(0.25)
    finally:
        terminate_process(bot_process)
        terminate_process(server_process)
        if server_process is not None:
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
        try:
            bot_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bot_process.kill()


def main() -> int:
    config = AppConfig.load()
    env = os.environ.copy()

    server_process: Optional[subprocess.Popen[bytes]] = None
    if config.llama_server.enabled:
        server_command = build_llama_server_command(config)
        server_process = subprocess.Popen(server_command, cwd=PROJECT_ROOT, env=env)
        ready = wait_for_port(
            resolve_probe_host(config.llama_server.host),
            config.llama_server.port,
            max(60, config.inference.timeout_seconds),
        )
        if not ready:
            terminate_process(server_process)
            return 1

    bot_process = subprocess.Popen([sys.executable, "bot.py"], cwd=PROJECT_ROOT, env=env)
    return _wait_and_forward(bot_process, server_process)


if __name__ == "__main__":
    raise SystemExit(main())
