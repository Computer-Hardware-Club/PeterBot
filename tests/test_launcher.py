import socket
import threading
from pathlib import Path

from peterbot.config import (
    AppConfig,
    BehaviorConfig,
    DiscordConfig,
    InferenceConfig,
    LlamaServerConfig,
    LoggingConfig,
    ModelProfile,
    PathsConfig,
    PersonaConfig,
)
from peterbot.launcher import build_llama_server_command, resolve_probe_host, wait_for_port


def build_bundled_config(tmp_path: Path) -> AppConfig:
    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake", encoding="utf-8")
    return AppConfig(
        discord_token="token",
        llama_cpp_api_key="secret-key",
        persona=PersonaConfig(
            name="Peter",
            system_prompt="You are Peter.",
            model_profile=ModelProfile.QWEN,
        ),
        discord=DiscordConfig(suggestion_channel_id=None),
        inference=InferenceConfig(
            base_url="http://127.0.0.1:8080",
            model="peterbot-gguf",
            timeout_seconds=300,
            max_tokens=512,
            temperature=0.3,
            top_p=None,
            extra_request_body={},
        ),
        llama_server=LlamaServerConfig(
            enabled=True,
            model_path=str(model_path),
            host="0.0.0.0",
            port=8080,
            ctx_size=4096,
            threads=8,
            batch_size=512,
            parallel=2,
            continuous_batching=True,
            n_gpu_layers=16,
            metrics=True,
            extra_args=["--log-format", "text"],
        ),
        paths=PathsConfig(
            data_dir=str(tmp_path / "state"),
            knowledge_file=None,
            channel_profiles_file=None,
            log_file="",
        ),
        logging=LoggingConfig(
            level="INFO",
            user_debug_ids_enabled=True,
            include_traceback_for_warning=False,
        ),
        behavior=BehaviorConfig(),
        config_path=str(tmp_path / "config.json"),
    )


def test_build_llama_server_command_includes_expected_flags(tmp_path: Path) -> None:
    command = build_llama_server_command(build_bundled_config(tmp_path), binary="/usr/local/bin/llama-server")

    assert command[:2] == ["/usr/local/bin/llama-server", "--model"]
    assert "--alias" in command
    assert "peterbot-gguf" in command
    assert "--api-key" in command
    assert "--metrics" in command
    assert "--cont-batching" in command
    assert "--n-gpu-layers" in command


def test_wait_for_port_detects_listening_socket() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()

    def accept_once() -> None:
        conn, _ = listener.accept()
        conn.close()
        listener.close()

    thread = threading.Thread(target=accept_once, daemon=True)
    thread.start()

    assert wait_for_port(host, port, 2.0) is True


def test_resolve_probe_host_maps_wildcard_bind_to_loopback() -> None:
    assert resolve_probe_host("0.0.0.0") == "127.0.0.1"
    assert resolve_probe_host("::") == "127.0.0.1"
    assert resolve_probe_host("127.0.0.1") == "127.0.0.1"
