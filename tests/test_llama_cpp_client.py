import asyncio
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
from peterbot.llama_cpp_client import (
    LlamaCppChatClient,
    build_chat_completion_payload,
    extract_chat_completion_content,
)


def build_config(tmp_path: Path) -> AppConfig:
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
            timeout_seconds=30,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            extra_request_body={"cache_prompt": True},
        ),
        llama_server=LlamaServerConfig(
            enabled=False,
            model_path=None,
            host="127.0.0.1",
            port=8080,
            ctx_size=4096,
            threads=0,
            batch_size=512,
            parallel=1,
            continuous_batching=True,
            n_gpu_layers=0,
            metrics=False,
            extra_args=[],
        ),
        paths=PathsConfig(
            data_dir=str(tmp_path),
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


def test_build_chat_completion_payload_uses_openai_shape() -> None:
    payload = build_chat_completion_payload(
        "peterbot-gguf",
        [{"role": "user", "content": "hello"}],
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
        extra_request_body={"cache_prompt": True},
    )

    assert payload["model"] == "peterbot-gguf"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is False
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["cache_prompt"] is True


def test_extract_chat_completion_content_reads_choice_message() -> None:
    data = {"choices": [{"message": {"content": "Hello from llama.cpp"}}]}
    assert extract_chat_completion_content(data) == "Hello from llama.cpp"


def test_llama_cpp_client_sets_bearer_auth_header(tmp_path: Path) -> None:
    client = LlamaCppChatClient(build_config(tmp_path))
    asyncio.run(client.ensure_http_session())
    try:
        assert client.http_session is not None
        assert client.http_session.headers["Authorization"] == "Bearer secret-key"
    finally:
        asyncio.run(client.close())
