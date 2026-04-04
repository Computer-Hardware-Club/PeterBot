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
    MULTIMODAL_SETUP_MESSAGE,
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


class FakeResponse:
    def __init__(self, *, status: int, json_data=None, text_data: str = "") -> None:
        self.status = status
        self._json_data = json_data
        self._text_data = text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def json(self, content_type=None):
        return self._json_data

    async def text(self) -> str:
        return self._text_data


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.closed = False
        self.requests = []

    def post(self, url: str, json):
        self.requests.append({"url": url, "json": json})
        return self.response

    async def close(self) -> None:
        self.closed = True


def test_llama_cpp_client_includes_images_in_chat_payload(tmp_path: Path) -> None:
    client = LlamaCppChatClient(build_config(tmp_path))
    session = FakeSession(
        FakeResponse(
            status=200,
            json_data={"choices": [{"message": {"content": "That looks fine."}}]},
        )
    )
    client.http_session = session

    reply = asyncio.run(
        client.call_chat(
            "Thoughts?",
            system_prompt="You are Peter.",
            user_images=["base64-image"],
        )
    )

    assert reply == "That looks fine."
    assert session.requests[0]["json"]["messages"][-1]["images"] == ["base64-image"]


def test_llama_cpp_client_returns_clear_message_for_multimodal_setup_errors(tmp_path: Path) -> None:
    client = LlamaCppChatClient(build_config(tmp_path))
    session = FakeSession(
        FakeResponse(
            status=500,
            text_data='{"error":{"message":"multimodal projector missing: pass --mmproj for image input"}}',
        )
    )
    client.http_session = session

    reply = asyncio.run(
        client.call_chat(
            "What do you think about this?",
            system_prompt="You are Peter.",
            user_images=["base64-image"],
        )
    )

    assert reply == MULTIMODAL_SETUP_MESSAGE
