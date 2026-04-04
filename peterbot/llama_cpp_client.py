from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

from .config import AppConfig
from .logging_utils import (
    build_user_debug_message,
    log_error_with_context,
    log_exception_with_context,
    log_with_context,
    new_debug_id,
    truncate_for_log,
)
from .prompts import CHAT_MODE, build_chat_messages, cleanup_response_text, strip_think_blocks

MULTIMODAL_SETUP_MESSAGE = (
    "I can only look at images if the llama.cpp backend is running a multimodal vision model."
)


def is_multimodal_backend_error(error_text: str) -> bool:
    normalized = (error_text or "").lower()
    if not normalized:
        return False

    image_hints = (
        "multimodal",
        "mmproj",
        "vision",
        "image",
        "images",
        "projector",
        "mtmd",
        "clip model",
        "image_url",
    )
    setup_hints = (
        "missing",
        "unsupported",
        "not supported",
        "not enabled",
        "requires",
        "failed to load",
        "cannot load",
        "no such file",
        "disabled",
        "unknown field",
        "invalid type",
    )
    return any(hint in normalized for hint in image_hints) and any(
        hint in normalized for hint in setup_hints
    )


def build_chat_completion_payload(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    extra_request_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(extra_request_body or {})
    payload.update(
        {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    )
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


def extract_chat_completion_content(data: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return strip_think_blocks(content)

    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return strip_think_blocks(content)
    return None


class LlamaCppChatClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def ensure_http_session(self) -> None:
        if self.http_session is None or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.inference.timeout_seconds)
            headers: Dict[str, str] = {}
            if self.config.llama_cpp_api_key:
                headers["Authorization"] = f"Bearer {self.config.llama_cpp_api_key}"
            self.http_session = aiohttp.ClientSession(timeout=timeout, headers=headers)

    async def close(self) -> None:
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()

    async def call_chat(
        self,
        prompt_text: str,
        *,
        system_prompt: str,
        author_name: Optional[str] = None,
        guild_name: Optional[str] = None,
        channel_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_content: Optional[str] = None,
        user_images: Optional[List[str]] = None,
        response_mode: str = CHAT_MODE,
    ) -> str:
        await self.ensure_http_session()
        url = f"{self.config.inference.base_url.rstrip('/')}/v1/chat/completions"
        request_debug_id = new_debug_id("REQ")

        messages = build_chat_messages(
            prompt_text,
            author_name=author_name,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            user_content=user_content,
            user_images=user_images,
            allow_thinking=False,
        )
        payload = build_chat_completion_payload(
            self.config.inference.model,
            messages,
            max_tokens=self.config.inference.max_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            extra_request_body=self.config.inference.extra_request_body,
        )

        log_with_context(
            logging.DEBUG,
            f"[{request_debug_id}] Sending llama.cpp chat request",
            url=url,
            model=self.config.inference.model,
            author_name=author_name,
            guild_name=guild_name,
            channel_name=channel_name,
            prompt_preview=truncate_for_log(prompt_text),
            user_content_preview=truncate_for_log(messages[-1]["content"]),
            history_count=len(conversation_history or []),
            user_image_count=len(user_images or []),
            mode=response_mode,
            timeout_seconds=self.config.inference.timeout_seconds,
        )

        try:
            if self.http_session is None:
                debug_id = log_error_with_context(
                    "HTTP session unavailable before model request",
                    request_id=request_debug_id,
                    url=url,
                    model=self.config.inference.model,
                )
                return build_user_debug_message(
                    "Sorry, my model backend failed to initialize.",
                    debug_id,
                )

            async with self.http_session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    if user_images and is_multimodal_backend_error(error_text):
                        log_with_context(
                            logging.WARNING,
                            f"[{request_debug_id}] Multimodal request rejected by llama.cpp backend",
                            status=resp.status,
                            response_preview=truncate_for_log(error_text, max_chars=500),
                            url=url,
                            model=self.config.inference.model,
                        )
                        return MULTIMODAL_SETUP_MESSAGE

                    debug_id = new_debug_id("LLM")
                    log_with_context(
                        logging.ERROR,
                        f"[{debug_id}] Model backend returned a non-200 status",
                        request_id=request_debug_id,
                        status=resp.status,
                        response_preview=truncate_for_log(error_text, max_chars=500),
                        url=url,
                        model=self.config.inference.model,
                    )
                    return build_user_debug_message(
                        "Sorry, I couldn't reach the model service right now.",
                        debug_id,
                    )

                data = await resp.json(content_type=None)
                content = extract_chat_completion_content(data) or "(No response from model)"
                return cleanup_response_text(
                    content,
                    profile=self.config.model_profile,
                    mode=response_mode,
                )
        except asyncio.TimeoutError:
            debug_id = log_exception_with_context(
                "Model request timed out",
                request_id=request_debug_id,
                url=url,
                model=self.config.inference.model,
                author_name=author_name,
                guild_name=guild_name,
                channel_name=channel_name,
            )
            return build_user_debug_message("Sorry, the model took too long to respond.", debug_id)
        except aiohttp.ClientError:
            debug_id = log_exception_with_context(
                "Model connection error",
                request_id=request_debug_id,
                url=url,
                model=self.config.inference.model,
                author_name=author_name,
                guild_name=guild_name,
                channel_name=channel_name,
            )
            return build_user_debug_message(
                "Sorry, my model backend is unavailable right now.",
                debug_id,
            )
        except Exception:
            debug_id = log_exception_with_context(
                "Unexpected model backend error",
                request_id=request_debug_id,
                url=url,
                model=self.config.inference.model,
                author_name=author_name,
                guild_name=guild_name,
                channel_name=channel_name,
            )
            return build_user_debug_message(
                "Sorry, something went wrong while generating a response.",
                debug_id,
            )
