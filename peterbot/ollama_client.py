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


def build_ollama_payload(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    think: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": think,
        "messages": messages,
    }
    if options:
        payload["options"] = options
    return payload


def extract_ollama_response_content(data: Dict[str, Any]) -> Optional[str]:
    msg = data.get("message", {}) if isinstance(data, dict) else {}
    content = msg.get("content")
    if not content:
        content = data.get("response") if isinstance(data, dict) else None
    return strip_think_blocks(content)


class OllamaChatClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def ensure_http_session(self) -> None:
        if self.http_session is None or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.ollama_timeout_seconds)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

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
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        request_debug_id = new_debug_id("REQ")

        messages = build_chat_messages(
            prompt_text,
            author_name=author_name,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            user_content=user_content,
            user_images=user_images,
            allow_thinking=self.config.ollama_think,
        )
        payload = build_ollama_payload(
            self.config.ollama_model,
            messages,
            think=self.config.ollama_think,
            options=self.config.ollama_options,
        )

        log_with_context(
            logging.DEBUG,
            f"[{request_debug_id}] Sending Ollama chat request",
            url=url,
            model=self.config.ollama_model,
            author_name=author_name,
            guild_name=guild_name,
            channel_name=channel_name,
            prompt_preview=truncate_for_log(prompt_text),
            user_content_preview=truncate_for_log(messages[-1]["content"]),
            history_count=len(conversation_history or []),
            user_image_count=len(user_images or []),
            mode=response_mode,
            timeout_seconds=self.config.ollama_timeout_seconds,
        )

        try:
            if self.http_session is None:
                debug_id = log_error_with_context(
                    "HTTP session unavailable before Ollama request",
                    request_id=request_debug_id,
                    url=url,
                    model=self.config.ollama_model,
                )
                return build_user_debug_message(
                    "Sorry, my model backend failed to initialize.",
                    debug_id,
                )

            allow_image_retry = bool(user_images)
            while True:
                async with self.http_session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        if allow_image_retry:
                            allow_image_retry = False
                            retry_messages = [dict(message) for message in payload["messages"]]
                            retry_messages[-1].pop("images", None)
                            payload = {**payload, "messages": retry_messages}
                            log_with_context(
                                logging.WARNING,
                                "Retrying Ollama chat without images after multimodal failure",
                                request_id=request_debug_id,
                                status=resp.status,
                                model=self.config.ollama_model,
                                response_preview=truncate_for_log(error_text, max_chars=500),
                            )
                            continue

                        debug_id = new_debug_id("OLL")
                        log_with_context(
                            logging.ERROR,
                            f"[{debug_id}] Ollama chat failed with non-200 status",
                            request_id=request_debug_id,
                            status=resp.status,
                            response_preview=truncate_for_log(error_text, max_chars=500),
                            url=url,
                            model=self.config.ollama_model,
                        )
                        return build_user_debug_message(
                            "Sorry, I couldn't reach the model service right now.",
                            debug_id,
                        )

                    data = await resp.json(content_type=None)
                    content = extract_ollama_response_content(data) or "(No response from model)"
                    return cleanup_response_text(
                        content,
                        profile=self.config.model_profile,
                        mode=response_mode,
                    )
        except asyncio.TimeoutError:
            debug_id = log_exception_with_context(
                "Ollama request timed out",
                request_id=request_debug_id,
                url=url,
                model=self.config.ollama_model,
                author_name=author_name,
                guild_name=guild_name,
                channel_name=channel_name,
            )
            return build_user_debug_message("Sorry, the model took too long to respond.", debug_id)
        except aiohttp.ClientError:
            debug_id = log_exception_with_context(
                "Ollama connection error",
                request_id=request_debug_id,
                url=url,
                model=self.config.ollama_model,
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
                "Unexpected Ollama error",
                request_id=request_debug_id,
                url=url,
                model=self.config.ollama_model,
                author_name=author_name,
                guild_name=guild_name,
                channel_name=channel_name,
            )
            return build_user_debug_message(
                "Sorry, something went wrong while generating a response.",
                debug_id,
            )
