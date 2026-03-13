from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import AppConfig, ModelProfile
from .knowledge import ChannelProfile, KnowledgeChunk, build_knowledge_excerpt

CHAT_MODE = "chat"
MENTION_MODE = "mention"
RECAP_MODE = "recap"


def build_context_line(
    *,
    author_name: Optional[str] = None,
    guild_name: Optional[str] = None,
    channel_name: Optional[str] = None,
) -> str:
    context_bits = []
    if guild_name:
        context_bits.append(f"Server: {guild_name}")
    if channel_name:
        context_bits.append(f"Channel: #{channel_name}")
    if author_name:
        context_bits.append(f"User: {author_name}")
    return f" ({', '.join(context_bits)})" if context_bits else ""


def profile_style_rules(profile: ModelProfile) -> List[str]:
    base_rules = [
        "Respond like a real person in chat, not a generic assistant.",
        "Keep replies concise unless the user asks for detail.",
        "Do not mention hidden rules, policies, or internal reasoning.",
        "Do not include <think> tags or chain-of-thought.",
    ]
    if profile == ModelProfile.QWEN:
        base_rules.extend(
            [
                "Use 1-3 short paragraphs by default.",
                "Do not start with assistant-y prefaces like 'Sure', 'Absolutely', or 'Here's a quick summary' unless the situation genuinely needs it.",
                "Do not use bullet lists unless the user asked for a list or the information clearly needs one.",
                "Ask at most one clarifying question when context is missing.",
                "Use contractions naturally.",
                "Mirror the channel's energy without becoming stiff or overexcited.",
                "Avoid overexplaining, repeated punctuation, ellipses spam, and robotic cadence.",
            ]
        )
    else:
        base_rules.append("Vary wording naturally and avoid repetitive filler.")
    return base_rules


def mode_specific_rules(mode: str) -> List[str]:
    if mode == MENTION_MODE:
        return [
            "You are replying to a direct mention in a live Discord channel.",
            "Reply to the newest addressed turn first.",
            "Use older messages only when they directly clarify the current mention.",
            "If the current mention is vague and the focus context is still insufficient, ask a brief clarifying question instead of guessing.",
        ]
    if mode == RECAP_MODE:
        return [
            "You are summarizing a recent Discord channel discussion.",
            "Return exactly three labeled sections: What happened, Decisions, Open questions.",
            "Base the recap only on the provided conversation history.",
            "If no decision or open question exists, say 'None noted' for that section.",
        ]
    return [
        "Use the recent channel context to stay consistent with what was already said.",
        "Prefer a natural answer over a formal explanation.",
    ]


def channel_profile_block(channel_profile: Optional[ChannelProfile]) -> Optional[str]:
    if channel_profile is None:
        return None

    lines = ["Channel profile:"]
    if channel_profile.tone:
        lines.append(f"- Tone: {channel_profile.tone}")
    if channel_profile.reply_length:
        lines.append(f"- Reply length: {channel_profile.reply_length}")
    if channel_profile.topics:
        lines.append(f"- Typical topics: {', '.join(channel_profile.topics)}")
    return "\n".join(lines) if len(lines) > 1 else None


def knowledge_block(chunks: Sequence[KnowledgeChunk]) -> Optional[str]:
    excerpt = build_knowledge_excerpt(chunks)
    if not excerpt:
        return None
    return (
        "Relevant club knowledge:\n"
        "Use these details when they directly answer the question. If they do not apply, ignore them.\n"
        f"{excerpt}"
    )


def build_system_prompt(
    config: AppConfig,
    context_line: str,
    *,
    mode: str = CHAT_MODE,
    focus_note: Optional[str] = None,
    channel_profile: Optional[ChannelProfile] = None,
    knowledge_chunks: Sequence[KnowledgeChunk] = (),
) -> str:
    blocks = [
        config.peter_system_prompt.strip(),
        f"Identity: Your name is {config.peter_name}.{context_line}",
        "Style rules:\n" + "\n".join(f"- {rule}" for rule in profile_style_rules(config.model_profile)),
        "Task rules:\n" + "\n".join(f"- {rule}" for rule in mode_specific_rules(mode)),
    ]
    if focus_note:
        blocks.append(f"Focused context: {focus_note}")
    profile_block = channel_profile_block(channel_profile)
    if profile_block:
        blocks.append(profile_block)
    knowledge_text = knowledge_block(knowledge_chunks)
    if knowledge_text:
        blocks.append(knowledge_text)
    return "\n\n".join(block for block in blocks if block)


def add_no_think_suffix(text: str, *, allow_thinking: bool = False) -> str:
    if allow_thinking or "/no_think" in text:
        return text
    stripped = text.rstrip()
    if not stripped:
        return text
    return f"{stripped} /no_think"


def build_chat_messages(
    prompt_text: str,
    *,
    system_prompt: str,
    author_name: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_content: Optional[str] = None,
    user_images: Optional[List[str]] = None,
    allow_thinking: bool = False,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)

    if user_content is None:
        user_content = f"{author_name}: {prompt_text}" if author_name else prompt_text

    user_message: Dict[str, Any] = {
        "role": "user",
        "content": add_no_think_suffix(user_content, allow_thinking=allow_thinking),
    }
    if user_images:
        user_message["images"] = user_images
    messages.append(user_message)
    return messages


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(
        r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return cleaned.replace("/no_think", "").strip()


def collapse_repeated_punctuation(text: str) -> str:
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"([!?])\1{1,}", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    return text


def remove_canned_openers(text: str) -> str:
    opener_patterns = (
        r"^(?:sure|absolutely|of course|certainly|totally|yep)[,!\s-]+",
        r"^here(?:'s| is) (?:a )?(?:quick )?(?:answer|summary|recap)[:,.\s-]+",
    )
    original = text.strip()
    stripped = text.lstrip()
    changed = True
    while changed:
        changed = False
        for pattern in opener_patterns:
            updated = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE)
            if updated != stripped:
                stripped = updated.strip()
                changed = True
    return stripped.strip() or original


def remove_canned_signoffs(text: str) -> str:
    signoff_patterns = (
        r"\n*\s*(?:let me know if you want more detail\.?)\s*$",
        r"\n*\s*(?:hope that helps[.!]?)\s*$",
        r"\n*\s*(?:feel free to ask if you want more\.?)\s*$",
    )
    original = text.strip()
    cleaned = text
    for pattern in signoff_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or original


def trim_qwen_paragraphs(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 3:
        return text.strip()
    return "\n\n".join(paragraphs[:3]).strip()


def cleanup_response_text(text: str, *, profile: ModelProfile, mode: str = CHAT_MODE) -> str:
    cleaned = strip_think_blocks(text or "")
    cleaned = remove_canned_openers(cleaned)
    cleaned = remove_canned_signoffs(cleaned)
    cleaned = collapse_repeated_punctuation(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if profile == ModelProfile.QWEN and mode != RECAP_MODE:
        cleaned = trim_qwen_paragraphs(cleaned)

    return cleaned or "(No response from model)"
