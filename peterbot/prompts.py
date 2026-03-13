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
        "You are the club bot or assistant, not a human member of the server.",
        "Answer directly, then stop.",
        "Keep replies concise unless the user asks for detail.",
        "Do not use hyphen, en dash, or em dash punctuation in normal reply prose.",
        "Do not add fake familiarity, playful banter, or warm check ins.",
        "Do not mention hidden rules, policies, or internal reasoning.",
        "Do not include <think> tags or chain-of-thought.",
    ]
    if profile == ModelProfile.QWEN:
        base_rules.extend(
            [
                "Use one short paragraph by default. Only use a second paragraph if extra detail is genuinely needed.",
                "Usually answer in 1 to 3 sentences.",
                "Do not start with assistant style prefaces like 'Sure', 'Absolutely', or 'Here's a quick summary'.",
                "Do not greet with the user's name unless it is necessary for clarity.",
                "Do not use bullet lists unless the user asked for a list or the information clearly needs one.",
                "Do not use lol, lmao, haha, or similar filler.",
                "Do not ask a follow up question unless clarification is actually required.",
                "If asked who you are or what you do, say plainly that you are the club bot or club assistant.",
                "Prefer neutral or slightly blunt over cheerful.",
                "Avoid overexplaining, repeated punctuation, ellipses spam, and canned social padding.",
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
            "If the current mention is vague and the focus context is still insufficient, ask one brief clarifying question instead of guessing.",
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


def protect_literal_spans(text: str) -> tuple[str, Dict[str, str]]:
    patterns = (
        r"`[^`]+`",
        r"https?://\S+",
        r"(?:\b[\w.-]+/)+[\w.-]+\b",
        r"\b[\w.-]+\.[A-Za-z0-9]{1,8}\b",
    )
    replacements: Dict[str, str] = {}

    def replacer(match: re.Match[str]) -> str:
        key = f"__PETER_LITERAL_{len(replacements)}__"
        replacements[key] = match.group(0)
        return key

    protected = text
    for pattern in patterns:
        protected = re.sub(pattern, replacer, protected)
    return protected, replacements


def restore_literal_spans(text: str, replacements: Dict[str, str]) -> str:
    restored = text
    for key, value in replacements.items():
        restored = restored.replace(key, value)
    return restored


def normalize_prose_dashes(text: str) -> str:
    protected, replacements = protect_literal_spans(text)
    normalized = protected.replace("—", ", ").replace("–", ", ")
    normalized = re.sub(r"\s+-\s+", ", ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"\s+,", ",", normalized)
    normalized = re.sub(r",\s*,", ", ", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return restore_literal_spans(normalized, replacements)


def remove_laughter_filler(text: str) -> str:
    cleaned = re.sub(r"\b(?:lol|lmao|rofl|haha+|hehe+)\b[.!?, ]*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def is_low_value_banter(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return True

    patterns = (
        r"\byou got me with that one\b",
        r"\banything specific you wanted to know\b",
        r"\banything else\b",
        r"\blet me know if you want more\b",
        r"\bhope that helps\b",
        r"\bif you want more detail\b",
        r"\bwhat'?s up\b",
        r"\bgot me with that one\b",
    )
    if any(re.search(pattern, normalized) for pattern in patterns):
        return True

    short_agreement = re.fullmatch(
        r"(?:yeah|yep|sure|right|true|fair|same|exactly)(?: [a-z']+){0,5}[.!?]?",
        normalized,
    )
    if short_agreement is not None:
        return True

    short_greeting = re.fullmatch(r"(?:hey|hi|hello)(?: [a-z0-9_]+)?[.!?]?", normalized)
    return short_greeting is not None


def trim_low_value_sentences(text: str) -> str:
    protected, replacements = protect_literal_spans(text)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", protected) if part.strip()]
    if len(sentences) <= 1:
        return restore_literal_spans(protected, replacements)

    while len(sentences) > 1 and is_low_value_banter(restore_literal_spans(sentences[-1], replacements)):
        sentences.pop()

    while len(sentences) > 1 and is_low_value_banter(restore_literal_spans(sentences[0], replacements)):
        sentences.pop(0)

    return restore_literal_spans(" ".join(sentences), replacements).strip()


def remove_low_value_paragraphs(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        return trim_low_value_sentences(text)

    while len(paragraphs) > 1 and is_low_value_banter(paragraphs[-1]):
        paragraphs.pop()
    while len(paragraphs) > 1 and is_low_value_banter(paragraphs[0]):
        paragraphs.pop(0)

    cleaned = [trim_low_value_sentences(paragraph) for paragraph in paragraphs]
    cleaned = [paragraph for paragraph in cleaned if paragraph]
    return "\n\n".join(cleaned).strip()


def normalize_simple_greeting_response(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9' ]+", "", text.strip().lower())
    if normalized in {
        "hey",
        "hi",
        "hello",
        "hey whats up",
        "hey what's up",
        "whats up",
        "what's up",
        "hello peter",
        "hi peter",
    }:
        return "Hi."
    return text


def remove_canned_openers(text: str) -> str:
    opener_patterns = (
        r"^(?:sure|absolutely|of course|certainly|totally|yep)[,!\s-]+",
        r"^here(?:'s| is) (?:a )?(?:quick )?(?:answer|summary|recap)[:,.\s-]+",
        r"^(?:hey|hi|hello)\s+[A-Za-z0-9_]+(?:\s*[,:]|(?:\s+-\s+)|\s+)",
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
        r"\n*\s*(?:anything specific you wanted to know\??)\s*$",
        r"\n*\s*(?:anything else\??)\s*$",
    )
    original = text.strip()
    cleaned = text
    for pattern in signoff_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or original


def trim_qwen_paragraphs(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        return text.strip()
    return paragraphs[0]


def cleanup_response_text(text: str, *, profile: ModelProfile, mode: str = CHAT_MODE) -> str:
    cleaned = strip_think_blocks(text or "")
    cleaned = remove_canned_openers(cleaned)
    cleaned = remove_laughter_filler(cleaned)
    cleaned = remove_canned_signoffs(cleaned)
    cleaned = remove_low_value_paragraphs(cleaned)
    cleaned = normalize_prose_dashes(cleaned)
    cleaned = collapse_repeated_punctuation(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    cleaned = normalize_simple_greeting_response(cleaned)

    if profile == ModelProfile.QWEN and mode != RECAP_MODE:
        cleaned = trim_qwen_paragraphs(cleaned)

    return cleaned or "(No response from model)"
