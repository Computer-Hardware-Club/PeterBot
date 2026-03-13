from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .logging_utils import log_exception_with_context


def tokenize_relevance(text: str) -> List[str]:
    raw_tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    stop_words = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "at",
        "be",
        "but",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "them",
        "these",
        "they",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
    tokens: list[str] = []
    for token in raw_tokens:
        if token.endswith("s") and len(token) > 4 and not token.endswith("'s"):
            token = token[:-1]
        if len(token) > 2 and token not in stop_words:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class ChannelProfile:
    key: str
    tone: str = ""
    reply_length: str = ""
    topics: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeChunk:
    heading: str
    body: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeIndex:
    chunks: tuple[KnowledgeChunk, ...] = ()
    channel_profiles: Dict[str, ChannelProfile] = field(default_factory=dict)


def parse_markdown_knowledge(text: str) -> List[KnowledgeChunk]:
    lines = text.splitlines()
    sections: list[KnowledgeChunk] = []
    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal body_lines
        heading_parts = [part for part in (current_h2, current_h3) if part]
        body = "\n".join(body_lines).strip()
        if not heading_parts or not body:
            body_lines = []
            return
        heading = " > ".join(heading_parts)
        sections.append(
            KnowledgeChunk(
                heading=heading,
                body=body,
                tokens=tuple(tokenize_relevance(f"{heading} {body}")),
            )
        )
        body_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
            current_h3 = None
            continue
        if line.startswith("### "):
            flush()
            current_h3 = line[4:].strip()
            continue
        body_lines.append(line)

    flush()
    return sections


def load_knowledge_chunks(path: Optional[str]) -> tuple[KnowledgeChunk, ...]:
    if not path:
        return ()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except Exception:
        log_exception_with_context("Failed reading knowledge file", knowledge_file=path)
        return ()
    return tuple(parse_markdown_knowledge(text))


def load_channel_profiles(path: Optional[str]) -> Dict[str, ChannelProfile]:
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        log_exception_with_context("Failed reading channel profiles file", channel_profiles_file=path)
        return {}

    if not isinstance(raw, dict):
        return {}

    resolved: Dict[str, ChannelProfile] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        topics = value.get("topics") or []
        if not isinstance(topics, list):
            topics = []
        resolved[str(key)] = ChannelProfile(
            key=str(key),
            tone=str(value.get("tone", "") or ""),
            reply_length=str(value.get("reply_length", "") or ""),
            topics=tuple(str(topic) for topic in topics if str(topic).strip()),
        )
    return resolved


def load_knowledge_index(
    *,
    knowledge_file: Optional[str],
    channel_profiles_file: Optional[str],
) -> KnowledgeIndex:
    return KnowledgeIndex(
        chunks=load_knowledge_chunks(knowledge_file),
        channel_profiles=load_channel_profiles(channel_profiles_file),
    )


def resolve_channel_profile(channel: Any, profiles: Dict[str, ChannelProfile]) -> Optional[ChannelProfile]:
    if not profiles:
        return None
    channel_id = getattr(channel, "id", None)
    channel_name = getattr(channel, "name", None)
    for key in (str(channel_id) if channel_id is not None else None, channel_name):
        if key and key in profiles:
            return profiles[key]
    return None


def rank_knowledge_chunks(
    prompt_text: str,
    chunks: Sequence[KnowledgeChunk],
    *,
    channel_profile: Optional[ChannelProfile] = None,
    max_chunks: int = 2,
) -> List[KnowledgeChunk]:
    if not prompt_text or not chunks:
        return []

    prompt_tokens = set(tokenize_relevance(prompt_text))
    if not prompt_tokens:
        return []
    topic_tokens = {
        token
        for topic in (channel_profile.topics if channel_profile else ())
        for token in tokenize_relevance(topic)
    }
    weighted_tokens = prompt_tokens.union(topic_tokens)
    if not weighted_tokens:
        return []

    scored: list[tuple[int, KnowledgeChunk]] = []
    for chunk in chunks:
        chunk_tokens = set(chunk.tokens)
        prompt_overlap = chunk_tokens.intersection(prompt_tokens)
        if not prompt_overlap:
            continue

        score = len(prompt_overlap) * 10
        heading_tokens = set(tokenize_relevance(chunk.heading))
        score += len(heading_tokens.intersection(prompt_tokens)) * 8
        score += len(chunk_tokens.intersection(topic_tokens)) * 4

        prompt_text_lower = prompt_text.lower()
        if chunk.heading.lower() in prompt_text_lower:
            score += 12

        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:max_chunks]]


def build_knowledge_excerpt(chunks: Iterable[KnowledgeChunk], max_chars: int = 1400) -> Optional[str]:
    sections: list[str] = []
    total = 0
    for chunk in chunks:
        candidate = f"{chunk.heading}\n{chunk.body}".strip()
        if not candidate:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(candidate) > remaining:
            if remaining < 32:
                break
            candidate = candidate[: remaining - 1].rstrip() + "…"
        sections.append(candidate)
        total += len(candidate)
        if total >= max_chars:
            break
    if not sections:
        return None
    return "\n\n".join(sections)
