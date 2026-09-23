"""Conservative parser for an officer's original Discord control message.

Parsing proposes an action; it never supplies authority. The gateway constructs
a source-bound ControlIntent only after fresh Discord role/channel checks.
Quoted text, pages, task output and prior messages must never call this parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .club_state import OfficerRequest


OFFICES = {
    "president": "president",
    "vice president": "vice_president",
    "vp": "vice_president",
    "treasurer": "treasurer",
    "secretary": "secretary",
    "events chair": "events_chair",
    "outreach chair": "outreach_chair",
}
_OFFICE_NAMES = "|".join(re.escape(name) for name in sorted(OFFICES, key=len, reverse=True))
_ROSTER_CLAUSE = re.compile(
    rf"(?P<name><@!?\d{{1,20}}>|[A-Za-z][A-Za-z .'-]{{0,79}}?)\s+is\s+(?:the\s+)?"
    rf"(?P<office>{_OFFICE_NAMES})(?:\s+for\s+(?P<term>[A-Za-z0-9 /'-]{{2,80}}))?",
    re.IGNORECASE,
)
_FACT = re.compile(
    r"(?:set|update|record|publish)\s+(?P<visibility>public|private)\s+"
    r"(?:club\s+)?fact\s+(?P<key>[a-z][a-z0-9_]{0,63})\s+(?:to|as|=)\s+(?P<value>.+)",
    re.IGNORECASE,
)
_ANNOUNCE = re.compile(r"(?:announce|post)\s+(?:in|to)\s+<#(?P<target>\d{1,20})>\s*[:,-]?\s*(?P<content>.+)", re.IGNORECASE)
_UNDO = re.compile(r"undo\s+(?:the\s+)?(?:last\s+)?(?P<kind>club fact|fact|roster|style)\s*", re.IGNORECASE)
_STYLE_START = re.compile(r"(?:be|sound|speak|talk|keep|make your replies|write)\b", re.IGNORECASE)
_STYLE_WORD = re.compile(r"\b(?:formal|casual|verbose|brief|short|humor|funny|reserved|quiet|chatty|serious)\b", re.IGNORECASE)
_PREFIX = re.compile(r"^peter[,:]?\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ControlRequest:
    action: str
    payload: dict


def parse_control_request(message_text: str, *, bot_user_id: int | None = None) -> ControlRequest | None:
    """Recognize only a clear single original request; None means ordinary chat."""
    if not isinstance(message_text, str) or not 1 <= len(message_text) <= 2000:
        return None
    text = message_text.strip()
    if not text or text.startswith(('>', '`', '"')) or '\n' in text or '\r' in text:
        return None
    if type(bot_user_id) is int and bot_user_id > 0:
        text = re.sub(rf"^<@!?{bot_user_id}>[,:]?\s+", '', text, count=1).strip()
    text = _PREFIX.sub('', text, count=1).strip()
    if match := _FACT.fullmatch(text):
        return ControlRequest('club_fact', {
            'key': match['key'].lower(), 'value': match['value'].strip(),
            'visibility': match['visibility'].lower(),
        })
    if match := _ANNOUNCE.fullmatch(text):
        return ControlRequest('announcement', {
            'target_channel_id': int(match['target']), 'content': match['content'].strip(),
        })
    if match := _UNDO.fullmatch(text):
        action = {'fact': 'club_fact', 'club fact': 'club_fact',
                  'roster': 'roster', 'style': 'style'}[match['kind'].lower()]
        return ControlRequest(action, {'undo': True})
    roster_text = re.sub(r"^(?:set|update|publish)\s+(?:the\s+)?roster\s*:?\s*", '', text, flags=re.IGNORECASE)
    if roster_text != text or ' is ' in text.lower():
        clauses = roster_text.split(';')
        if not 1 <= len(clauses) <= 8:
            return None
        assignments = []
        terms = []
        for clause in clauses:
            match = _ROSTER_CLAUSE.fullmatch(clause.strip())
            if match is None:
                return None
            assignments.append(OfficerRequest(name=match['name'].strip(),
                                              office=OFFICES[match['office'].lower()]))
            if match['term']:
                terms.append(match['term'].strip())
        if not terms or len({term.casefold() for term in terms}) != 1:
            return ControlRequest('roster', {'ambiguous': 'Please specify one term for the roster update.'})
        return ControlRequest('roster', {'assignments': tuple(assignments),
                                         'term': terms[0], 'replace_all': False})
    if _STYLE_START.match(text) and _STYLE_WORD.search(text) and len(text) <= 200:
        return ControlRequest('style', {'request_text': text})
    return None
