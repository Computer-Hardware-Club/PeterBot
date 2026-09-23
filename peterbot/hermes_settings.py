"""Separate, explicit opt-in configuration for the Hermes deployment."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class HermesSettings:
    allowed_guild_ids: frozenset[int]
    officer_role_ids: frozenset[int]
    owner_user_ids: frozenset[int]
    runner_url: str
    tool_service_url: str
    runner_token: str
    state_dir: str
    officer_only: bool = True
    member_work_enabled: bool = False
    listen_channel_ids: frozenset[int] = frozenset()
    control_channel_ids: frozenset[int] = frozenset()
    announcement_destination_ids: frozenset[int] = frozenset()
    conversation_lease_seconds: int = 120
    max_iterations: int = 30
    max_tokens: int = 8192
    max_model_calls: int = 40
    max_job_output_tokens: int = 131072
    max_tool_calls: int = 100
    job_timeout: int = 1200

    @classmethod
    def load(cls, path: str) -> 'HermesSettings':
        with open(path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError('Hermes config must be an object')
        ids = {}
        for key in ('allowed_guild_ids','officer_role_ids','owner_user_ids'):
            values = raw.get(key, [])
            if not isinstance(values, list) or any(type(v) is not int or not 0 < v < 2**63 for v in values):
                raise ValueError(f'{key} must contain positive integer IDs')
            ids[key] = frozenset(values)
        if not ids['allowed_guild_ids'] or not (ids['officer_role_ids'] or ids['owner_user_ids']):
            raise ValueError('Hermes requires an explicit guild and officer/owner allowlist')
        token = os.environ.get('PETERBOT_RUNNER_TOKEN','')
        if len(token) < 32:
            raise ValueError('PETERBOT_RUNNER_TOKEN must be at least 32 characters')
        urls = {}
        for key in ('runner_url','tool_service_url'):
            value = raw.get(key,'')
            if not isinstance(value, str):
                raise ValueError(f'Invalid {key}')
            p = urlparse(value)
            _ = p.port
            if p.scheme not in {'http','https'} or not p.hostname or p.username or p.password or p.query or p.fragment:
                raise ValueError(f'Invalid {key}')
            urls[key] = value.rstrip('/')
        officer_only = raw.get('officer_only', True)
        if type(officer_only) is not bool:
            raise ValueError('officer_only must be a boolean')
        member_work_enabled = raw.get('member_work_enabled', False)
        if type(member_work_enabled) is not bool:
            raise ValueError('member_work_enabled must be a boolean')
        state_dir = raw.get('state_dir','/app/peterbot-data/hermes')
        if not isinstance(state_dir, str) or not state_dir or not Path(state_dir).is_absolute():
            raise ValueError('state_dir must be an absolute path')
        if officer_only and not ids['officer_role_ids']:
            raise ValueError('Officer pilot requires officer_role_ids')
        channel_ids = {}
        for key in ('listen_channel_ids', 'control_channel_ids', 'announcement_destination_ids'):
            values = raw.get(key, [])
            if not isinstance(values, list) or any(type(v) is not int or not 0 < v < 2**63 for v in values):
                raise ValueError(f'{key} must contain positive integer IDs')
            channel_ids[key] = frozenset(values)
        lease_seconds = raw.get('conversation_lease_seconds', 120)
        if type(lease_seconds) is not int or not 30 <= lease_seconds <= 600:
            raise ValueError('Invalid conversation_lease_seconds')
        limits = {}
        for key, minimum, maximum in (('max_iterations',1,60),('max_tokens',1024,16384),('max_model_calls',1,80),('max_job_output_tokens',8192,262144),('max_tool_calls',1,200),('job_timeout',60,1800)):
            value = raw.get(key, cls.__dataclass_fields__[key].default)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f'Invalid {key}')
            limits[key] = value
        return cls(**ids, **urls, runner_token=token, state_dir=state_dir,
                   officer_only=officer_only, member_work_enabled=member_work_enabled,
                   **channel_ids, conversation_lease_seconds=lease_seconds, **limits)
