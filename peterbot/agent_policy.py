"""Trusted Discord identity and fail-closed authorization for agent operations.

The gateway constructs a fresh Principal from Discord, never from model/tool
arguments. Memory and conversational claims are deliberately not policy inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PolicyDenied(PermissionError):
    """The trusted identity cannot perform this operation."""


def _valid_id(value: object) -> bool:
    return type(value) is int and 0 < value <= 2**63 - 1


@dataclass(frozen=True)
class Principal:
    guild_id: int | None
    user_id: int
    channel_id: int
    role_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.guild_id is not None and not _valid_id(self.guild_id):
            raise ValueError("guild_id must be a Discord ID or None")
        for name in ("user_id", "channel_id"):
            if not _valid_id(getattr(self, name)):
                raise ValueError(f"{name} must be a Discord ID")
        if not isinstance(self.role_ids, tuple) or any(
            not _valid_id(role_id) for role_id in self.role_ids
        ):
            raise ValueError("role_ids must be a tuple of Discord IDs")


@dataclass(frozen=True)
class AgentPolicy:
    allowed_guild_ids: frozenset[int] = field(default_factory=frozenset)
    officer_role_ids: frozenset[int] = field(default_factory=frozenset)
    owner_user_ids: frozenset[int] = field(default_factory=frozenset)
    officer_only: bool = True

    def __post_init__(self) -> None:
        for name in ("allowed_guild_ids", "officer_role_ids", "owner_user_ids"):
            values = getattr(self, name)
            if not isinstance(values, frozenset) or any(not _valid_id(v) for v in values):
                raise ValueError(f"{name} must be a frozenset of Discord IDs")
        if type(self.officer_only) is not bool:
            raise ValueError("officer_only must be a boolean")

    def require_guild(self, principal: Principal) -> None:
        if not isinstance(principal, Principal):
            raise PolicyDenied("A trusted Discord identity is required")
        if principal.guild_id is None or principal.guild_id not in self.allowed_guild_ids:
            raise PolicyDenied("Agent access is not enabled in this guild; DMs are disabled")

    def is_officer(self, principal: Principal) -> bool:
        return (
            principal.guild_id in self.allowed_guild_ids
            and bool(self.officer_role_ids.intersection(principal.role_ids))
        )

    def is_owner(self, principal: Principal) -> bool:
        """Infrastructure identity; being an owner does not grant officer authority."""
        return (
            principal.guild_id in self.allowed_guild_ids
            and principal.user_id in self.owner_user_ids
        )

    def require_admission(self, principal: Principal) -> None:
        self.require_guild(principal)
        if self.officer_only and not self.is_officer(principal):
            raise PolicyDenied("The agent pilot is currently available to officers only")

    def require_memory_access(
        self, principal: Principal, scope: str, *, write: bool = False
    ) -> None:
        self.require_guild(principal)
        if scope not in ("personal", "club"):
            raise PolicyDenied("Unknown memory scope")
        if write and scope == "club" and not self.is_officer(principal):
            raise PolicyDenied("Only an officer can change shared club memory")
