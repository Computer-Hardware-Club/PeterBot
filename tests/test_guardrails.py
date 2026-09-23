import pytest
import json
from pathlib import Path

from peterbot.guardrails import GuardLimits, RequestGuard


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def request(guard, user=1, guild=10, prompt="hello"):
    return guard.acquire(user_id=user, guild_id=guild, prompt=prompt)


def complete(guard, user=1, guild=10):
    result = request(guard, user=user, guild=guild)
    if result[0]:
        guard.release(user_id=user)
    return result


@pytest.mark.parametrize("field", [
    "max_concurrent", "user_requests_per_minute", "guild_requests_per_minute", "max_prompt_chars"
])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "3", None])
def test_limits_require_positive_integers(field, value):
    with pytest.raises(ValueError, match=field):
        GuardLimits(**{field: value})


@pytest.mark.parametrize("value", [(0,), (-1,), (True,), ("10",), [10], None])
def test_guild_allowlist_validation(value):
    with pytest.raises(ValueError, match="allowed_guild_ids"):
        GuardLimits(allowed_guild_ids=value)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_dm_setting_requires_boolean(value):
    with pytest.raises(ValueError, match="allow_dms"):
        GuardLimits(allow_dms=value)


def test_guild_and_dm_policies():
    guard = RequestGuard(GuardLimits(allowed_guild_ids=(10,)))
    assert not request(guard, guild=None)[0]
    assert not request(guard, guild=20)[0]
    assert request(guard) == (True, None)
    assert complete(RequestGuard(GuardLimits()), guild=999)[0]
    dm_guard = RequestGuard(GuardLimits(allow_dms=True, allowed_guild_ids=(10,)))
    assert complete(dm_guard, guild=None)[0]
    assert not dm_guard._guild_requests


@pytest.mark.parametrize("prompt", ["", " \n\t", "123456", None])
def test_bad_prompt_does_not_consume_quota_or_slot(prompt):
    guard = RequestGuard(GuardLimits(max_prompt_chars=5, user_requests_per_minute=1))
    accepted, message = request(guard, prompt=prompt)
    assert not accepted and message
    assert not guard._user_requests
    assert not guard._guild_requests
    assert request(guard, prompt="12345") == (True, None)


def test_concurrency_and_per_user_inflight():
    guard = RequestGuard(GuardLimits(max_concurrent=2))
    assert request(guard, user=1)[0]
    assert not request(guard, user=1)[0]
    assert request(guard, user=2)[0]
    assert not request(guard, user=3)[0]
    guard.release(user_id=1)
    assert request(guard, user=3)[0]
    assert len(guard._user_requests[1]) == 1
    assert len(guard._user_requests[3]) == 1


def test_duplicate_and_unknown_release_cannot_free_another_users_slot():
    guard = RequestGuard(GuardLimits())
    assert request(guard, user=1)[0]
    guard.release(user_id=2)
    assert not request(guard, user=2)[0]
    guard.release(user_id=1)
    guard.release(user_id=1)
    assert request(guard, user=2)[0]
    guard.release(user_id=1)
    assert not request(guard, user=3)[0]


def test_user_quota_spans_guilds_and_release_does_not_reset_it():
    guard = RequestGuard(GuardLimits(user_requests_per_minute=2))
    assert complete(guard, guild=10)[0]
    assert complete(guard, guild=20)[0]
    assert not complete(guard, guild=30)[0]
    assert 30 not in guard._guild_requests
    assert request(guard, user=2, guild=30)[0]


def test_configured_quota_does_not_cut_off_ordinary_rapid_chat():
    config = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    agent = config["agent"]
    guard = RequestGuard(GuardLimits(
        user_requests_per_minute=agent["user_requests_per_minute"],
        guild_requests_per_minute=agent["guild_requests_per_minute"],
    ))
    for _ in range(20):
        assert complete(guard)[0]
    assert not complete(guard)[0]


def test_guild_quota_spans_users_and_does_not_consume_rejected_user_quota():
    guard = RequestGuard(GuardLimits(guild_requests_per_minute=2))
    assert complete(guard, user=1)[0]
    assert complete(guard, user=2)[0]
    assert not complete(guard, user=3)[0]
    assert 3 not in guard._user_requests
    assert request(guard, user=3, guild=20)[0]


def test_sliding_window_expires_at_exact_boundary():
    clock = Clock()
    guard = RequestGuard(GuardLimits(user_requests_per_minute=2), clock=clock)
    assert complete(guard)[0]
    clock.now += 30
    assert complete(guard)[0]
    clock.now = 159.999
    assert not complete(guard)[0]
    clock.now = 160
    assert complete(guard)[0]
    assert not complete(guard)[0]
    assert list(guard._user_requests[1]) == [130, 160]
    clock.now = 190
    assert complete(guard)[0]


def test_guild_window_expires_at_exact_boundary():
    clock = Clock()
    guard = RequestGuard(GuardLimits(guild_requests_per_minute=1), clock=clock)
    assert complete(guard)[0]
    clock.now = 159.999
    assert not complete(guard, user=2)[0]
    clock.now = 160
    assert complete(guard, user=2)[0]


def test_identity_flood_has_hard_cap_without_eviction_or_quota_bypass():
    clock = Clock()
    guard = RequestGuard(GuardLimits(user_requests_per_minute=1), clock=clock)
    for identity in range(1, 5001):
        assert complete(guard, user=identity, guild=identity)[0]
    assert len(guard._user_requests) + len(guard._guild_requests) == 10_000
    for identity in range(5001, 5101):
        assert not complete(guard, user=identity, guild=identity)[0]
    assert len(guard._user_requests) + len(guard._guild_requests) == 10_000
    assert not complete(guard, user=1, guild=1)[0]
    assert not guard._active_users
    clock.now = 160
    assert complete(guard, user=6000, guild=6000)[0]
    assert len(guard._user_requests) == len(guard._guild_requests) == 1


def test_capacity_check_accounts_for_both_maps_without_partial_insert():
    guard = RequestGuard(GuardLimits())
    guard._MAX_QUOTA_ENTRIES = 3
    assert complete(guard, user=1, guild=1)[0]
    assert not complete(guard, user=2, guild=2)[0]
    assert 2 not in guard._user_requests
    assert 2 not in guard._guild_requests
    assert complete(guard, user=2, guild=1)[0]
    assert complete(guard, user=1, guild=1)[0]


def test_pruning_preserves_refreshed_identity_and_removes_stale_ones():
    clock = Clock()
    guard = RequestGuard(GuardLimits(), clock=clock)
    assert complete(guard, user=1, guild=1)[0]
    assert complete(guard, user=2, guild=2)[0]
    clock.now = 130
    assert complete(guard, user=1, guild=1)[0]
    clock.now = 160
    assert complete(guard, user=3, guild=3)[0]
    assert set(guard._user_requests) == {1, 3}
    assert set(guard._guild_requests) == {1, 3}


def test_window_expiry_never_releases_an_active_request():
    clock = Clock()
    guard = RequestGuard(GuardLimits(), clock=clock)
    assert request(guard)[0]
    clock.now += 120
    assert not request(guard, user=2)[0]
    guard.release(user_id=1)
    assert request(guard, user=2)[0]
