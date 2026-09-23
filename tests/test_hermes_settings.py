from dataclasses import FrozenInstanceError
import json

import pytest

from peterbot.hermes_settings import HermesSettings


@pytest.fixture
def load(tmp_path, monkeypatch):
    monkeypatch.setenv("PETERBOT_RUNNER_TOKEN", "a" * 48)

    def run(overrides=None, raw=None):
        settings = {
            "allowed_guild_ids": [10], "officer_role_ids": [100], "owner_user_ids": [1],
            "runner_url": "http://runner:8090/", "tool_service_url": "http://gateway:8091/",
        }
        settings.update(overrides or {})
        path = tmp_path / "hermes.json"
        path.write_text(json.dumps(settings if raw is None else raw))
        return HermesSettings.load(str(path))

    return run


def test_valid_settings_defaults_and_normalization(load):
    settings = load()
    assert settings.allowed_guild_ids == frozenset({10})
    assert settings.officer_role_ids == frozenset({100})
    assert settings.owner_user_ids == frozenset({1})
    assert settings.runner_url == "http://runner:8090"
    assert settings.tool_service_url == "http://gateway:8091"
    assert settings.officer_only is True
    assert settings.max_tokens == 8192
    assert settings.listen_channel_ids == frozenset()
    assert settings.control_channel_ids == frozenset()
    assert settings.conversation_lease_seconds == 120
    with pytest.raises(FrozenInstanceError):
        settings.officer_only = False


def test_listening_requires_explicit_valid_channel_ids(load):
    assert load({"listen_channel_ids": [20], "conversation_lease_seconds": 90}).listen_channel_ids == frozenset({20})
    for channels in (None, "all", [True], [0], ["20"]):
        with pytest.raises(ValueError, match="listen_channel_ids"):
            load({"listen_channel_ids": channels})
    for duration in (None, True, 29, 601):
        with pytest.raises(ValueError, match="conversation_lease_seconds"):
            load({"conversation_lease_seconds": duration})
    assert load({"control_channel_ids": [21]}).control_channel_ids == frozenset({21})
    for channels in (None, "all", [True], [0], ["21"]):
        with pytest.raises(ValueError, match="control_channel_ids"):
            load({"control_channel_ids": channels})


@pytest.mark.parametrize("key", ["allowed_guild_ids", "officer_role_ids", "owner_user_ids"])
@pytest.mark.parametrize("values", [None, "10", {}, [True], [0], [-1], ["10"], [2**63]])
def test_malformed_identity_lists_rejected(load, key, values):
    with pytest.raises(ValueError):
        load({key: values})


def test_required_guild_and_authority_allowlists(load):
    with pytest.raises(ValueError):
        load({"allowed_guild_ids": []})
    with pytest.raises(ValueError):
        load({"officer_role_ids": [], "owner_user_ids": []})


@pytest.mark.parametrize("token", [None, "", "short", "x" * 31])
def test_required_runner_secret(load, monkeypatch, token):
    if token is None:
        monkeypatch.delenv("PETERBOT_RUNNER_TOKEN")
    else:
        monkeypatch.setenv("PETERBOT_RUNNER_TOKEN", token)
    with pytest.raises(ValueError, match="PETERBOT_RUNNER_TOKEN"):
        load()


@pytest.mark.parametrize("key", ["runner_url", "tool_service_url"])
@pytest.mark.parametrize("value", ["", "ftp://host", "http:///path", "http://u:p@host", "http://host?q=1", "http://host#x", "http://host:abc", "http://host:65536", None, 123, []])
def test_malformed_service_urls_rejected(load, key, value):
    with pytest.raises(ValueError):
        load({key: value})


@pytest.mark.parametrize("raw", [[], "settings", 42, True])
def test_top_level_settings_must_be_object(load, raw):
    with pytest.raises(ValueError):
        load(raw=raw)


@pytest.mark.parametrize("value", [None, 123, "", "relative/path"])
def test_state_dir_requires_absolute_nonempty_path(load, value):
    with pytest.raises(ValueError):
        load({"state_dir": value})


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_pilot_switch_requires_boolean(load, value):
    with pytest.raises(ValueError):
        load({"officer_only": value})


@pytest.mark.parametrize("key,lower,upper", [
    ("max_iterations", 1, 60), ("max_tokens", 1024, 16384),
    ("max_model_calls", 1, 80), ("max_job_output_tokens", 8192, 262144),
    ("max_tool_calls", 1, 200), ("job_timeout", 60, 1800),
])
def test_limits_are_bounded_integers(load, key, lower, upper):
    for value in (lower - 1, upper + 1, True, "10", None):
        with pytest.raises(ValueError):
            load({key: value})
    assert getattr(load({key: lower}), key) == lower
    assert getattr(load({key: upper}), key) == upper
