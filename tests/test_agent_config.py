import json
from pathlib import Path

import pytest

from peterbot.config import AppConfig


def write_config(tmp_path, monkeypatch, updates):
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    data = json.loads((Path(__file__).parents[1] / "config.json").read_text())
    data["paths"] = {"data_dir": str(tmp_path / "data")}
    data["agent"].update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("updates", [
    {"max_concurrent": 3}, {"max_tool_calls": 9}, {"max_tool_rounds": 5},
    {"max_total_tokens": 1000000}, {"request_timeout_seconds": 121},
    {"max_prompt_chars": 99999}, {"allowed_guild_ids": [True]},
    {"allowed_guild_ids": [-1]}, {"allow_dms": "false"},
    {"search_base_url": "file:///etc/passwd"}, {"max_tool_calls": True},
])
def test_agent_configuration_rejects_unsafe_limits(tmp_path, monkeypatch, updates):
    with pytest.raises(ValueError):
        AppConfig.load(str(write_config(tmp_path, monkeypatch, updates)))


def test_production_configuration_keeps_tools_inside_club(tmp_path, monkeypatch):
    config = AppConfig.load(str(write_config(tmp_path, monkeypatch, {"allowed_guild_ids": [123]})))
    assert config.agent.enabled
    assert config.agent.allowed_guild_ids == (123,)
    assert not config.agent.allow_dms
    assert config.agent.max_concurrent == 1
    assert config.inference.model == "Qwen3.8-27B"


def test_legacy_configuration_does_not_enable_tools(tmp_path, monkeypatch):
    path = write_config(tmp_path, monkeypatch, {})
    data = json.loads(path.read_text())
    del data["agent"]
    path.write_text(json.dumps(data))
    assert not AppConfig.load(str(path)).agent.enabled
