"""Generated deployment settings must pass the actual application's validation."""
import json
from pathlib import Path

from deploy.prepare_hermes_config import prepare
from peterbot.config import AppConfig


def test_prepared_config_validates_and_removes_static_roster(tmp_path, monkeypatch):
    monkeypatch.setenv('DISCORD_TOKEN','test-only')
    source=Path(__file__).parents[1]/'config.json'
    persona=Path(__file__).parents[1]/'deploy/peter-persona.md'
    output=tmp_path/'config.json'
    prepare(source,output,persona)
    settings=AppConfig.load(str(output))
    assert settings.inference.extra_request_body['chat_template_kwargs']['enable_thinking'] is True
    assert settings.agent.request_timeout_seconds==120
    assert settings.agent.max_total_tokens==8192
    assert 'President:** Oliver' not in settings.peter_system_prompt
    assert settings.inference.base_url==json.loads(source.read_text())['inference']['base_url'].rstrip('/')
