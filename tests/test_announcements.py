import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from peterbot.announcements import draft_announcement


def test_draft_uses_only_public_supplied_context_and_has_no_tools():
    config = SimpleNamespace(peter_system_prompt='You are Peter.', inference=SimpleNamespace(model='qwen'))
    result = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'what are you building?'}}]}

    async def scenario():
        with patch('peterbot.announcements._post', new=AsyncMock(return_value=result)) as model:
            content = await draft_announcement(object(), config, request='a project check-in',
                                               channel_name='general', public_facts='Meet in room 1005.')
            assert content == 'what are you building?'
            payload = model.await_args.args[2]
            assert 'tools' not in payload
            assert 'room 1005' in payload['messages'][1]['content']
            assert payload['chat_template_kwargs']['enable_thinking'] is False
    asyncio.run(scenario())


@pytest.mark.parametrize('finish,content,tool_calls', [
    ('length', 'truncated post', None), ('stop', '', None),
    ('stop', '@everyone hello', None), ('stop', 'hello', [{'name': 'send_message'}]),
    ('stop', 'a' * 1801, None),
])
def test_unusable_draft_is_never_published(finish, content, tool_calls):
    config = SimpleNamespace(peter_system_prompt='You are Peter.', inference=SimpleNamespace(model='qwen'))
    result = {'choices': [{'finish_reason': finish, 'message': {'content': content, 'tool_calls': tool_calls}}]}

    async def scenario():
        with patch('peterbot.announcements._post', new=AsyncMock(return_value=result)):
            with pytest.raises(ValueError):
                await draft_announcement(object(), config, request='hello', channel_name='general')
    asyncio.run(scenario())
