import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from peterbot.agent_policy import Principal
from peterbot.conversation import BLANK_ANSWER_REPLY, reply_or_use_tools
from peterbot.knowledge import KnowledgeChunk
from test_hermes_gateway import UpstreamSession


def config(**inference):
    settings = SimpleNamespace(base_url='http://model/v1', model='qwen', max_tokens=4096, timeout_seconds=180)
    for key, value in inference.items():
        setattr(settings, key, value)
    return SimpleNamespace(peter_system_prompt='You are Peter.', inference=settings, llama_cpp_api_key='private-key')


def run(response, context=None, *, knowledge_chunks=(), **inference):
    session = UpstreamSession()
    session.result = {'choices': [{'message': response}]}
    result = asyncio.run(reply_or_use_tools(session, config(**inference), Principal(10, 1, 20, (100,)), 'hi',
                                           context or [], knowledge_chunks=knowledge_chunks))
    return result, session.calls


def test_banter_returns_text_without_sandbox_and_preserves_thinking():
    result, calls = run({'content': 'Only on Tuesdays.'}, [{'created_at': datetime.now(timezone.utc), 'content': 'toaster?'}])
    assert result == 'Only on Tuesdays.'
    url, args = calls[0]
    assert url == 'http://model/v1/chat/completions'
    assert args['json']['chat_template_kwargs']['enable_thinking'] is True
    # Thinking is billed against this budget, so it must leave room for the answer.
    assert args['json']['max_tokens'] == 4096
    assert args['json']['tools'][0]['function']['name'] == 'use_tools'
    assert args['headers']['Authorization'] == 'Bearer private-key'
    assert 'private-key' not in str(args['json'])
    assert len(calls) == 1


def test_low_budget_is_raised_to_leave_room_for_thinking():
    _, calls = run({'content': 'Sure.'}, max_tokens=1024)
    assert calls[0][1]['json']['max_tokens'] == 4096


def test_tool_handoff_returns_no_premature_answer():
    result, _ = run({'content': 'I will start a big project!',
                     'tool_calls': [{'function': {'name': 'use_tools', 'arguments': '{"reason":"Need tools"}'}}]})
    assert result is None


def test_blank_answer_is_retried_without_thinking_and_never_errors():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'thought about it'}}]},
                       {'choices': [{'message': {'content': 'KEC 1005, Fridays at 6.'}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == 'KEC 1005, Fridays at 6.'
    assert session.calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is True
    assert session.calls[1][1]['json']['chat_template_kwargs']['enable_thinking'] is False
    assert 'came back with no answer' in session.calls[1][1]['json']['messages'][-1]['content']


def test_truncated_thinking_only_turn_is_retried():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'long thoughts'},
                                     'finish_reason': 'length'}]},
                       {'choices': [{'message': {'content': 'Short answer.'}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == 'Short answer.'
    assert len(session.calls) == 2


@pytest.mark.parametrize('name,arguments', [
    ('terminal', '{}'),
    ('use_tools', '{"user_id":2}'),
    ('use_tools', '{}'),
    ('use_tools', '{"reason":"' + 'x' * 400 + '"}'),
])
def test_invented_or_malformed_tool_decision_hands_off_without_erroring(name, arguments):
    """The fast model may invent a tool name or misuse the handoff. That must not reach
    a member as an error string: the sandbox re-checks authority and only honours its
    own allowlist, so failing toward doing the work is the safe direction."""
    result, calls = run({'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]})
    assert result is None
    assert len(calls) == 1


def test_two_blank_answers_end_in_a_human_reply_not_an_error():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'one'}}]},
                       {'choices': [{'message': {'content': '   '}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == BLANK_ANSWER_REPLY
    assert len(session.calls) == 2


def test_club_knowledge_is_injected_and_outranks_guessing():
    chunks = (KnowledgeChunk(heading='Meetings', body='Fridays 18:00 in KEC 1005.',
                             tokens=('meet', 'kec', '1005')),
              KnowledgeChunk(heading='Membership', body='Join through Ideal Logic with an ONID.',
                             tokens=('join', 'ideal', 'logic')))
    _, calls = run({'content': 'Fridays.'}, knowledge_chunks=chunks)
    system = calls[0][1]['json']['messages'][0]['content']
    assert 'KEC 1005' in system
    assert 'Authoritative club facts' in system


def test_no_knowledge_file_means_no_empty_block():
    _, calls = run({'content': 'Hey.'})
    assert 'Authoritative club facts' not in calls[0][1]['json']['messages'][0]['content']


@pytest.mark.parametrize('text', [
    'Here: file:///workspace/artifacts/results.txt',
    'Here: [download](/workspace/artifacts/results.txt)',
    'Here: /workspace/artifacts/results.txt',
])
def test_deliverables_use_attachment_names_not_sandbox_links(text):
    from peterbot.hermes_gateway import attachment_answer
    assert attachment_answer(text, '[{"name":"results.txt"}]') == 'Here: `results.txt`'
