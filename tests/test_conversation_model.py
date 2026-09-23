import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from peterbot.agent_policy import Principal
from peterbot.config import AppConfig
from peterbot.conversation import (BLANK_ANSWER_REPLY, CASUAL, DEEP, MODEL_UNAVAILABLE_REPLY, NORMAL,
                                   RESCUE_RESERVE_SECONDS, TIER_PROFILES, reply_or_use_tools, select_tier)
from peterbot.knowledge import KnowledgeChunk
from test_hermes_gateway import UpstreamSession


def config(**inference):
    settings = SimpleNamespace(base_url='http://model/v1', model='qwen', max_tokens=4096, timeout_seconds=180)
    for key, value in inference.items():
        setattr(settings, key, value)
    return SimpleNamespace(peter_system_prompt='You are Peter.', inference=settings, llama_cpp_api_key='private-key')


def run(response, prompt='hi', context=None, *, knowledge_chunks=(), **kwargs):
    inference = kwargs.pop('inference', {})
    session = UpstreamSession()
    session.result = {'choices': [{'message': response}]}
    result = asyncio.run(reply_or_use_tools(session, config(**inference), Principal(10, 1, 20, (100,)), prompt,
                                           context or [], knowledge_chunks=knowledge_chunks, **kwargs))
    return result, session.calls


def test_greeting_gets_one_fast_non_thinking_attempt_without_the_4096_allowance():
    result, calls = run({'content': 'Only on Tuesdays.'}, 'hey Peter, what\'s up?',
                        context=[{'created_at': datetime.now(timezone.utc), 'content': 'toaster?'}])
    assert result == 'Only on Tuesdays.'
    url, args = calls[0]
    assert url == 'http://model/v1/chat/completions'
    assert args['json']['chat_template_kwargs']['enable_thinking'] is False
    # A greeting must not carry the deep tier's 4096-token thinking allowance.
    assert args['json']['max_tokens'] == TIER_PROFILES[CASUAL]['budget_tokens'] < 4096
    assert args['json']['tools'][0]['function']['name'] == 'use_tools'
    assert args['headers']['Authorization'] == 'Bearer private-key'
    assert 'private-key' not in str(args['json'])
    assert len(calls) == 1


@pytest.mark.parametrize('prompt', [
    'Build a Rust CLI. Compile and test it in your sandbox, then attach the source files.',
    'Please attach the source and a README for that program.',
])
def test_clean_promise_cannot_replace_requested_execution_or_attachment(prompt):
    result, calls = run({'content': 'Got it — sending the files now.'}, prompt)
    assert result is None
    assert len(calls) == 1


def test_conceptual_coding_question_can_still_get_a_direct_answer():
    result, _calls = run({'content': 'Use cargo build, then cargo test.'},
                         'How do I compile and test a Rust CLI?')
    assert result == 'Use cargo build, then cargo test.'


def test_tier_selection_shapes_generation_not_routing():
    assert select_tier('hey') == CASUAL
    assert select_tier('lmao nice one') == CASUAL
    assert select_tier('explain how PCIe lanes differ from channels') == NORMAL
    assert select_tier('find the current stable Rust release and cite it') == DEEP
    assert select_tier('who is our current president?') == DEEP
    assert select_tier('my code won\'t compile, traceback attached', has_attachments=True) == DEEP
    assert select_tier('how do I wire my Raspberry Pi for undervolting?') == NORMAL
    # Explicit depth wins over the casual shape of the message.
    assert select_tier('quick one: explain the tradeoffs in detail') == DEEP
    # Long or multi-topic messages are not treated as casual even without keywords.
    assert select_tier('so I was thinking about the meetup and maybe we could '
                       'move it later since the room is booked') == NORMAL


def test_serious_research_request_keeps_thinking_for_reliable_tool_routing():
    """Earlier live notes report unreliable tool routing without thinking: deep tiers
    keep thinking on so a genuine research request can reach the sandbox."""
    prompt = 'check the current club meeting schedule and the latest board decision'
    _, calls = run({'tool_calls': [{'function': {'name': 'use_tools', 'arguments': '{"reason":"need the schedule"}'}}]},
                   prompt, inference={'timeout_seconds': 420})
    assert calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is True
    assert calls[0][1]['json']['max_tokens'] == TIER_PROFILES[DEEP]['budget_tokens']


def test_deep_budget_follows_configured_max_tokens_within_bounds():
    _, calls = run({'content': 'Because reasons.'}, 'explain why the kernel panics here',
                   inference={'max_tokens': 1024, 'timeout_seconds': 420})
    # A configured allowance too low to survive thinking is raised to the floor.
    assert calls[0][1]['json']['max_tokens'] == 4096
    _, calls = run({'content': 'Because reasons.'}, 'explain why the kernel panics here',
                   inference={'max_tokens': 5000, 'timeout_seconds': 420})
    assert calls[0][1]['json']['max_tokens'] == 5000


def test_handoff_requires_completion_marker_not_a_truncated_tool_call():
    """finish_reason=length means the arguments may have been cut off mid-call; a
    missing marker is the same risk. Neither may count as tool authorization."""
    for finish in ('length', 'content_filter'):
        session = UpstreamSession()
        session.results = [
            {'choices': [{'message': {'content': '',
                                      'tool_calls': [{'function': {'name': 'use_tools',
                                                                   'arguments': '{"reason":"Need tools"}'}}]},
                          'finish_reason': finish}]},
            {'choices': [{'message': {'content': 'Rust 1.91 is current, from the release page.'}}]},
        ]
        result = asyncio.run(reply_or_use_tools(
            session, config(), Principal(10, 1, 20, (100,)), 'find the current stable Rust release', []))
        assert result == 'Rust 1.91 is current, from the release page.'
    # Two attempts each, and the second one is the non-thinking rescue.
    assert len(session.calls) == 2
    assert session.calls[1][1]['json']['chat_template_kwargs']['enable_thinking'] is False


@pytest.mark.parametrize('name,arguments', [
    ('terminal', '{}'),
    ('use_tools', '{"user_id":2}'),
    ('use_tools', '{}'),
    ('use_tools', '{"reason":"' + 'x' * 400 + '"}'),
    ('use_tools', '{"reason":"nee'),
])
def test_malformed_tool_decision_never_hands_off(name, arguments):
    """A malformed or invented tool decision is a model wobble, not authorization: the
    turn retries and ends with text, not a sandbox handoff on garbage arguments."""
    session = UpstreamSession()
    session.results = [
        {'choices': [{'message': {'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]},
                      'finish_reason': 'tool_calls'}]},
        {'choices': [{'message': {'content': 'Ask me something concrete.'}}]},
    ]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == 'Ask me something concrete.'
    assert len(session.calls) == 2


def test_blank_answer_is_retried_and_never_errors():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'thought about it'}}]},
                       {'choices': [{'message': {'content': 'KEC 1005, Fridays at 6.'}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == 'KEC 1005, Fridays at 6.'
    assert session.calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is False
    assert session.calls[1][1]['json']['chat_template_kwargs']['enable_thinking'] is False
    assert 'came back with no answer' in session.calls[1][1]['json']['messages'][-1]['content']


def test_truncated_thinking_only_turn_is_rescued_as_plain_text():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'long thoughts'},
                                     'finish_reason': 'length'}]},
                       {'choices': [{'message': {'content': 'Short answer.'}}]}]
    result = asyncio.run(reply_or_use_tools(
        session, config(), Principal(10, 1, 20, (100,)), 'explain how does NVMe naming work', []))
    assert result == 'Short answer.'
    assert len(session.calls) == 2
    assert session.calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is True
    assert session.calls[0][1]['json']['reasoning_effort'] == 'low'
    # The thinking part is capped so the rescue path keeps room to answer.
    assert session.calls[0][1]['json']['chat_template_kwargs']['thinking_budget'] < \
        session.calls[0][1]['json']['max_tokens']
    assert session.calls[1][1]['json']['chat_template_kwargs']['enable_thinking'] is False


def test_reasoning_text_is_never_returned_as_the_answer():
    result, _ = run({'content': '', 'reasoning': 'secret chain of thought about salaries'})
    assert result == BLANK_ANSWER_REPLY
    assert 'secret' not in result
    # An inline think block is stripped from the visible text.
    inline, _ = run({'content': '<think>hidden</think>The meeting is Friday.'})
    assert inline == 'The meeting is Friday.'


def test_two_blank_answers_end_in_a_human_reply_not_an_error():
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'one'}}]},
                       {'choices': [{'message': {'content': '   '}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert result == BLANK_ANSWER_REPLY
    assert len(session.calls) == 2


def test_dropped_stream_falls_back_to_the_rescue_attempt():
    """A stalled or dropped stream must not become a member-facing error while a second
    attempt can still answer."""
    session = UpstreamSession()
    session.fail_times = 1
    session.result = {'choices': [{'message': {'content': 'Recovered.'}}]}
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)),
                                            'explain how does page caching work', []))
    assert result == 'Recovered.'
    assert session.calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is True
    assert session.calls[1][1]['json']['chat_template_kwargs']['enable_thinking'] is False


def test_model_unreachable_on_every_attempt_raises_a_human_message():
    session = UpstreamSession()
    session.fail_times = 5
    with pytest.raises(ValueError) as error:
        asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', []))
    assert str(error.value) == MODEL_UNAVAILABLE_REPLY
    assert 'Traceback' not in str(error.value)
    assert len(session.calls) == 2


def test_conversation_requests_stream_so_a_slow_turn_is_not_a_timeout():
    _, calls = run({'content': 'Slow but alive.'})
    assert calls[0][1]['json']['stream'] is True


def test_budget_reserves_a_slice_for_the_rescue_attempt():
    """A hard question can spend the whole first attempt thinking. The rescue must keep a
    guaranteed slice of the deadline, or the turn ends as a failure line."""
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': '', 'reasoning': 'thinking'}}]},
                       {'choices': [{'message': {'content': 'Answered cheaply.'}}]}]
    result = asyncio.run(reply_or_use_tools(
        session, config(timeout_seconds=420), Principal(10, 1, 20, (100,)),
        'explain how does ZFS checksumming work', []))
    assert result == 'Answered cheaply.'
    assert session.calls[0][1]['timeout'].total == pytest.approx(420 - RESCUE_RESERVE_SECONDS, abs=1)
    assert session.calls[1][1]['timeout'].total <= 120


def test_single_budget_across_attempts_from_caller():
    """The scheduler passes the *remaining* turn budget; the whole turn (both attempts
    plus any retry) must stay inside that one wall-clock figure."""
    session = UpstreamSession()
    session.results = [{'choices': [{'message': {'content': ''}}]},
                       {'choices': [{'message': {'content': 'Two short tries.'}}]}]
    result = asyncio.run(reply_or_use_tools(session, config(timeout_seconds=420),
                                            Principal(10, 1, 20, (100,)), 'hi', [], budget_seconds=40))
    assert result == 'Two short tries.'
    # Both ceilings come from the one shared remaining budget: neither attempt may be
    # granted more wall-clock than the scheduler handed over.
    assert all(call[1]['timeout'].total <= 40 for call in session.calls)


def test_no_oversized_call_starts_near_the_deadline():
    """With 12s left, a 4096-token thinking call cannot plausibly finish; the turn must
    skip it instead of burning the deadline and failing."""
    session = UpstreamSession()
    result = asyncio.run(reply_or_use_tools(
        session, config(), Principal(10, 1, 20, (100,)), 'explain why the kernel panics here', [],
        budget_seconds=12))
    assert session.calls == []
    assert result == BLANK_ANSWER_REPLY


def test_deadline_already_spent_answers_safely_without_calling():
    session = UpstreamSession()
    result = asyncio.run(reply_or_use_tools(session, config(), Principal(10, 1, 20, (100,)), 'hi', [],
                                            budget_seconds=0))
    assert session.calls == []
    assert result == BLANK_ANSWER_REPLY


def test_tool_arguments_split_across_deltas_are_reassembled():
    from peterbot.conversation import _read_stream

    lines = [
        'data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': None,
                                            'delta': {'tool_calls': [{'index': 0, 'id': 'call-1', 'type': 'function',
                                                                      'function': {'name': 'use', 'arguments': ''}}]}}]}),
        'data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': None,
                                            'delta': {'tool_calls': [{'index': 0, 'function': {'name': '_tools'}}]}}]}),
        'data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': None,
                                            'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"reason":'}}]}}]}),
        'data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': None,
                                            'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"needs tools"}'}}]}}]}),
        'data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'delta': {}}]}),
        'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5}, "choices": []}',
        'data: [DONE]',
    ]

    class Stream:
        def __aiter__(self):
            async def generate():
                for line in lines:
                    yield (line + '\n').encode()
            return generate()

    completion = asyncio.run(_read_stream(SimpleNamespace(content=Stream())))
    message = completion['choices'][0]['message']
    assert message['tool_calls'][0]['function']['name'] == 'use_tools'
    assert json.loads(message['tool_calls'][0]['function']['arguments']) == {'reason': 'needs tools'}
    assert completion['choices'][0]['finish_reason'] == 'tool_calls'
    assert completion['usage']['completion_tokens'] == 5


def test_stream_without_a_finish_reason_still_returns_what_arrived():
    from peterbot.conversation import _read_stream

    lines = ['data: ' + json.dumps({'choices': [{'index': 0, 'finish_reason': None,
                                                 'delta': {'content': 'Half an ans'}}]})]

    class Stream:
        def __aiter__(self):
            async def generate():
                for line in lines:
                    yield (line + '\n').encode()
            return generate()

    completion = asyncio.run(_read_stream(SimpleNamespace(content=Stream())))
    assert completion['choices'][0]['message']['content'] == 'Half an ans'
    assert completion['choices'][0]['finish_reason'] is None


def test_long_truncated_answer_survives_as_a_partial_reply():
    """Text the model actually wrote beats a canned failure line; but it must not be
    treated as a clean answer when the rescue also truncates."""
    partial = 'The kernel panics because the driver dereferences a freed page table entry, ' \
              'which the IOMMU reports as a DMA remapping fault.'
    session = UpstreamSession()
    session.results = [
        {'choices': [{'message': {'content': partial}, 'finish_reason': 'length'}]},
        {'choices': [{'message': {'content': partial}, 'finish_reason': 'length'}]},
    ]
    result = asyncio.run(reply_or_use_tools(
        session, config(), Principal(10, 1, 20, (100,)), 'explain why the kernel panics here', []))
    assert result == partial
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


def test_live_club_snapshot_and_style_replace_stale_static_context():
    session = UpstreamSession()
    session.result = {'choices': [{'message': {'content': 'Sam is president.'}}]}
    chunks = (KnowledgeChunk(heading='Officers', body='Old Bob is president.', tokens=('president',)),)
    result = asyncio.run(reply_or_use_tools(
        session, config(), Principal(10, 1, 20, (100,)), 'Who is president?', [],
        knowledge_chunks=chunks, club_context='Current officer roster: Sam is president.',
        style_instruction='Be a little more reserved.'))
    assert result == 'Sam is president.'
    system = session.calls[0][1]['json']['messages'][0]['content']
    assert 'Sam is president' in system and 'Old Bob' not in system
    assert 'Be a little more reserved' in system
    assert 'never changes truthfulness' in system


def test_durable_scoped_context_reaches_the_model_as_untrusted():
    _, calls = run({'content': 'Yeah, still rainy.'}, 'any plans this weekend?',
                   context=[{'role': 'user', 'content': 'it keeps raining'},
                            {'role': 'assistant', 'content': 'classic April'}])
    first_user = calls[0][1]['json']['messages'][1]['content']
    assert 'untrusted' in first_user
    assert 'classic April' in first_user


def test_tier_config_block_is_optional_and_validated(tmp_path):
    from peterbot.config import ConversationConfig

    # No conversation block at all: built-in profiles, existing configs load unchanged.
    assert AppConfig.__dataclass_fields__['conversation'].default_factory() == ConversationConfig(tiers={})
    base = {'persona': {'name': 'Peter', 'system_prompt': 'x', 'model_profile': 'auto'},
            'discord': {'suggestion_channel_id': None},
            'inference': {'base_url': 'http://h:1/v1', 'model': 'qwen', 'timeout_seconds': 60, 'max_tokens': 4096},
            'llama_server': {'enabled': False, 'model_path': None, 'host': 'h', 'port': 1, 'ctx_size': 4096,
                             'threads': 0, 'batch_size': 512, 'parallel': 1, 'continuous_batching': True,
                             'n_gpu_layers': 0, 'metrics': False, 'extra_args': []},
            'paths': {'data_dir': str(tmp_path), 'knowledge_file': None, 'channel_profiles_file': None,
                      'log_file': ''},
            'logging': {'level': 'INFO', 'user_debug_ids_enabled': False, 'include_traceback_for_warning': False},
            'behavior': {}, 'agent': {'enabled': False}}
    path = tmp_path / 'config.json'

    def load(extra):
        path.write_text(json.dumps({**base, **extra}), encoding='utf-8')
        return AppConfig.load(str(path))

    import os
    os.environ['DISCORD_TOKEN'] = 'test-token'
    try:
        assert load({}).conversation.tiers == {}
        configured = load({'conversation': {'tiers': {'casual': {'thinking': False, 'budget_tokens': 256,
                                                                 'temperature': 0.5}}}})
        assert configured.conversation.tiers['casual']['budget_tokens'] == 256
        with pytest.raises(ValueError, match='unknown tier'):
            load({'conversation': {'tiers': {'turbo': {}}}})
        with pytest.raises(ValueError, match='reasoning_effort'):
            load({'conversation': {'tiers': {'normal': {'reasoning_effort': 'insane'}}}})
        with pytest.raises(ValueError, match='budget_tokens'):
            load({'conversation': {'tiers': {'deep': {'budget_tokens': 0}}}})
    finally:
        os.environ.pop('DISCORD_TOKEN', None)


def test_configured_tier_overrides_shape_the_payload():
    settings = SimpleNamespace(base_url='http://model/v1', model='qwen', max_tokens=4096, timeout_seconds=180)
    cfg = SimpleNamespace(peter_system_prompt='You are Peter.', inference=settings, llama_cpp_api_key='',
                          conversation={'tiers': {'casual': {'budget_tokens': 256, 'temperature': 0.9}}})
    session = UpstreamSession()
    session.result = {'choices': [{'message': {'content': 'yo.'}}]}
    result = asyncio.run(reply_or_use_tools(session, cfg, Principal(10, 1, 20, (100,)), 'hey', []))
    assert result == 'yo.'
    assert session.calls[0][1]['json']['max_tokens'] == 256
    assert session.calls[0][1]['json']['temperature'] == 0.9
    # A non-thinking attempt must never carry reasoning_effort: the served vLLM rejects it.
    assert 'reasoning_effort' not in session.calls[0][1]['json']


def test_reasoning_effort_only_ever_accompanies_thinking():
    _, calls = run({'content': 'Because the bus saturates.'}, 'explain how does DMA work')
    assert calls[0][1]['json']['reasoning_effort'] in ('low', 'medium')
    assert calls[0][1]['json']['chat_template_kwargs']['enable_thinking'] is True
    _, calls = run({'content': 'lol ok.'}, 'lol ok')
    assert 'reasoning_effort' not in calls[0][1]['json']


@pytest.mark.parametrize('text', [
    'Here: file:///workspace/artifacts/results.txt',
    'Here: [download](/workspace/artifacts/results.txt)',
    'Here: /workspace/artifacts/results.txt',
])
def test_deliverables_use_attachment_names_not_sandbox_links(text):
    from peterbot.hermes_gateway import attachment_answer
    assert attachment_answer(text, '[{"name":"results.txt"}]') == 'Here: `results.txt`'
