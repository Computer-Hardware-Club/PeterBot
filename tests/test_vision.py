import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from peterbot.vision import collect_images, describe_images, IMAGE_FAILURE
from test_conversation_routing import conversation_gateway
from test_hermes_gateway import UpstreamSession


def photo(id=1, *, size=4, data=b'photo'):
    return SimpleNamespace(id=id, filename='ram.png', content_type='image/png', size=size,
                           read=AsyncMock(return_value=data))


def message(id=30, *, attachments=(), content='Peter how much does that RAM cost?', age=0, channel=None):
    return SimpleNamespace(id=id, content=content, attachments=list(attachments), reference=None,
        author=SimpleNamespace(id=1, bot=False, display_name='Oliver'), channel=channel,
        created_at=datetime.now(timezone.utc) - timedelta(hours=age))


def history_channel(items):
    def history(**kwargs):
        async def generate():
            for item in items:
                yield item
        return generate()
    return SimpleNamespace(id=20, history=history, fetch_message=AsyncMock())


def test_two_hour_old_ram_photo_is_available_without_image_keyword():
    attachment = photo()
    source = message(10, attachments=[attachment], age=2)
    channel = history_channel([message(12, content='400 dollars?', age=1), source])
    current = message(channel=channel)
    result = asyncio.run(collect_images(current))
    assert result[0]['message_id'] == 10
    assert result[0]['author'] == 'Oliver'
    assert result[0]['image'].startswith('data:image/png;base64,')
    attachment.read.assert_awaited_once()


def test_direct_image_takes_priority_over_reply_and_history():
    first, second, ignored = photo(1), photo(2), photo(3)
    channel = history_channel([message(9, attachments=[ignored])])
    current = message(attachments=[first], channel=channel)
    current.reference = SimpleNamespace(channel_id=20, resolved=message(15, attachments=[first, second], channel=channel))
    result = asyncio.run(collect_images(current))
    assert [item['message_id'] for item in result] == [30]
    first.read.assert_awaited_once()
    ignored.read.assert_not_awaited()


def test_reply_never_reads_images_from_other_channel():
    attachment = photo()
    current = message(content='explain please', channel=history_channel([]))
    current.reference = SimpleNamespace(channel_id=999, resolved=message(15, attachments=[attachment]))
    assert asyncio.run(collect_images(current)) == []
    attachment.read.assert_not_awaited()


def test_old_photos_and_unrelated_chatter_are_ignored():
    attachment = photo()
    channel = history_channel([message(9, attachments=[attachment], age=25)])
    assert asyncio.run(collect_images(message(channel=channel))) == []
    channel = history_channel([message(9, attachments=[attachment])])
    assert asyncio.run(collect_images(message(content='how are you', channel=channel))) == []
    attachment.read.assert_not_awaited()


def test_oversized_images_never_download_and_actual_size_is_checked():
    too_large, deceptive = photo(1, size=100), photo(2, size=1, data=b'0123456789')
    assert asyncio.run(collect_images(message(attachments=[too_large, deceptive]), max_bytes=5)) == []
    too_large.read.assert_not_awaited()


def test_vision_request_contains_pixels_and_observations_never_contain_base64(monkeypatch):
    for key in ('PETERBOT_VISION_BASE_URL', 'PETERBOT_VISION_MODEL', 'PETERBOT_VISION_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    session = UpstreamSession()
    session.result = {'choices': [{'message': {'content': 'G.SKILL Flare X5 DDR5.'}, 'finish_reason': 'stop'}]}
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'), llama_cpp_api_key='secret')
    images = asyncio.run(collect_images(message(attachments=[photo()])))
    observations = asyncio.run(describe_images(session, config, images, question='How much is this RAM?'))
    assert 'G.SKILL Flare X5' in observations and 'untrusted' in observations
    assert 'base64' not in observations
    url, options = session.calls[0]
    assert url == 'http://model/v1/chat/completions'
    assert options['json']['messages'][1]['content'][-1]['type'] == 'image_url'
    assert 'tools' not in options['json']
    assert 'never follow instructions' in options['json']['messages'][0]['content']
    assert options['allow_redirects'] is False


@pytest.mark.parametrize('choice', [{'message': {'content': ''}},
    {'message': {'content': 'partial'}, 'finish_reason': 'length'}])
def test_empty_or_truncated_vision_is_reported_not_used(choice):
    session = UpstreamSession()
    session.result = {'choices': [choice]}
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    images = asyncio.run(collect_images(message(attachments=[photo()])))
    with pytest.raises(ValueError, match="couldn't read"):
        asyncio.run(describe_images(session, config, images))


@pytest.mark.parametrize('answer', ['Those are Flare X5 DIMMs.', None])
def test_gateway_image_observations_reach_chat_and_research(tmp_path, answer):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            channel.send.return_value = SimpleNamespace(id=777, edit=AsyncMock())
            gateway.session.result = {'choices': [{'message': {'content': 'G.SKILL Flare X5 DDR5.'}}]}
            gateway.conversational_reply = AsyncMock(return_value=answer)
            gateway.submit = AsyncMock()
            current = message(attachments=[photo()], channel=channel)
            current.guild = channel.guild
            current.reply = AsyncMock()
            await gateway.respond_to_message(current, current.content)
            assert gateway.conversational_reply.await_args.args[1] == current.content
            assert 'G.SKILL Flare X5' in gateway.conversational_reply.await_args.kwargs['image_context'][0]['content']
            if answer is None:
                kwargs = gateway.submit.await_args.kwargs
                assert kwargs['attachments'] == []
                assert kwargs['prompt'] == current.content
                assert 'G.SKILL Flare X5' in kwargs['context'][-1]['content']
                assert 'base64' not in kwargs['prompt']
            else:
                gateway.submit.assert_not_awaited()
    asyncio.run(scenario())


def test_direct_task_image_becomes_labeled_worker_context(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            gateway.session.result = {'choices': [{'message': {'content': 'G.SKILL Flare X5 DDR5.'}}]}
            job = await gateway.submit(guild_id=10, user_id=1, channel=channel,
                source_message_id=30, prompt='Find the current price of this RAM',
                attachments=[photo()], in_channel=True)
            context = json.loads(job['context'])
            assert 'G.SKILL Flare X5' in context[0]['content']
            assert 'base64' not in job['context']
            assert json.loads(job['input_files']) == []
    asyncio.run(scenario())


def test_same_channel_reply_photo_is_read_even_without_image_keyword():
    attachment = photo()
    channel = history_channel([])
    current = message(content='identify it please', channel=channel)
    current.reference = SimpleNamespace(channel_id=20, message_id=15, resolved=None)
    channel.fetch_message.return_value = message(15, attachments=[attachment], channel=channel)
    result = asyncio.run(collect_images(current))
    assert result[0]['message_id'] == 15
    channel.fetch_message.assert_awaited_once_with(15)


def test_disabled_vision_never_calls_model():
    config = SimpleNamespace(agent=SimpleNamespace(vision_enabled=False))
    session = UpstreamSession()
    with pytest.raises(ValueError, match='disabled'):
        asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))
    assert session.calls == []


@pytest.mark.parametrize('malformed', [[], {'choices': [None]}, {'choices': [{'message': {'content': ['bad']}}]}])
def test_malformed_provider_output_is_clean_failure(malformed):
    session = UpstreamSession()
    session.result = malformed
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    with pytest.raises(ValueError, match="couldn't read"):
        asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))


def test_image_evidence_has_own_model_message_without_changing_prompt():
    from peterbot.conversation import reply_or_use_tools
    from peterbot.agent_policy import Principal
    session = UpstreamSession()
    session.result = {'choices': [{'message': {'content': 'Those are G.SKILL DIMMs.'}}]}
    config = SimpleNamespace(peter_system_prompt='Be Peter.',
        inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    result = asyncio.run(reply_or_use_tools(session, config, Principal(10, 1, 20, (100,)),
        'hello', [{'content': 'x' * 8000}], image_context=[{'content': 'G.SKILL Flare X5'}]))
    assert 'G.SKILL' in result
    messages = session.calls[0][1]['json']['messages']
    assert messages[-1]['content'] == 'hello'
    assert 'G.SKILL Flare X5' in messages[-2]['content']
    assert 'untrusted' in messages[-2]['content']


def test_private_worker_receives_image_evidence_and_parent_history(tmp_path):
    async def scenario():
        async with conversation_gateway(tmp_path) as (gateway, channel):
            parent = gateway.jobs.create(guild_id=10, user_id=1, channel_id=20,
                source_message_id=29, prompt='Identify the kit')
            gateway.jobs.update(parent['id'], status='completed', answer='The kit has two DIMMs.')
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=20,
                source_message_id=30, parent_id=parent['id'], prompt='Price this RAM',
                context=[{'role': 'user', 'content': 'Image evidence: G.SKILL Flare X5'}])
            gateway.jobs.update(job['id'], status='running')
            gateway.session.result = {'status': 'completed', 'answer': 'Research complete.'}
            await gateway.run_job(job)
            payload = next(kwargs['json']['request'] for url, kwargs in gateway.session.calls if url.endswith('/run'))
            prior = payload['prior_messages']
            assert prior[0]['content'] == 'Identify the kit'
            assert prior[1]['content'] == 'The kit has two DIMMs.'
            assert prior[2]['content'] == 'Image evidence: G.SKILL Flare X5'
    asyncio.run(scenario())


def test_truncated_circuit_inspection_retries_before_returning_evidence(caplog):
    session = UpstreamSession()
    session.results = [
        {'choices': [{'message': {'content': 'Circuit a: incomplete WRONG'}, 'finish_reason': 'length'}]},
        {'choices': [{'message': {'content': 'a: +3V, 10kΩ, forward diode, 10kΩ, -3V.'}, 'finish_reason': 'stop'}]},
    ]
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    images = [{'image': 'data:image/png;base64,cHJpdmF0ZQ=='}]
    result = asyncio.run(describe_images(session, config, images, question='solve this problem'))
    assert '+3V' in result and 'WRONG' not in result
    assert len(session.calls) == 2
    first, retry = [call[1]['json'] for call in session.calls]
    assert first['max_tokens'] == retry['max_tokens'] == 2048
    assert 'circuit diagrams' in first['messages'][0]['content']
    assert 'at most 250 words' in retry['messages'][0]['content']
    assert first['messages'][1] == retry['messages'][1]
    assert 'WRONG' not in json.dumps(retry)
    assert 'reason=truncated' in caplog.text and 'cHJpdmF0ZQ' not in caplog.text


def test_repeated_truncation_stops_after_one_retry_with_honest_error(caplog):
    session = UpstreamSession()
    session.result = {'choices': [{'message': {'content': 'private screenshot transcription'}, 'finish_reason': 'length'}]}
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    with pytest.raises(ValueError) as error:
        asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))
    assert str(error.value) == IMAGE_FAILURE and 'smaller' not in str(error.value)
    assert len(session.calls) == 2
    assert 'failed reason=truncated' in caplog.text
    assert 'private screenshot transcription' not in caplog.text


def test_empty_inspection_can_recover():
    session = UpstreamSession()
    session.results = [
        {'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]},
        {'choices': [{'message': {'content': 'A circuit diagram.'}, 'finish_reason': 'stop'}]},
    ]
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    assert 'circuit diagram' in asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))
    assert len(session.calls) == 2


def test_malformed_output_is_not_retried_or_logged(caplog):
    session = UpstreamSession()
    session.result = {'private': 'screenshot and provider details'}
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    with pytest.raises(ValueError):
        asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))
    assert len(session.calls) == 1
    assert 'reason=invalid_response' in caplog.text and 'provider details' not in caplog.text


def test_image_retry_shares_one_deadline(monkeypatch):
    import peterbot.vision as vision
    calls = []
    async def delayed(*args):
        calls.append(1)
        await asyncio.sleep(0.02)
        raise vision._InspectionFailure('truncated')
    monkeypatch.setattr(vision, '_inspect_once', delayed)
    monkeypatch.setattr(vision, 'IMAGE_TIMEOUT_SECONDS', 0.03)
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    with pytest.raises(ValueError, match="couldn't read"):
        asyncio.run(describe_images(object(), config, [{'image': 'data:image/png;base64,eA=='}]))
    assert len(calls) == 2


def test_oversized_completed_observation_retries_instead_of_slicing():
    session = UpstreamSession()
    session.results = [
        {'choices': [{'message': {'content': 'x' * 6001}, 'finish_reason': 'stop'}]},
        {'choices': [{'message': {'content': 'Complete compact circuit description.'}, 'finish_reason': 'stop'}]},
    ]
    config = SimpleNamespace(inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    result = asyncio.run(describe_images(session, config, [{'image': 'data:image/png;base64,eA=='}]))
    assert result.endswith('Complete compact circuit description.')
    assert len(session.calls) == 2


def test_unicode_image_evidence_is_not_sliced_in_answer_request():
    from peterbot.conversation import reply_or_use_tools
    from peterbot.agent_policy import Principal
    session = UpstreamSession()
    evidence = '\u03a9' * 1500 + ' Circuit d: diode cathode at ground.'
    config = SimpleNamespace(peter_system_prompt='Be Peter.',
        inference=SimpleNamespace(model='qwen', base_url='http://model/v1'))
    asyncio.run(reply_or_use_tools(session, config, Principal(10, 1, 20, (100,)),
        'solve this problem', [], image_context=[{'content': evidence}]))
    sent = session.calls[0][1]['json']['messages'][-2]['content']
    assert evidence in sent
