import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from peterbot.agent_policy import Principal
from peterbot.conversation import reply_or_use_tools
from test_hermes_gateway import UpstreamSession


def run(response, context=None):
    session=UpstreamSession()
    session.result={'choices':[{'message':response}]}
    config=SimpleNamespace(peter_system_prompt='You are Peter.',inference=SimpleNamespace(base_url='http://model/v1',model='qwen'),llama_cpp_api_key='private-key')
    result=asyncio.run(reply_or_use_tools(session,config,Principal(10,1,20,(100,)),'hi',context or []))
    return result,session.calls[0]


def test_banter_returns_text_without_sandbox_and_skips_reasoning():
    result,(url,args)=run({'content':'Only on Tuesdays.'},[{'created_at':datetime.now(timezone.utc),'content':'toaster?'}])
    assert result=='Only on Tuesdays.'
    assert url=='http://model/v1/chat/completions'
    assert args['json']['chat_template_kwargs']['enable_thinking'] is False
    assert args['json']['max_tokens']==2048
    assert args['json']['tools'][0]['function']['name']=='use_tools'
    assert args['headers']['Authorization']=='Bearer private-key'
    assert 'private-key' not in str(args['json'])


def test_tool_handoff_returns_no_premature_answer():
    result,_=run({'content':'I will start a big project!', 'tool_calls':[{'function':{'name':'use_tools','arguments':'{"reason":"Need tools"}'}}]})
    assert result is None


@pytest.mark.parametrize('message',[
    {'content':''},
    {'tool_calls':[{'function':{'name':'shell','arguments':'{"reason":"Need tools"}'}}]},
    {'tool_calls':[{'function':{'name':'use_tools','arguments':'{"user_id":2}'}}]},
])
def test_unknown_or_malformed_decisions_fail_without_actions(message):
    with pytest.raises(ValueError):
        run(message)


@pytest.mark.parametrize('text',[
    'Here: file:///workspace/artifacts/results.txt',
    'Here: [download](/workspace/artifacts/results.txt)',
    'Here: /workspace/artifacts/results.txt',
])
def test_deliverables_use_attachment_names_not_sandbox_links(text):
    from peterbot.hermes_gateway import attachment_answer
    assert attachment_answer(text,'[{"name":"results.txt"}]')=='Here: `results.txt`'
