"""Generate upgraded bot JSON without printing endpoints or credentials.

The generated file is a deployment artifact: paths in it are resolved by the container
that will read it, not by whoever ran this script. Relative knowledge/profile paths are
therefore rewritten under --container-root (the app's working directory inside the image).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Deployed model settings. Keep these in step with the config actually running in
# production: a regeneration must not quietly revert a reliability fix.
INFERENCE_TIMEOUT_SECONDS = 180
INFERENCE_MAX_TOKENS = 4096
AGENT_MAX_TOTAL_TOKENS = 8192
AGENT_REQUEST_TIMEOUT_SECONDS = 240
AGENT_MAX_CONCURRENT = 2


def prepare(source: Path, output: Path, persona: Path, container_root: Path = Path('/app')) -> None:
    config=json.loads(source.read_text())
    config['persona']['system_prompt']=persona.read_text().strip()
    inference=config['inference']
    inference['extra_request_body']={**inference.get('extra_request_body',{}),
        'chat_template_kwargs':{'enable_thinking':True}}
    inference['max_tokens']=INFERENCE_MAX_TOKENS
    inference['timeout_seconds']=INFERENCE_TIMEOUT_SECONDS
    config.setdefault('agent',{}).update(max_total_tokens=AGENT_MAX_TOTAL_TOKENS,
        request_timeout_seconds=AGENT_REQUEST_TIMEOUT_SECONDS,max_concurrent=AGENT_MAX_CONCURRENT)
    paths=config.setdefault('paths',{})
    for key in ('knowledge_file','channel_profiles_file'):
        raw=paths.get(key)
        if raw and not Path(raw).is_absolute():
            paths[key]=str(container_root / raw)
    output.write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--persona',type=Path,default=Path(__file__).with_name('peter-persona.md'))
    parser.add_argument('--container-root',type=Path,default=Path('/app'),
        help='Directory the generated paths must resolve from inside the deployment image')
    args=parser.parse_args()
    prepare(args.source,args.output,args.persona,args.container_root)
