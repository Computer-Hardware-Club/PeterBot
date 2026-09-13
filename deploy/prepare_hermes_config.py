"""Generate upgraded bot JSON without printing endpoints or credentials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(source: Path, output: Path, persona: Path) -> None:
    config=json.loads(source.read_text())
    config['persona']['system_prompt']=persona.read_text().strip()
    inference=config['inference']
    inference['extra_request_body']={**inference.get('extra_request_body',{}),
        'chat_template_kwargs':{'enable_thinking':True}}
    inference['max_tokens']=4096
    inference['timeout_seconds']=120
    config.setdefault('agent',{}).update(max_total_tokens=12288,request_timeout_seconds=180)
    output.write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--persona',type=Path,default=Path(__file__).with_name('peter-persona.md'))
    args=parser.parse_args()
    prepare(args.source,args.output,args.persona)
