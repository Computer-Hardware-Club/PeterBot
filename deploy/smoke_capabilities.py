"""Read-only live probes for image inspection, search and announcement drafting.

Run with production configuration inside the gateway container. Optional --image
points to a local photo copied into /tmp; it is sent only to the configured model.
No Discord connection or post is made and no application state is written.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp

from peterbot.announcements import draft_announcement
from peterbot.config import AppConfig
from peterbot.tools import ToolExecutor
from peterbot.vision import describe_images


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--expect-image-text', default='')
    args = parser.parse_args()
    config = AppConfig.load()
    tools = ToolExecutor(config.agent.search_base_url)
    async with aiohttp.ClientSession(trust_env=False) as session:
        try:
            if args.image:
                raw = args.image.read_bytes()
                assert 0 < len(raw) <= config.mention_max_image_bytes, 'image exceeds configured bound'
                mime = mimetypes.guess_type(args.image.name)[0] or 'image/png'
                observations = await describe_images(session, config, [{
                    'message_id': None, 'author': 'live verification', 'filename': 'verification-photo',
                    'image': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode(),
                }], question='Identify the visible hardware brand and product family. Do not guess unreadable labels.')
                assert args.expect_image_text.casefold() in observations.casefold(), 'image identification mismatch'
                print(json.dumps({'check': 'vision', 'status': 'passed', 'model': config.inference.model}), flush=True)
            result = json.loads(await tools.execute('web_search', json.dumps({
                'query': 'G.Skill Flare X5 32GB DDR5 6000 CL36 retail price',
            })))
            assert result.get('provider') == 'exa' and len(result.get('results', [])) >= 2, 'Exa search failed'
            assert any('$' in item['snippet'] for item in result['results']), 'no price evidence'
            print(json.dumps({'check': 'search', 'status': 'passed', 'provider': 'exa',
                              'sources': len(result['results'])}), flush=True)
            source = result['results'][0]['url']
            page = json.loads(await tools.execute('fetch_public_page', json.dumps({'url': source})))
            assert page.get('text') and page.get('status') in ('ok', 'partial'), 'page extraction failed'
            print(json.dumps({'check': 'page', 'status': 'passed', 'provider': page.get('provider', 'direct')}), flush=True)
            draft = await draft_announcement(session, config,
                request='asking what hardware projects people are working on', channel_name='general')
            assert draft and len(draft) <= 1800, 'draft failed'
            print(json.dumps({'check': 'announcement_draft', 'status': 'passed', 'posted': False}), flush=True)
        finally:
            await tools.close()


if __name__ == '__main__':
    asyncio.run(main())
