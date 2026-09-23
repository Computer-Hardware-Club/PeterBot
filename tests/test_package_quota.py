"""Concurrent package fetches share one task's byte budget."""
import asyncio
import hashlib
from types import SimpleNamespace

from peterbot.package_access import PACKAGE_TASK_BYTES
from test_hermes_gateway import capability, gateway_client, headers


def test_concurrent_fetches_cannot_spend_the_same_remaining_bytes(tmp_path):
    async def scenario():
        async with gateway_client(tmp_path) as (gateway, client):
            payload = b'x' * 1024

            class SlowBroker:
                calls = 0

                async def serve(self, _request, *, quota):
                    self.calls += 1
                    assert quota == len(payload)
                    await asyncio.sleep(0.05)
                    return SimpleNamespace(
                        data=payload, sha256=hashlib.sha256(payload).hexdigest(),
                        filename='pinned.whl', source='image_cache', index_line='')

            broker = SlowBroker()
            gateway.packages = broker
            cap = capability(gateway)
            cap.package_bytes = PACKAGE_TASK_BYTES - len(payload)
            body = {'registry': 'pypi', 'name': 'six', 'version': '1.17.0'}
            responses = await asyncio.gather(*(
                client.post('/package', headers=headers(), json=body) for _ in range(2)))
            assert sorted(response.status for response in responses) == [200, 429]
            assert broker.calls == 1
            assert cap.package_bytes == PACKAGE_TASK_BYTES
    asyncio.run(scenario())
