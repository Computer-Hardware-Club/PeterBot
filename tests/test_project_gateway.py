"""Gateway→runner→project handoff with durable continuation and partial output."""
import asyncio
import base64
import hashlib

from peterbot.agent_policy import Principal
from test_hermes_gateway import ControlledRunner, task_gateway


SOURCE = b'fn main() { println!("42"); }'


def project_file(name='src/main.rs', data=SOURCE):
    return {'name': name, 'data_base64': base64.b64encode(data).decode(),
            'sha256': hashlib.sha256(data).hexdigest()}


class RecordingRunner(ControlledRunner):
    def __init__(self):
        super().__init__()
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return super().post(url, **kwargs)


async def drain(gateway):
    while gateway.active:
        await asyncio.sleep(0.01)


def test_generated_source_is_saved_and_restored_for_private_followup(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            runner = RecordingRunner()
            runner.result = {'status': 'completed', 'answer': 'Compiled and tested.',
                             'artifacts': [{'name': 'peter-artifacts.zip', 'data_base64': 'eA=='}],
                             'project_files': [project_file()]}
            gateway.session = runner
            runner.release.set()
            first = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                        source_message_id=30, prompt='Make a Rust CLI')
            await gateway.queue_tick()
            await drain(gateway)
            saved = gateway.jobs.get(first['id'])
            assert saved['status'] == 'completed' and saved['project_id']
            principal = Principal(10, 1, 21, (100,))
            assert gateway.projects.read_file(principal, saved['project_id'], 'src/main.rs') == SOURCE
            assert gateway.projects.list_files(principal, saved['project_id'])['state'] == 'verified'
            followup = await gateway.submit(guild_id=10, user_id=1, channel=gateway.thread,
                source_message_id=31, prompt='Add a test', parent_id=first['id'])
            assert followup['project_id'] == saved['project_id']
            await gateway.queue_tick()
            await drain(gateway)
            run_requests = [args['json']['request'] for url, args in runner.calls if url.endswith('/run')]
            assert run_requests[-1]['project_files']['files'][0]['name'] == 'src/main.rs'
            assert run_requests[-1]['project_files']['files'][0]['sha256'] == hashlib.sha256(SOURCE).hexdigest()

    asyncio.run(scenario())


def test_cancelled_worker_keeps_unverified_partial_files_before_delivery(tmp_path):
    async def scenario():
        async with task_gateway(tmp_path) as gateway:
            gateway.thread.released.set()
            runner = RecordingRunner()
            runner.result = {'status': 'completed', 'answer': 'late completion',
                             'artifacts': [], 'project_files': [project_file('src/partial.rs')]}
            gateway.session = runner
            job = gateway.jobs.create(guild_id=10, user_id=1, channel_id=21,
                                      source_message_id=30, prompt='Long Rust build')
            await gateway.queue_tick()
            await asyncio.sleep(0)
            assert job['id'] in gateway.active
            await gateway.cancel(job['id'], 10, 1)
            assert gateway.jobs.get(job['id'])['status'] == 'cancelled'
            # A follow-up must wait until the runner has returned its salvage.
            try:
                await gateway.submit(guild_id=10, user_id=1, channel=gateway.thread,
                    source_message_id=32, prompt='Continue', parent_id=job['id'])
            except ValueError as exc:
                assert 'still stopping' in str(exc)
            else:
                raise AssertionError('Continuation started before worker cleanup')
            runner.release.set()
            await drain(gateway)
            stored = gateway.jobs.get(job['id'])
            assert stored['status'] == 'cancelled' and stored['project_id']
            principal = Principal(10, 1, 21, (100,))
            assert gateway.projects.list_files(principal, stored['project_id'])['state'] == 'partial'
            assert gateway.projects.read_file(principal, stored['project_id'], 'src/partial.rs') == SOURCE
            await gateway.queue_tick()
            assert gateway.jobs.get(job['id'])['delivered'] == 1

    asyncio.run(scenario())
