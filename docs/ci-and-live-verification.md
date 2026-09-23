# CI and live verification

The release history is stacked: bounded web tools first, then the Hermes gateway,
runner, and worker, followed by Peter's foreground conversation and club-member
workflows. The CI workflow validates pushes to `main` and feature branches as
well as pull requests. Passing CI proves deterministic behavior and image
construction; live Discord, Qwen, and VM verification remain separate gates.

## Local deterministic baseline

Use Python 3.12, then run:

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m json.tool config.json >/dev/null
python -m json.tool deploy/hermes.example.json >/dev/null
python -m compileall -q peterbot deploy tests
python -m pytest -q
```

The ordinary suite intentionally skips `tests/test_hermes_runtime_integration.py` when the pinned upstream runtime is absent. This skip is not evidence that the adapter works.

The pull-request workflow builds the gateway, runner, and pinned Hermes worker without production credentials. It runs that integration fixture inside the actual worker image, where a skip must fail because the fixture reports no tests run. It also checks key container isolation properties. The worker's upstream Hermes revision is pinned in `requirements-hermes.txt`; changing it requires adapter and live-backend validation.

The gateway's direct Python dependencies are pinned to the versions observed in the September 22 P910 image (`discord.py` 2.7.1, `aiohttp` 3.14.3, `python-dotenv` 1.2.3). Transitive packages and base-image tags are not yet locked by digest, so a future rebuild still needs a verification run.

## Deployment validation

CI has no Discord token, production secrets, deployment key, or privileged self-hosted runner. On an authorized staging host, run `deploy/smoke_hermes.py`, the production worker isolation checks in `deploy/check_hermes_isolation.py`, a real model turn, and a Discord test-channel exchange. Record the gateway, runner, and worker image revision labels, model identity, tests, failures, and skips. Run only one gateway connected to the bot token. Back up SQLite with its backup API or while the service is stopped; copying a live WAL file alone is insufficient.

`deploy/state_backup.py backup SOURCE DESTINATION` creates a private, checksummed snapshot of the whole state directory, using SQLite's online backup API for live databases. Put `DESTINATION` outside `SOURCE`. Run `verify DESTINATION` before a cutover. With the gateway stopped, `restore DESTINATION EMPTY_PATH` creates a new restore directory; it refuses to overwrite an existing directory. Keep older snapshots outside the appdata state tree so future backups do not recursively include them.
