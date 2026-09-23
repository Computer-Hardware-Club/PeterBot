# Dedicated worker VM on P910 (PETER-12 / PETER-13 target topology)

Design and operator recipe. On September 23 the dedicated `peterbot-worker`
domain and `virbr-ctl` network were provisioned on P910. The unrelated desktop
VM was left running. The guest runner is healthy at `192.168.241.2:8780`; the
existing P910 gateway container reached its health endpoint. A disposable
worker compiled and ran the pinned Rust fixture and passed 29 checks before
and after a guest reboot. The broker capability probe was a declared skip
because the final gateway overlay is not yet deployed. Production still uses
the earlier gateway and host runner images. The recipe remains the recovery
path for the pinned Debian generic-cloud `20260909-2596` and Rust `2026-09-03`
inputs.

## Shape

```
P910 (trusted host, stays as-is)                       guest: peterbot-worker
┌───────────────────────────────────────────┐   ┌─────────────────────────────────┐
│ gateway container  (Discord, broker :8770)│   │ Debian 12, docker-ce            │
│   publishes 192.168.241.1:8770 (overlay)  │   │  vNIC2 static 192.168.241.2     │
│ model server / proxy (100.73.210.66:8000) │   │   runner container (:8780 only  │
│ libvirt: desktop VM (untouched)           │   │   bound there; peterbot_control)│
│ bridge virbr-ctl 192.168.241.1/24  (new)  │◄──┤  worker container (transient,   │
│   host-only: no NAT, no DHCP, no routing  │   │   one at a time; pbworkers      │
│                                           │   │   bridge, egress: broker alias  │
│                                           │   │   192.168.240.2:8770 ONLY)      │
└───────────────────────────────────────────┘   └─────────────────────────────────┘
```

- The **gateway, Discord, model access and all state stay on P910**. Only the
  trusted runner supervisor + disposable workers move into the guest.
- **One worker at a time**: the guest compose pins `PETERBOT_RUNNER_CONCURRENCY: 1`
  so the contract holds even if gateway config drifts.
- The desktop VM is never touched: new domain `peterbot-worker`, new network
  `virbr-ctl`. The existing default NAT is used only for the guest's own package
  updates at setup time (detachable afterwards, see posture B).

## Addresses (contract — change all together)

| address | who | why |
|---|---|---|
| `192.168.241.1` | P910 host on `virbr-ctl` | broker `:8770` is published **only** here via the VM overlay; nothing else |
| `192.168.241.2` | guest vNIC2, static on `virbr-ctl` | runner API, published **only** here; also the MASQUERADE source for broker traffic |
| `192.168.242.0/24` (`ctl0`) | guest `peterbot_control` bridge | runner's container network; NO L2 with `pbworkers` |
| `192.168.240.1` | guest on `pbworkers` | bridge gateway address; worker default route, then DROPned |
| `192.168.240.2` | broker alias on `pbworkers` (DNAT) | worker-side target stays identical to the P910 compose design |

Broker ingress is a docker published port on P910 bound to `192.168.241.1`
ONLY (`deploy/vm/compose.hermes-vm-gateway.yml`). The overlay also disables
the old host runner service and detaches the gateway from the old host worker
network; remove the settled host runner at cutover and verify it is stopped.
Trade-off, stated plainly:
this puts a docker-proxy listener on the P910 host stack, but on the
host-only `virbr-ctl` segment — unreachable from any network the worker or an
untrusted process can reach, never `0.0.0.0`, never the tailnet. The alternative
(macvlan-attaching the gateway container to `virbr-ctl`) would need a
`compose.hermes.yml` change owned by the gateway agent; the overlay keeps the
VM rollout additive and one-command reversible.

## Traffic + authentication

1. **Gateway → runner** (`http://192.168.241.2:8780/{health,run,cancel}`):
   `Authorization: Bearer $PETERBOT_RUNNER_TOKEN`, constant-time checked in the
   runner (`peterbot/sandbox_runner.py`). Reachable only from `virbr-ctl`: the
   port is bound to `.241.2`, the guest INPUT wall drops `.240/24`, and
   `virbr-ctl` is host-only (no external L2). Token compromise alone is not
   enough without also being on that segment. mTLS via a tunnel is later
   hardening, not required for pilot.
2. **Worker → gateway broker** (`POST http://192.168.240.2:8770/internal/...`):
   per-job capability token from `/run/secrets/worker_token`, bound server-side
   to the job — identical to the P910 compose design. The guest DNATs
   `.240.2:8770 → 192.168.241.1:8770` and MASQUERADEs the source (P910 has no
   route back into `192.168.240.0/24`); the FORWARD chain allows **only**
   `192.168.240.0/24 → 192.168.241.1:8770` and drops everything else (internet,
   runner, guest hosts, cloud metadata, spoofed sources —
   `deploy/vm/peterbot-vm-firewall.sh`). `br_netfilter` is enabled or bridged
   frames would bypass that wall entirely; verify asserts it.
3. **Runner → worker**: docker API over the **guest-local** socket. P910's
   socket is never visible in the guest, let alone the worker. The runner
   supervises by container name over the socket — it shares no L2 with workers
   (`peterbot_control` vs `pbworkers`), so a compromised runner cannot even
   reach a worker IP; the wall independently denies worker→runner.
4. **Worker → guest/runner services**: blocked twice. The `pbworkers` bridge is
   `--internal`, ICC-off, Docker-masquerade-off (created EXTERNALLY by
   `peterbot-worker-vm-setup.sh` so names/subnets never drift); the only egress
   path is the reviewed firewall's DNAT. An INPUT chain drops `.240/24` from the
   guest stack, so the published runner port on `.241.2` is also unreachable
   from a worker (its DNAT'd destination would be local; INPUT kills it).
5. **Worker → anything else**: denied (`--dns 127.0.0.1`, no host mounts,
   cap-drop ALL, guest egress wall). DNS for model calls never happens
   worker-side; the gateway proxies inference.

A compromised worker can speak capability-authenticated broker protocol to the
gateway and nothing else.

## Images

Built on the trusted build host from pinned inputs: Debian base digest, pip/npm
pins, Hermes `v2026.9.11@939e45c…`, and Rust 1.98.1 with per-arch SHA-256 from
the `static.rust-lang.org/dist/2026-09-03` channel manifest
(`docker/Dockerfile.hermes-worker` header; the build verifies each tarball with
`sha256sum -c` before extraction and fails closed). Transfer is
`docker save` → SSH → digest compare → `docker load`
(`deploy/vm/peterbot-vm-transfer.sh`); the guest holds no registry credentials.
Tag images with the Git revision and record `docker image inspect` digests in
the deployment log.

## Provisioning recipe (operator actions)

All P910-side steps run as the **operator user** (member of `libvirt`/`kvm`/
`docker` — virsh/qemu actions are performed by libvirtd; no sudo, no root
login). One idempotent script; it refuses to run unless KVM,
`qemu:///system`, the `default` network and an operator key are present, and
touches only `peterbot-worker*` assets in `$PETERBOT_VM_DIR`
(default `/mnt/NVME/docker/appdata/peterbot/vm`, group-`kvm` for the qemu
runtime uid) plus `virbr-ctl`.

1. Operator key (auto-generated if absent, private half never printed):
   `ssh-keygen -t ed25519 -f ~/.ssh/peterbot_vm -N ''`
2. Provision the guest network and domain (operator user, from a repo checkout):
   ```sh
   sh deploy/vm/peterbot-worker-vm-provision.sh
   ```
   It verifies the pinned Debian 12 generic-cloud qcow2 (`20260909-2596`,
   SHA-512 pinned in the script, `sha512sum -c` before use), defines/starts
   `virbr-ctl` from `deploy/vm/virbr-ctl.xml`, copies the base to a dedicated
   60 GiB qcow2 (grows only; no backing-file coupling), builds a NoCloud seed
   (user-data + meta-data + a SEPARATE network-config file — the NoCloud
   contract; via `cloud-localds -N`, mkisofs-equivalent fallback), pins both
   vNIC MACs in `$VM_DIR/peterbot-worker.state` (network-config matches BY
   MAC: vNIC2 → static `.241.2`, vNIC1 → DHCP for setup only), stages the guest
   payload to `/opt/peterbot`, and defines/starts/autostarts the domain.
   The VM directory is setgid `kvm` so both the operator and the guest QEMU
   process can access the disk without sudo; the seed explicitly permits
   key-only root SSH inside this dedicated guest.
   Re-running converges (fresh instance-id re-runs first-boot setup; all
   stages idempotent).
3. At the final gateway cutover, once `virbr-ctl` owns `192.168.241.1` and the
   candidate image/configuration are staged, publish the broker on that address
   only (VM gateway overlay, removing the old host runner dependency):
   ```sh
   docker compose -f compose.hermes.yml -f deploy/vm/compose.hermes-vm-gateway.yml up -d --remove-orphans peterbot
   ```
   Until this is applied the broker alias has no live target; the guest smoke
   reports that honestly (SKIP gate). Docker cannot bind `.241.1` before the
   network exists.
4. First boot (~2–4 min on NAT): cloud-init runs
   `peterbot-worker-vm-setup.sh` — apt docker-ce and docker-compose-plugin (Docker's signed repo),
   sshd key-only hardening, `ip_forward` + `br_netfilter` persistence
   (deliberately NOT `ip_nonlocal_bind`), the external networks
   (`peterbot_workers` bridge `pbworkers` internal/ICC-off/masq-off;
   `peterbot_control` bridge `ctl0` — routable because the runner's published
   port needs the DNAT path; workers NEVER join it), the egress-wall systemd
   unit (pre-creates the bare bridge so the broker alias answers ARP before
   dockerd starts) plus a Docker restart hook that reasserts firewall rule
   order, docker + qemu-guest-agent enabled, and
   `python3-aiohttp`/`python3-urllib3` for the in-guest smoke.
   Log: guest `/var/log/peterbot-setup.log`.
5. Stage the candidate runner/worker images on P910 first (build on P910 or
   `docker save | ssh p910 docker load` from the trusted build host), then
   transfer those exact tags into the guest:
   ```sh
   deploy/vm/peterbot-vm-transfer.sh \
     peterbot-hermes-runner:REV peterbot-hermes-worker:REV
   ```
6. Deploy the stack (guest): write a root-only `/opt/peterbot/.env` with
   `PETERBOT_RUNNER_IMAGE`, `PETERBOT_WORKER_IMAGE` (the transferred tags) and
   `PETERBOT_RUNNER_TOKEN` (the same protected token used by the gateway,
   never printed or committed), then
   ```sh
   ssh -i ~/.ssh/peterbot_vm root@192.168.241.2 \
     'cd /opt/peterbot && docker compose -f deploy/vm/compose.hermes-vm.yml --env-file .env up -d \
      && systemctl reload peterbot-vm-firewall'
   ```
   The wall unit installs before docker at boot; the reload re-asserts rule
   order now that dockerd has had its own pass over the chains.
7. Gateway config (handoff, gateway agent): set `runner_url` to
   `http://192.168.241.2:8780` in the gateway's hermes config + same
   `PETERBOT_RUNNER_TOKEN`. No code change; `hermes_gateway` already calls
   this API through `hermes_settings`.
8. Verify:
   - guest posture: `ssh -i ~/.ssh/peterbot_vm root@192.168.241.2 'sh /opt/peterbot/deploy/vm/peterbot-worker-vm-verify.sh'`
     (asserts wall rule position before Docker's rules, br_netfilter, alias ARP,
     DNAT target `.241.1`, INPUT wall, both external networks with pinned
     bridge options and internal/routable posture, `ip_nonlocal_bind=0`,
     runner health **on `.241.2` only** and bound to it).
   - end-to-end: INSIDE the guest (it drives the guest-local docker socket; the
     transfer script stages `peterbot/`, `deploy/`, `tests/fixtures/edigits` and
     the Hermes fixture at `/opt/peterbot`). Find the runner's container IP on
     ctl0 first (`docker inspect peterbot-hermes-runner -f
     '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'`), then:
     ```sh
     ssh -i ~/.ssh/peterbot_vm root@192.168.241.2 \
       'cd /opt/peterbot && PETERBOT_SMOKE_TOKEN=*** \
        python3 deploy/smoke_rust_worker.py \
          --image peterbot-hermes-worker:REV \
          --gateway-host 192.168.240.2 \
          --runner-probe <runner-ctl0-ip>:8780 --runner-probe 192.168.241.2:8780'
     ```
     It compiles + runs the Rust e-digits fixture inside the real restricted
     worker, compares against the Python decimal reference, re-runs the pinned
     Hermes runtime fixture and the in-container isolation probes (broker 401
     included — `--gateway-host` proves the full DNAT→P910→container path), and
     proves the runner is unreachable from the worker. A SKIP only exits 0 with
     `--expect-no-gateway` explicitly declared; otherwise it fails the run.
   - isolation sweep from the gateway container's network (env-only):
     `PETERBOT_ISOLATION_GATEWAY_HOST=192.168.240.2`,
     `PETERBOT_ISOLATION_RUNNER_HOST=192.168.241.2`,
     `PETERBOT_ISOLATION_HOST_GATEWAY=192.168.241.1`,
     `PETERBOT_ISOLATION_INFERENCE_HOST=100.73.210.66`,
     `PETERBOT_ISOLATION_P910_HOST=100.99.6.59`.
   - P910 host gate: the published broker port means P910 INPUT must ACCEPT
     new TCP from `192.168.241.2` on `virbr-ctl` (stock Docker hosts ACCEPT
     INPUT; a ufw/`--deny`-input host needs the operator to allow that one
     source). The step-8 smoke is the real end-to-end proof.

Posture B (no internet at all in the guest): after first boot,
`virsh detach-device peterbot-worker <vnic1.xml>` and remove the NAT interface
from netplan; package updates then need a re-attach window. The wall already
gives workers no internet either way; this only narrows the guest OS itself.

## Reboot behavior

- `virsh autostart` + guest systemd: firewall unit (installs `Before=docker`,
  pre-creates the bare `pbworkers` bridge + alias so the broker path works even
  before any worker started — dockerd must adopt the matching bridge); a Docker
  `ExecStartPost` hook moves the wall to rule 1 after every daemon restart. Docker and
  the runner compose (`restart: unless-stopped`) come back; the runner clears
  stale worker containers at startup.
- Run `peterbot-worker-vm-verify.sh` after every reboot: unit active, wall
  first in FORWARD, br_netfilter on, alias present, runner `/health` ok.
- Gateway side: runner `/health` failing keeps Peter degraded; no state lost.

## Resource profiles

Guest 6 vCPU / 12 GiB covers the `build` profile (4 CPU, 4 GiB, 2 GiB + 512 MiB
tmpfs) with runner+OS headroom; `pids_limit` ceilings stay, memory-swap equals
memory (no swap escape). One worker at a time — a second task is refused by the
gateway's single foreground slot, not dropped by the runner.

## Rollback

1. Gateway: `runner_url` back to the P910-local runner and re-apply compose
   WITHOUT the overlay (`docker compose -f compose.hermes.yml up -d peterbot`
   — removes the 8770 publish; the pilot runner container was never removed) →
   Peter keeps working on-host.
2. Guest: `docker compose -f deploy/vm/compose.hermes-vm.yml down`;
   `virsh shutdown peterbot-worker` (graceful) or `virsh destroy` (hard).
3. `virsh net-destroy/net-undefine virbr-ctl`; delete disks only after incident
   review. The desktop VM and P910 state are untouched at every step; no P910
   data ever lived in the guest.

## Explicit non-goals (open gates, not silently included)

- No unrestricted internet for workers or persisted member projects — the
  pinned offline toolchain covers dependency-free Rust; dependency access is a
  later slice (proxy/allowlist design), as is per-officer project storage.
- Member-facing execution stays behind the existing officer-only gate; this VM
  work changes where workers run, not who may invoke Peter.
- mTLS gateway↔runner and a guest-side read-only artifact cache are future
  hardening, not part of this recipe.
