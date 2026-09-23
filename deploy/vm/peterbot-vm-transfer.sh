#!/bin/sh
# Image transfer: P910 (build host) -> worker guest. Images move as
# docker-archive streams over the operator's authenticated SSH, never through a
# shared registry with pull credentials on the guest. The guest runs NO registry
# login; its only images are the ones the operator pushed here.
#
# Run ON P910 from a repo checkout, after building the pinned images:
#   deploy/vm/peterbot-vm-transfer.sh \
#     peterbot-hermes-runner:REV peterbot-hermes-worker:REV
# Guest endpoint: root@192.168.241.2 over the host-only virbr-ctl bridge, key
# from provisioning (~/.ssh/peterbot_vm). Override PETERBOT_VM_SSH /
# PETERBOT_VM_KEY for a different jump setup.
set -eu
: "${PETERBOT_VM_SSH:=root@192.168.241.2}"
: "${PETERBOT_VM_KEY:=$HOME/.ssh/peterbot_vm}"
guest_ssh() {
  ssh -i "$PETERBOT_VM_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "$PETERBOT_VM_SSH" "$@"
}
[ $# -ge 1 ] || { echo "usage: $0 IMAGE[:tag]..." >&2; exit 2; }
tmp=/var/tmp/peterbot-xfer.tar
for image in "$@"; do
  echo "transferring $image"
  docker save "$image" -o "$tmp"
  src=$(sha256sum "$tmp" | cut -d' ' -f1)
  guest_ssh "cat > /var/tmp/peterbot-image.tar" < "$tmp"
  # Integrity: the guest's own hash of the landed stream must match the source
  # hash before docker load trusts it.
  dst=$(guest_ssh 'sha256sum /var/tmp/peterbot-image.tar' | cut -d' ' -f1)
  [ "$src" = "$dst" ] || { echo "FATAL: digest mismatch for $image ($src vs $dst)" >&2; exit 1; }
  guest_ssh 'docker load -i /var/tmp/peterbot-image.tar && rm -f /var/tmp/peterbot-image.tar'
  rm -f "$tmp"
done
echo "transfer complete; verify with: ssh -i $PETERBOT_VM_KEY $PETERBOT_VM_SSH docker images"

# Smoke dependencies: Rust and package smokes run IN the guest against the
# guest-local socket and need these repo files present at
# /opt/peterbot. Push the exact subset (trusted operator channel, same as images).
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
tar -C "$repo" -czf /var/tmp/peterbot-smoke-src.tgz \
  peterbot/__init__.py peterbot/sandbox_runner.py \
  deploy/smoke_rust_worker.py deploy/smoke_package_worker.py \
  deploy/check_hermes_isolation.py \
  tests/fixtures/edigits tests/test_hermes_runtime_integration.py
guest_ssh 'mkdir -p /opt/peterbot && tar -xz -C /opt/peterbot' < /var/tmp/peterbot-smoke-src.tgz
rm -f /var/tmp/peterbot-smoke-src.tgz
echo "smoke sources staged at /opt/peterbot"
