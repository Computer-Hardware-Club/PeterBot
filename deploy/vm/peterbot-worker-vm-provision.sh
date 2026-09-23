#!/bin/sh
# End-to-end provisioning of the dedicated Peter worker guest on P910
# (PETER-12). Run ON P910 as the OPERATOR USER (ofhd — member of libvirt/kvm/
# docker; no sudo needed: libvirtd performs the privileged actions). Idempotent:
# re-running converges (re-verifies the base, re-seeds the ISO, re-runs
# first-boot setup because the instance-id changes; every stage is idempotent).
# Touches ONLY the Peter VM directory + the pinned base qcow2, the virbr-ctl
# network, and the peterbot-worker domain. The desktop VM and `default` network
# are never modified.
#
# Stages:
#   1. read-only precondition checks (KVM, virsh access, default network, tools)
#   2. download + SHA-512-verify the PINNED Debian 12 generic-cloud qcow2
#   3. define/start/autostart host-only virbr-ctl from virbr-ctl.xml
#   4. full-copy the verified base to the guest disk (no backing-file coupling),
#      grow the virtual size to 60G (cloud-init grows the FS on first boot)
#   5. NoCloud seed: user-data + meta-data + SEPARATE network-config (NoCloud
#      requires network config as its own root-level file; user-data cannot
#      configure early networking). cloud-localds -N preferred; mkisofs ISO with
#      a root-level network-config is the verified equivalent.
#   6. render + define + start + autostart the domain (pinned MACs: cloud-init
#      matches network-config BY MAC)
set -eu
VM=peterbot-worker
# Operator-writable VM home on P910 storage. Group kvm so libvirt-qemu (the
# qemu runtime uid) can read/write the disks through group membership.
VM_DIR=${PETERBOT_VM_DIR:-/mnt/NVME/docker/appdata/peterbot/vm}
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BASE_URL=https://cloud.debian.org/images/cloud/bookworm/20260909-2596/debian-12-genericcloud-amd64-20260909-2596.qcow2
BASE_SHA512=08fea112563461f251f3c95a5c5cf8cb25eb60f74cec03e85a97ff91d3efef3059d35837598bbb476008f20db6d3bdc7143c5f2f2a9a6da394a0acc601fd5986
BASE_IMG=$VM_DIR/debian-12-genericcloud-20260909-2596-amd64.qcow2
DISK=$VM_DIR/$VM.qcow2
SEED=$VM_DIR/$VM-seed.img
STATE=$VM_DIR/$VM.state        # pinned MACs, survive re-runs (cloud-init matches on them)
SSHKEY=${PETERBOT_VM_SSHKEY:-$HOME/.ssh/peterbot_vm.pub}
DISK_BYTES=64424509440           # 60 GiB virtual size

# --- 1. preconditions (read-only; refuse to improvise on any other host) ---
[ -e /dev/kvm ] || { echo "FATAL: /dev/kvm absent — this recipe targets the P910 KVM host" >&2; exit 1; }
virsh -c qemu:///system list --all >/dev/null 2>&1 || \
  { echo "FATAL: no qemu:///system access as $(id -un)" >&2; exit 1; }
virsh -c qemu:///system net-list --all | grep -qE '^ *default' || \
  { echo "FATAL: libvirt 'default' network missing — preflight assumption broken" >&2; exit 1; }
for bin in virsh qemu-img curl openssl mktemp base64 sed awk grep stat; do
  command -v "$bin" >/dev/null || { echo "FATAL: $bin missing" >&2; exit 1; }
done
# Seed builder: cloud-localds (installs via `cloud-image-utils`) or mkisofs/
# genisoimage. Both emit a cidata-labeled volume with root-level user-data,
# meta-data and network-config.
CLOUD_LOCALDS=$(command -v cloud-localds || true)
MKISO=$(command -v genisoimage || command -v mkisofs || true)
[ -n "$CLOUD_LOCALDS$MKISO" ] || { echo "FATAL: need cloud-localds or genisoimage/mkisofs" >&2; exit 1; }
# Generate the operator key if absent; private material is NEVER printed.
if [ ! -s "$SSHKEY" ]; then
  echo "generating guest operator key at ${SSHKEY%.pub}"
  ssh-keygen -q -t ed25519 -N '' -f "${SSHKEY%.pub}"
fi
[ -r "$SSHKEY" ] || { echo "FATAL: operator pubkey $SSHKEY unreadable" >&2; exit 1; }

# The setgid bit makes new disk/seed files inherit group kvm. A permissive
# fallback would either break QEMU access or expose state, so fail closed.
install -d -m 2770 -g kvm "$VM_DIR" || {
  echo "FATAL: $VM_DIR must be writable by the operator and group kvm" >&2; exit 1;
}
chmod 2770 "$VM_DIR"
[ "$(stat -c %G "$VM_DIR")" = kvm ] || {
  echo "FATAL: $VM_DIR did not retain group kvm" >&2; exit 1;
}

# --- 2. pinned base image, verified before any use ---
if [ ! -s "$BASE_IMG" ]; then
  echo "fetching pinned Debian generic-cloud base (one-time, ~700 MB)"
  curl -fsSLo "$BASE_IMG.part" "$BASE_URL"
  echo "$BASE_SHA512  $BASE_IMG.part" | sha512sum -c -
  mv "$BASE_IMG.part" "$BASE_IMG"
fi
echo "$BASE_SHA512  $BASE_IMG" | sha512sum -c -

# --- 3. host-only control network (virbr-ctl.xml: no forward, no DHCP) ---
virsh -c qemu:///system net-info virbr-ctl >/dev/null 2>&1 || \
  virsh -c qemu:///system net-define "$REPO_DIR/deploy/vm/virbr-ctl.xml"
virsh -c qemu:///system net-start virbr-ctl 2>/dev/null || true
virsh -c qemu:///system net-autostart virbr-ctl
ip -4 addr show virbr-ctl | grep -q 192.168.241.1 || \
  { echo "FATAL: virbr-ctl lacks 192.168.241.1 — libvirt owns that address; investigate first" >&2; exit 1; }

# --- 4. guest disk: independent copy of the verified base, grown once ---
[ -s "$DISK" ] || qemu-img convert -f qcow2 -O qcow2 "$BASE_IMG" "$DISK"
vs=$(qemu-img info --output=json "$DISK" | sed -n 's/.*"virtual-size": *\([0-9][0-9]*\).*/\1/p' | head -1)
if [ "${vs:-0}" -lt "$DISK_BYTES" ]; then
  # grow only, never shrink; cloud-init's growpart enlarges the FS on boot
  qemu-img resize "$DISK" "$DISK_BYTES" >/dev/null
fi
chmod g+rw "$DISK" 2>/dev/null || [ -w "$DISK" ] || {
  echo "FATAL: QEMU/operator cannot write $DISK" >&2; exit 1;
}

# --- 5. NoCloud seed: user-data + meta-data + separate network-config ---
if [ -s "$STATE" ]; then
  . "$STATE"
else
  MAC1="52:54:00:$(openssl rand -hex 3 | sed 's/../&:/g;s/:$//')"
  MAC2="52:54:00:$(openssl rand -hex 3 | sed 's/../&:/g;s/:$//')"
  printf 'MAC1=%s\nMAC2=%s\n' "$MAC1" "$MAC2" > "$STATE"
fi
SEED_DIR=$(mktemp -d); XML=$(mktemp)
trap 'rm -rf "$SEED_DIR" "$XML"' EXIT
# openssl base64 -A: deterministic single-line encoding (coreutils -w0 is not
# portable to BSD/macOS; a naive `base64 | tr` has bitten us in review).
b64() { openssl base64 -A < "$1"; }
stage() { printf %s "$(b64 "$REPO_DIR/deploy/vm/$1")"; }  # one safe token per file

cat > "$SEED_DIR/meta-data" <<EOF
instance-id: peterbot-worker-$(date -u +%Y%m%d%H%M%S)
local-hostname: peterbot-worker
EOF
{
  echo '#cloud-config'
  echo 'ssh_pwauth: false'
  echo 'disable_root: false'
  echo 'users:'
  echo '  - name: root'
  echo '    lock_passwd: true'
  echo '    ssh_authorized_keys:'
  echo "      - $(cat "$SSHKEY")"
  echo 'write_files:'
  # Guest needs only the stage-2/verify/transfer payload; provision, virbr-ctl
  # and the domain template are P910-side files and never leave the host.
  for f in peterbot-worker-vm-setup.sh peterbot-vm-firewall.sh peterbot-vm-firewall.service \
           compose.hermes-vm.yml peterbot-worker-vm-verify.sh peterbot-vm-transfer.sh; do
    payload=$(stage "$f")
    # Fail-closed: decode what we are about to ship and byte-compare to the
    # source. A truncated/rotated embed must abort, never boot a broken guest.
    printf %s "$payload" | openssl base64 -d -A | cmp -s - "$REPO_DIR/deploy/vm/$f" || {
      echo "FATAL: seed staging round-trip failed for $f" >&2; exit 1; }
    echo "  - path: /opt/peterbot/deploy/vm/$f"
    echo '    encoding: base64'
    echo "    content: $payload"
    echo '    permissions: "0755"'
  done
  echo 'runcmd:'
  echo '  - sh /opt/peterbot/deploy/vm/peterbot-worker-vm-setup.sh 2>&1 | tee /var/log/peterbot-setup.log'
} > "$SEED_DIR/user-data"
# EARLY NETWORKING lives here, not in user-data (NoCloud contract). Version 2,
# matched BY MAC so interface names are never assumed.
{
  echo 'version: 2'
  echo 'renderer: networkd'
  echo 'ethernets:'
  echo '  ctl:'
  echo "    match: {macaddress: \"$MAC2\"}"
  echo '    addresses: ["192.168.241.2/24"]'
  echo '    dhcp4: false'
  echo '  nat-setup:'
  echo "    match: {macaddress: \"$MAC1\"}"
  echo '    dhcp4: true'
} > "$SEED_DIR/network-config"

rm -f "$SEED"
if [ -n "$CLOUD_LOCALDS" ]; then
  # Preferred: cloud-localds writes the cidata FAT volume with the separate
  # network-config file (-N) exactly as NoCloud expects.
  "$CLOUD_LOCALDS" -N "$SEED_DIR/network-config" "$SEED" \
    "$SEED_DIR/user-data" "$SEED_DIR/meta-data"
else
  # Equivalent: ISO9660 cidata volume with root-level user-data, meta-data and
  # network-config (verified layout: files land at the volume root).
  (cd "$SEED_DIR" && "$MKISO" -output "$SEED" -volid cidata -joliet -rock \
    user-data meta-data network-config >/dev/null)
fi
chmod g+r "$SEED" 2>/dev/null || true

# --- 6. render domain XML, define, boot ---
# \x27 escapes are gawk-only; P910 ships mawk. m1 builds a whole <mac .../>
# attribute (hence pre-quoted); the template already quotes @MAC@ (m2 bare).
awk -v m1="'$MAC1'" -v m2="$MAC2" -v disk="$DISK" -v seed="$SEED" '
  {gsub(/@DISK@/, disk); gsub(/@SEED_ISO@/, seed); gsub(/@MAC@/, m2);
   gsub(/@MAC_NAT@/, "<mac address=" m1 "/>")}
  {print}' "$REPO_DIR/deploy/vm/peterbot-worker-vm-domain.xml" > "$XML"
grep -qE '<mac address=' "$XML" || { echo "FATAL: MAC pinning lost in render" >&2; exit 1; }
# Redefine on every run so disk/seed paths in the XML stay authoritative.
if virsh -c qemu:///system dominfo "$VM" >/dev/null 2>&1; then
  virsh -c qemu:///system dominfo "$VM" | grep -q 'Id:' || virsh -c qemu:///system undefine "$VM" 2>/dev/null || true
fi
virsh -c qemu:///system define "$XML"
virsh -c qemu:///system start "$VM" 2>/dev/null || true
virsh -c qemu:///system autostart "$VM"

cat <<EOF
provisioned: domain=$VM disk=$DISK seed=$SEED
  vNIC1 (default NAT, setup-time)  $MAC1
  vNIC2 (virbr-ctl control)        $MAC2 -> guest 192.168.241.2
first boot: ~2-4 min (cloud-init -> apt/docker/firewall unit; log: guest /var/log/peterbot-setup.log)
GATE: boot is only PROVEN after the guest actually answers with 192.168.241.2
(next steps in docs/worker-vm.md):
  ssh -i ~/.ssh/peterbot_vm root@192.168.241.2 'sh /opt/peterbot/deploy/vm/peterbot-worker-vm-verify.sh'
EOF
