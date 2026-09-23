#!/bin/sh
# Stage-2 provisioning INSIDE the dedicated Peter worker guest. Executed on
# FIRST BOOT by cloud-init's runcmd (peterbot-worker-vm-provision.sh embeds it
# in the seed ISO at /opt/peterbot/deploy/vm/), and safe to re-run by hand.
# Everything here is reproducible from this file + /opt/peterbot; the guest is
# disposable (delete + re-provision beats patching).
#
# It does NOT: touch the desktop VM, enable member execution, open general
# internet for workers, or install anything from outside apt/Docker's own repos.
set -eu

# 0. Preconditions from cloud-init: control address on the virbr-ctl vNIC.
# Network config comes from the seed's separate network-config file (NoCloud
# root-level, matched by MAC); a missing address means that did not apply —
# fix provisioning, never hand-patch a disposable guest.
ip -4 addr show | grep -q '192\.168\.241\.2' || {
  echo "FATAL: 192.168.241.2 absent — cloud-init network-config did not apply (seed MAC mismatch? check /var/log/cloud-init-output.log and re-provision)" >&2
  exit 1
}

# 1. Host identity + base tooling. python3-aiohttp/python3-urllib3 power the
# in-guest smoke (sandbox_runner imports aiohttp; the smoke shells to docker CLI).
hostnamectl hostname peterbot-worker || true
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl openssh-server qemu-guest-agent iptables \
  python3-aiohttp python3-urllib3

# 2. Docker CE from Docker's own signed apt repo over HTTPS from
# download.docker.com. The operator records the installed docker-ce version in
# the deployment log (apt-get policy shows it; pin via apt-mark if required).
install -m 0755 -d /etc/apt/keyrings
. /etc/os-release  # VERSION_CODENAME/$ID; Debian 12 -> bookworm stable channel
curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update  # pick up the docker-ce channel just added
apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 3. SSH hardening (cloud-init installed the operator key for root; password
# auth stays off; console+key only).
cat > /etc/ssh/sshd_config.d/90-peterbot.conf <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
EOF
systemctl reload ssh || systemctl reload sshd || true

# 4. Forwarding + bridge netfilter. NO ip_nonlocal_bind: with the runner's
# published port bound to a REAL vNIC address (192.168.241.2) it is unnecessary,
# and leaving it on would let any guest process bind arbitrary source addresses.
cat > /etc/sysctl.d/90-peterbot-worker.conf <<'EOF'
net.ipv4.ip_forward = 1
EOF
# br_netfilter must load BEFORE the sysctl exists; modules-load.d persists it.
echo br_netfilter > /etc/modules-load.d/peterbot-bridge.conf
modprobe br_netfilter
sysctl --system >/dev/null

# 5. External networks, owned HERE so bridge names/subnets/IPv6 are fixed, not
# invented by whoever runs compose first. Both are `internal` (no Docker DNAT
# egress paths); the ONLY sanctioned worker egress is the firewall's DNAT.
systemctl enable --now docker
docker network create --driver bridge \
  --opt com.docker.network.bridge.name=pbworkers \
  --opt com.docker.network.bridge.enable_icc=false \
  --opt com.docker.network.bridge.enable_ip_masquerade=false \
  --subnet 192.168.240.0/24 --gateway 192.168.240.1 \
  --internal \
  peterbot_workers 2>/dev/null || docker network inspect peterbot_workers >/dev/null
# Control net is NOT internal: the runner's published port needs Docker's
# inbound DNAT path on .241.2 (internal networks skip published-port rules).
# Trust boundary: only the supervisor joins it — never a worker.
docker network create --driver bridge \
  --opt com.docker.network.bridge.name=ctl0 \
  --opt com.docker.network.bridge.enable_icc=true \
  --subnet 192.168.242.0/24 --gateway 192.168.242.1 \
  peterbot_control 2>/dev/null || docker network inspect peterbot_control >/dev/null

# 6. Egress wall installed as a boot unit (the seed ISO staged the repo here).
# The unit runs Before=docker.service; re-run now to add the broker alias to the
# (already up) bridge and assert rule order after docker's own rules exist.
install -m 0755 /opt/peterbot/deploy/vm/peterbot-vm-firewall.sh /usr/local/bin/peterbot-vm-firewall
install -m 0644 /opt/peterbot/deploy/vm/peterbot-vm-firewall.service /etc/systemd/system/
install -d -m 0755 /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/90-peterbot-firewall.conf <<'EOF'
[Service]
ExecStartPost=/usr/local/bin/peterbot-vm-firewall
EOF
systemctl daemon-reload
systemctl enable --now peterbot-vm-firewall.service
/usr/local/bin/peterbot-vm-firewall

systemctl enable --now qemu-guest-agent
echo "peterbot-worker-vm-setup: done; verify: sh /opt/peterbot/deploy/vm/peterbot-worker-vm-verify.sh"
