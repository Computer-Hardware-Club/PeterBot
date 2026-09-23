#!/bin/sh
# Guest-side egress wall for the dedicated Peter worker VM (root-owned service).
# The worker docker bridge cannot be `internal: true` here because workers must
# reach the P910 gateway broker; the chains below replace that with an explicit
# allow-list. Same discipline as deploy/hermes-firewall.sh: never relax this to
# pass a task.
#
# Ingress interface AND source subnet are matched together: `-i pbworkers`
# makes a spoofed-source packet arriving on any other interface miss the chain
# entirely, and everything else from the worker subnet hits the default-deny.
set -eu
IPT=/usr/sbin/iptables
# Bridged worker traffic only reaches FORWARD/nat once bridge netfilter is on;
# without this the whole wall below is silently bypassed for docker bridges.
modprobe br_netfilter 2>/dev/null || true
sysctl -qw net.bridge.bridge-nf-call-iptables=1
WORKER_SUBNET=192.168.240.0/24
WORKER_BRIDGE=pbworkers
# Broker alias .240.2:8770 DNATs to the broker published on the guest's CONTROL
# vNIC address on P910 (gateway compose.hermes-vm-gateway.yml overlay publishes
# 8770 on 192.168.241.1 ONLY). Deliberately not 0.0.0.0, and never a runner or
# host-management port.
BROKER_REAL=192.168.241.1:8770
# This unit runs BEFORE docker. Pre-create the bare worker bridge (docker adopts
# an existing same-name bridge) so the alias is live even on a boot where no
# worker has started yet; the alias IP must answer ARP on the bridge or frames
# die before PREROUTING ever sees them.
if ! ip link show "$WORKER_BRIDGE" >/dev/null 2>&1; then
  ip link add name "$WORKER_BRIDGE" type bridge
  ip link set "$WORKER_BRIDGE" up
fi
ip -4 addr show "$WORKER_BRIDGE" | grep -q '192\.168\.240\.2' || \
  ip addr add 192.168.240.2/32 dev "$WORKER_BRIDGE"

# --- FORWARD: worker egress wall ---------------------------------------------
$IPT -w -N PETERBOT-WORKER-FWD 2>/dev/null || true
$IPT -w -F PETERBOT-WORKER-FWD
# Conntrack FIRST: the broker's reply re-enters FORWARD AFTER conntrack
# un-DNAT/un-MASQUERADE with source 192.168.240.2 — INSIDE the worker subnet —
# so without this the default-deny below kills every response. Keyed to
# conntrack only: an unsolicited or spoofed flow never matches.
$IPT -w -A PETERBOT-WORKER-FWD -o "$WORKER_BRIDGE" \
  -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# The single permitted NEW worker egress. FORWARD runs AFTER nat PREROUTING, so
# the broker flow is matched on its POST-DNAT destination (alias .240.2 ->
# .241.1); ingress interface AND source subnet are both required.
$IPT -w -A PETERBOT-WORKER-FWD -i "$WORKER_BRIDGE" -s "$WORKER_SUBNET" \
  -d 192.168.241.1 -p tcp --dport 8770 -j ACCEPT
# Everything else from a worker — runner, guest services, other bridges,
# tailnet, metadata, internet — dies here, regardless of interface.
$IPT -w -A PETERBOT-WORKER-FWD -i "$WORKER_BRIDGE" -j DROP
$IPT -w -A PETERBOT-WORKER-FWD -s "$WORKER_SUBNET" -j DROP
# Default-deny the whole worker subnet BEFORE Docker's FORWARD ACCEPT rules;
# position 1 is asserted by the verify script and re-applied on every run
# (docker restart flushes nothing here, but reboots reorder chains).
while $IPT -w -C FORWARD -j PETERBOT-WORKER-FWD 2>/dev/null; do
  $IPT -w -D FORWARD -j PETERBOT-WORKER-FWD
done
$IPT -w -I FORWARD 1 -j PETERBOT-WORKER-FWD

# --- nat: broker alias --------------------------------------------------------
$IPT -w -t nat -N PETERBOT-BROKER 2>/dev/null || true
$IPT -w -t nat -F PETERBOT-BROKER
$IPT -w -t nat -A PETERBOT-BROKER -i "$WORKER_BRIDGE" -s "$WORKER_SUBNET" \
  -p tcp -d 192.168.240.2 --dport 8770 -j DNAT --to-destination "$BROKER_REAL"
while $IPT -w -t nat -C PREROUTING -j PETERBOT-BROKER 2>/dev/null; do
  $IPT -w -t nat -D PREROUTING -j PETERBOT-BROKER
done
$IPT -w -t nat -I PREROUTING 1 -j PETERBOT-BROKER
# MASQUERADE on the way out: P910 has no route back into the guest-internal
# 192.168.240.0/24, so the reply path needs source rewriting. POSTROUTING sees
# the POST-DNAT destination. Identity is the per-job capability token, not IP.
$IPT -w -t nat -A POSTROUTING -s "$WORKER_SUBNET" -p tcp -d 192.168.241.1 --dport 8770 \
  -j MASQUERADE

# --- INPUT: workers never touch guest services --------------------------------
$IPT -w -N PETERBOT-WORKER-IN 2>/dev/null || true
$IPT -w -F PETERBOT-WORKER-IN
$IPT -w -A PETERBOT-WORKER-IN -i "$WORKER_BRIDGE" -j DROP
$IPT -w -A PETERBOT-WORKER-IN -s "$WORKER_SUBNET" -j DROP
while $IPT -w -C INPUT -j PETERBOT-WORKER-IN 2>/dev/null; do
  $IPT -w -D INPUT -j PETERBOT-WORKER-IN
done
$IPT -w -I INPUT 1 -j PETERBOT-WORKER-IN

# Status echo for the unit log.
$IPT -w -S FORWARD | head -5
echo "peterbot-vm-firewall: worker subnet pinned to broker alias -> $BROKER_REAL"
