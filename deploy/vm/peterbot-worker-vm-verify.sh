#!/bin/sh
# Read-only posture verification for the dedicated Peter worker guest.
# Run inside the guest:  sh /opt/peterbot/deploy/vm/peterbot-worker-vm-verify.sh
# Exit 0 = every invariant holds. Never mutates state.
set -u
FAIL=0
ok()   { printf 'ok    %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; FAIL=1; }

# 1. Control addressing: virbr-ctl vNIC present with 192.168.241.2.
ip -4 addr show | grep -q '192\.168\.241\.2' && ok "control address 192.168.241.2" \
  || bad "control address 192.168.241.2 missing"

# 2. Egress wall loaded, correctly targeted, and first in FORWARD so Docker's
# own rules can never precede it.
ip link show pbworkers >/dev/null 2>&1 && ok "worker bridge pbworkers exists" \
  || bad "pbworkers bridge missing (compose uses a fixed bridge name)"
iptables -w -S FORWARD 2>/dev/null | grep -q 'PETERBOT-WORKER-FWD' \
  && ok "FORWARD wall present" || bad "PETERBOT-WORKER-FWD absent from FORWARD"
iptables -w -S PETERBOT-WORKER-FWD 2>/dev/null | grep -q -- '-i pbworkers -j DROP' \
  && ok "worker interface default-deny present" || bad "worker interface default-deny missing"
first=$(iptables -w -S FORWARD 2>/dev/null | sed -n '/^-A FORWARD /{p;q;}')
case "$first" in
  *-j\ PETERBOT-WORKER-FWD*) ok "wall precedes Docker rules in FORWARD" ;;
  *) bad "wall is not the first FORWARD rule: $first" ;;
esac
# Bridged worker traffic only traverses the wall when bridge netfilter is on.
[ "$(sysctl -n net.bridge.bridge-nf-call-iptables 2>/dev/null)" = 1 ] \
  && ok "bridge netfilter enabled" || bad "br_netfilter/sysctl not applied — wall bypassed"
# The broker alias must own ARP on the bridge or DNAT never sees frames.
ip -4 addr show pbworkers 2>/dev/null | grep -q '192\.168\.240\.2' \
  && ok "broker alias .240.2 answers ARP" || bad "broker alias missing from pbworkers"
iptables -w -t nat -S PETERBOT-BROKER 2>/dev/null | grep -q 'DNAT.*192\.168\.241\.1:8770' \
  && ok "broker DNAT targets the P910 host-publish address (not the guest)" \
  || bad "broker DNAT must target 192.168.241.1:8770 (host publish)"
# Worker ingress to guest services is dead regardless of FORWARD ordering.
iptables -w -S INPUT 2>/dev/null | grep -q 'PETERBOT-WORKER-IN' \
  && ok "INPUT wall for worker subnet present" || bad "PETERBOT-WORKER-IN absent from INPUT"
iptables -w -S PETERBOT-WORKER-IN 2>/dev/null | grep -q -- '-i pbworkers -j DROP' \
  && ok "worker-to-guest INPUT default-deny present" || bad "worker-to-guest INPUT rule missing"

# 3. Guest services up.
systemctl is-active --quiet peterbot-vm-firewall && ok "firewall unit active" || bad "firewall unit inactive"
systemctl is-active --quiet docker && ok "docker active" || bad "docker inactive"
systemctl cat docker.service 2>/dev/null | grep -q 'ExecStartPost=/usr/local/bin/peterbot-vm-firewall' \
  && ok "Docker restart re-applies worker wall" || bad "Docker restart hook missing"
systemctl is-active --quiet qemu-guest-agent && ok "guest agent active" || bad "guest agent inactive"

# 4. Networks: control net must exist and NOT be internal (published runner
# port needs the DNAT path); worker net internal with pinned bridge options.
docker network inspect peterbot_control --format '{{.Name}}' 2>/dev/null | grep -q '^peterbot_control$' \
  && ok "peterbot_control exists" || bad "peterbot_control missing (run setup.sh)"
internal=$(docker network inspect peterbot_control --format '{{.Internal}}' 2>/dev/null)
[ "$internal" = "false" ] && ok "control net routable for published port" \
  || bad "control net must NOT be internal (runner publish breaks)"
ctl=$(docker network inspect peterbot_control --format '{{index .Options "com.docker.network.bridge.name"}}' 2>/dev/null)
[ "$ctl" = ctl0 ] && ok "control bridge name pinned (ctl0)" || bad "control bridge name drifted: $ctl"

# 5. Worker bridge must be the fixed-name, ICC-off, no-IP-masq, internal bridge.
# A masquerade here would silently restore general egress behind the wall.
docker network inspect peterbot_workers --format '{{index .Options "com.docker.network.bridge.name"}}' 2>/dev/null | grep -q '^pbworkers$' \
  && ok "bridge name pinned" || bad "bridge name not pinned to pbworkers"
docker network inspect peterbot_workers --format '{{index .Options "com.docker.network.bridge.enable_icc"}}' 2>/dev/null | grep -q '^false$' \
  && ok "ICC disabled" || bad "worker bridge ICC must be disabled (firewall is the only path)"
docker network inspect peterbot_workers --format '{{index .Options "com.docker.network.bridge.enable_ip_masquerade"}}' 2>/dev/null | grep -q '^false$' \
  && ok "no hidden docker masquerade" || bad "bridge IP masquerade must be off"
wi=$(docker network inspect peterbot_workers --format '{{.Internal}}' 2>/dev/null)
[ "$wi" = "true" ] && ok "worker net internal (no docker egress path)" \
  || bad "worker net must be --internal; our ALLOW is the only door"

# 6. No ip_nonlocal_bind: a real-interface bind is required, so the flag must
# stay off or any process could bind arbitrary source addresses.
[ "$(sysctl -qn net.ipv4.ip_nonlocal_bind 2>/dev/null)" = 0 ] \
  && ok "ip_nonlocal_bind off" || bad "ip_nonlocal_bind must be 0"

# 7. Runner health over the CONTROL address only. The compose unit publishes to
# 192.168.241.2:8780, so a loopback probe is impossible by design; a failure
# here means the bind drifted to 0.0.0.0 — investigate, do not "fix" by adding
# a loopback exception.
code=$(curl -sS -m 8 -o /dev/null -w '%{http_code}' http://192.168.241.2:8780/health 2>/dev/null || true)
[ "$code" = 200 ] && ok "runner healthy (control bind)" || bad "runner /health on 192.168.241.2 -> '$code'"
docker inspect peterbot-hermes-runner --format '{{json .NetworkSettings.Ports}}' 2>/dev/null | grep -q '192.168.241.2' \
  && ok "runner port bound to control address only" \
  || bad "runner publish must bind 192.168.241.2 (0.0.0.0 = P910 could reach the runner directly)"

# 8. The trusted runner owns a guest-local Docker socket but has NO L2 path
# to the hostile worker bridge. Its image may run as root inside this guest.
runner_nets=$(docker inspect peterbot-hermes-runner --format '{{json .NetworkSettings.Networks}}' 2>/dev/null || true)
case "$runner_nets" in
  *peterbot_control*)
    case "$runner_nets" in
      *peterbot_workers*) bad "trusted runner shares worker network" ;;
      *) ok "runner is control-only, separate from workers" ;;
    esac ;;
  *) bad "runner control-only network missing" ;;
esac

# 9. Inspect ONLY disposable workers. The trusted runner's guest Docker socket
# is intentional; no worker may receive any host bind mount or that socket.
workers=$(docker ps -aq --filter label=io.peterbot.worker=hermes 2>/dev/null || true)
if [ -z "$workers" ]; then
  echo "SKIP  no live workers to inspect; run smoke_rust_worker.py after reboot"
else
  for n in $workers; do
    mounts=$(docker inspect "$n" --format '{{json .Mounts}}' 2>/dev/null || true)
    case "$mounts" in
      '[]') ok "worker has no host bind mounts" ;;
      *) bad "worker has host mounts or cannot be inspected" ;;
    esac
  done
fi

[ "$FAIL" = 0 ] && echo "VERIFY: PASS" || echo "VERIFY: FAIL"
exit "$FAIL"
