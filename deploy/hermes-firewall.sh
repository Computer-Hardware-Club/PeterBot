#!/bin/sh
# Root-owned service invocation; affects ONLY the dedicated Peter worker subnet.
set -eu
IPT=/usr/sbin/iptables
$IPT -w -N PETERBOT-WORKER-HOST 2>/dev/null || true
$IPT -w -F PETERBOT-WORKER-HOST
# Trusted gateway is the only fixed-address exception.
$IPT -w -A PETERBOT-WORKER-HOST -s 192.168.240.2/32 -j RETURN
$IPT -w -A PETERBOT-WORKER-HOST -j DROP
$IPT -w -C INPUT -s 192.168.240.0/24 -j PETERBOT-WORKER-HOST 2>/dev/null || \
  $IPT -w -I INPUT 1 -s 192.168.240.0/24 -j PETERBOT-WORKER-HOST

# VM workers reach the gateway through Docker's host-only published port.
# DOCKER-USER sees the packet AFTER Docker DNAT, so match its original host
# destination with conntrack. The source and ingress interface are pinned to
# the dedicated guest; no tailnet or other Docker service is opened.
while $IPT -w -C DOCKER-USER -i virbr-ctl -s 192.168.241.2/32 -p tcp \
    -m conntrack --ctorigdst 192.168.241.1 --ctorigdstport 8770 -j ACCEPT 2>/dev/null; do
  $IPT -w -D DOCKER-USER -i virbr-ctl -s 192.168.241.2/32 -p tcp \
    -m conntrack --ctorigdst 192.168.241.1 --ctorigdstport 8770 -j ACCEPT
done
$IPT -w -I DOCKER-USER 1 -i virbr-ctl -s 192.168.241.2/32 -p tcp \
  -m conntrack --ctorigdst 192.168.241.1 --ctorigdstport 8770 -j ACCEPT
