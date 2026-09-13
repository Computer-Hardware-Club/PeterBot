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
