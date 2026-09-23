#!/usr/bin/env bash
# Run only on P910 after the new gateway image is deployed. This job is
# read-only against live state; it writes a private snapshot outside that state.
set -euo pipefail
umask 077

appdata=${1:-/mnt/NVME/docker/appdata/peterbot}
state=$appdata/data
backups=$appdata/backups
test -d "$state" && test -d "$backups"
apps_gid=$(getent group apps | cut -d: -f3)
test -n "$apps_gid"
image=$(docker inspect peterbot --format '{{.Config.Image}}')
revision=$(docker inspect peterbot --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
test -n "$image"

run_cli() {
  docker run --rm --network none --read-only --user 10000:10000 \
    --group-add "$apps_gid" -e "PETERBOT_REVISION=$revision" \
    -v "$state:/state:ro" -v "$backups:/backups" \
    --entrypoint python "$image" -I "/app/deploy/$1" "${@:2}"
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot="/backups/weekly-$stamp"
run_cli state_backup.py backup /state "$snapshot"
run_cli state_backup.py verify "$snapshot"
run_cli housekeeping.py diagnose /state
run_cli housekeeping.py retention /state
