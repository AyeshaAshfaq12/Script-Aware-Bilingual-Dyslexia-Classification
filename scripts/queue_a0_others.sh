#!/usr/bin/env bash
# Wait for the current runner to release runs/.runner.lock, then run the
# A0_others tuning sweep. Serialised deliberately: concurrent runners
# caused the duplicate rows recorded in DEVIATIONS.md D-008, and this
# machine has only 2 physical cores, so overlap would not be faster.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$REPO/runs/.runner.lock"
echo "queue: waiting for $LOCK to clear ..."
while [ -f "$LOCK" ]; do sleep 60; done
echo "queue: lock clear at $(date -u +%FT%TZ), starting A0_others"
cd "$REPO/src"
exec python -u run_tuning.py --arms A0_others
