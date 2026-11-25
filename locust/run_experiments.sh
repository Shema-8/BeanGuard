#!/usr/bin/env bash
# Runs the same Locust load test against 1, 2 and 3 API containers and stores CSVs.
# Usage:  ./locust/run_experiments.sh [users] [spawn_rate] [duration]
# Run it from the repository root on the machine that hosts the Docker stack.
set -euo pipefail

USERS="${1:-50}"
SPAWN="${2:-5}"
DURATION="${3:-120s}"
HOST="${HOST:-http://localhost}"
mkdir -p locust/results

for N in 1 2 3; do
  echo "=== ${N} API container(s) ==="
  docker compose up -d --build --scale api="${N}"
  echo "Waiting for the stack to be healthy..."
  for _ in $(seq 1 60); do
    curl -fs "${HOST}/health" >/dev/null 2>&1 && break
    sleep 2
  done
  sleep 10   # let every replica load the model
  locust -f locust/locustfile.py --headless --host "${HOST}" \
         -u "${USERS}" -r "${SPAWN}" -t "${DURATION}" \
         --csv "locust/results/${N}_containers" --html "locust/results/${N}_containers.html" \
         --only-summary
done

python locust/summarize_results.py
