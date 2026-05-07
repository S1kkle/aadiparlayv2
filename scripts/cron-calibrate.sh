#!/usr/bin/env bash
# Render Cron Job entrypoint — fires the calibration endpoint on the running
# backend service. Render injects BACKEND_URL + optional CRON_TOKEN as env vars.
set -euo pipefail

if [ -z "${BACKEND_URL:-}" ]; then
  echo "BACKEND_URL not set — aborting calibration cron"
  exit 1
fi

URL="${BACKEND_URL%/}/calibration/run"
HEADERS=()
if [ -n "${CRON_TOKEN:-}" ]; then
  HEADERS+=(-H "X-Cron-Token: ${CRON_TOKEN}")
fi

echo "Triggering calibration at $(date -u +%FT%TZ): ${URL}"
RESPONSE=$(curl -fsS -X POST "${HEADERS[@]}" \
  --max-time 600 \
  -H "Content-Type: application/json" \
  -d '{"source": "render-cron"}' \
  "${URL}")
echo "Calibration response:"
echo "${RESPONSE}"
