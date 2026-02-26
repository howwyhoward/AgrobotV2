#!/usr/bin/env bash
###############################################################################
# compose.sh — Docker Compose wrapper for Agrobot TOM v2
#
# WHY this script exists:
#   Docker Compose resolves relative bind-mount paths relative to the compose
#   file's directory. Our compose file lives in `deployment/compose/`, so a
#   relative `source: ../..` resolves to `deployment/compose/../..` — which
#   is the repo root in theory, but Docker Desktop on Mac handles this
#   inconsistently when `-f` points to a subdirectory.
#
#   This script exports AGROBOT_ROOT=$(pwd) before invoking compose, so the
#   bind mount `source: ${AGROBOT_ROOT:-.}` always resolves to the repo root
#   regardless of where the compose file lives.
#
# Usage (always run from the AgrobotV2/ repo root):
#   ./compose.sh run --rm dev bash         ← interactive dev shell
#   ./compose.sh up -d dev                 ← start detached
#   ./compose.sh down                      ← stop all services
#   ./compose.sh build dev                 ← rebuild dev image
#
# Pass-through: all arguments are forwarded to `docker compose`.
###############################################################################

set -euo pipefail

# Verify we're being run from the repo root (where this script lives).
if [ ! -f "$(pwd)/compose.sh" ]; then
    echo "[agrobot] ERROR: Run compose.sh from the AgrobotV2/ repo root." >&2
    echo "          cd /path/to/AgrobotV2 && ./compose.sh $*" >&2
    exit 1
fi

export AGROBOT_ROOT="$(pwd)"

exec docker compose \
    -f "${AGROBOT_ROOT}/deployment/compose/docker-compose.yml" \
    "$@"
