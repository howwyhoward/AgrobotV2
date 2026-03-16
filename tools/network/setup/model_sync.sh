#!/usr/bin/env bash
###############################################################################
# model_sync.sh — Transfer Models: Mac ↔ NucBox over Tailscale
#
# Two-machine stack: training runs on NucBox; model artifacts live in the
# NucBox repo at ~/AgrobotV2/models/. This script syncs them to/from your Mac.
#
# Usage (Mac terminal, Tailscale running):
#   bash tools/network/setup/model_sync.sh --pull
#   bash tools/network/setup/model_sync.sh --onnx ./models/detector_v1.onnx
#   bash tools/network/setup/model_sync.sh --all
#
# Prerequisites:
#   - Tailscale running (`tailscale status`)
#   - SSH access to NucBox. Default: 100.78.233.101, user robotics-club.
#   - Override: NUCBOX_HOST=... NUCBOX_USER=... bash model_sync.sh --pull
###############################################################################

set -euo pipefail

LOCAL_MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/models"
NUCBOX_HOST="${NUCBOX_HOST:-100.78.233.101}"
NUCBOX_USER="${NUCBOX_USER:-robotics-club}"
NUCBOX_MODELS_DIR="${NUCBOX_MODELS_DIR:-AgrobotV2/models}"

# SSH target: user@host if NUCBOX_USER set, else host (uses ~/.ssh/config)
NUCBOX_SSH="${NUCBOX_HOST}"
[ -n "${NUCBOX_USER}" ] && NUCBOX_SSH="${NUCBOX_USER}@${NUCBOX_HOST}"

mkdir -p "${LOCAL_MODELS_DIR}"

usage() {
    echo "Usage:"
    echo "  model_sync.sh --pull              Pull models/ from NucBox to Mac"
    echo "  model_sync.sh --onnx <file.onnx>  Push one ONNX file to NucBox"
    echo "  model_sync.sh --all               Push entire local models/ to NucBox"
    echo ""
    echo "Env: NUCBOX_HOST (default 100.78.233.101), NUCBOX_USER (default robotics-club)"
    exit 1
}

check_reachable() {
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no \
        "${NUCBOX_SSH}" "echo ok" &>/dev/null; then
        echo "ERROR: Cannot reach ${NUCBOX_SSH} (SSH over Tailscale)."
        echo "  Run: tailscale status"
        echo "  Override: NUCBOX_HOST=... NUCBOX_USER=... $0 --pull"
        exit 1
    fi
    echo "  ✓ Reachable: ${NUCBOX_SSH}"
}

# ── Pull: NucBox → Mac ─────────────────────────────────────────────────────
sync_pull() {
    echo ""
    echo "── Pull NucBox → Mac (rsync over Tailscale) ───────────────────────"
    check_reachable

    echo "  Pulling from: ${NUCBOX_SSH}:${NUCBOX_MODELS_DIR}/"
    echo "  Saving to:    ${LOCAL_MODELS_DIR}/"

    rsync -avz --progress \
        "${NUCBOX_SSH}:${NUCBOX_MODELS_DIR}/" \
        "${LOCAL_MODELS_DIR}/"

    echo "  ✓ Models pulled to ${LOCAL_MODELS_DIR}/"
}

# ── Push: Mac → NucBox ─────────────────────────────────────────────────────
sync_push() {
    local source_path="$1"

    echo ""
    echo "── Push Mac → NucBox (rsync over Tailscale) ───────────────────────"
    check_reachable

    echo "  Pushing: ${source_path}"
    echo "  To:      ${NUCBOX_SSH}:${NUCBOX_MODELS_DIR}/"

    ssh "${NUCBOX_SSH}" "mkdir -p ${NUCBOX_MODELS_DIR}"

    rsync -avz --progress \
        "${source_path}" \
        "${NUCBOX_SSH}:${NUCBOX_MODELS_DIR}/"

    echo "  ✓ Pushed to ${NUCBOX_HOST}:${NUCBOX_MODELS_DIR}/"
}

# ── Main ──────────────────────────────────────────────────────────────────────
[ $# -eq 0 ] && usage

case "$1" in
    --pull)
        sync_pull
        ;;
    --onnx)
        [ -z "${2:-}" ] && { echo "ERROR: Provide a path to an ONNX file."; usage; }
        echo "Pushing ONNX file to NucBox..."
        sync_push "$2"
        ;;
    --all)
        echo "Pushing entire models/ directory to NucBox..."
        sync_push "${LOCAL_MODELS_DIR}/"
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "Done. Verify on NucBox:"
printf '  ssh %s ls %s/\n' "${NUCBOX_SSH}" "${NUCBOX_MODELS_DIR}"
