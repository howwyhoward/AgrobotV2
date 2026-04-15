#!/usr/bin/env bash
###############################################################################
# pull_crops.sh — Pull spatial JPEG crops from NucBox → Mac over Tailscale
#
# Rsyncs eval_reports/spatial_crops/ from NucBox to the same folder on Mac.
# Files are named: <timestamp>_f<frame>_t<tomato_id>_z<depth>m_r<radius>cm_s<score>.jpg
#
# Usage (Mac terminal, Tailscale running):
#   bash tools/pull_crops.sh
#
# Override NucBox address:
#   NUCBOX_HOST=100.x.x.x bash tools/pull_crops.sh
###############################################################################

set -euo pipefail

NUCBOX_HOST="${NUCBOX_HOST:-100.78.233.101}"
NUCBOX_USER="${NUCBOX_USER:-robotics-club}"
NUCBOX_SSH="${NUCBOX_USER}@${NUCBOX_HOST}"
NUCBOX_CROPS_DIR="AgrobotV2/eval_reports/spatial_crops"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CROPS_DIR="${SCRIPT_DIR}/../eval_reports/spatial_crops"

mkdir -p "${LOCAL_CROPS_DIR}"

echo ""
echo "── Pull crops: NucBox → Mac (rsync over Tailscale) ────────────────"

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no \
    "${NUCBOX_SSH}" "echo ok" &>/dev/null; then
    echo "ERROR: Cannot reach ${NUCBOX_SSH}. Is Tailscale running?"
    echo "  Run: tailscale status"
    exit 1
fi
echo "  ✓ Reachable: ${NUCBOX_SSH}"
echo "  From: ${NUCBOX_SSH}:${NUCBOX_CROPS_DIR}/"
echo "  To:   ${LOCAL_CROPS_DIR}/"
echo ""

rsync -avz --progress \
    "${NUCBOX_SSH}:${NUCBOX_CROPS_DIR}/" \
    "${LOCAL_CROPS_DIR}/"

echo ""
echo "  ✓ Done. Open in Finder:"
echo "    open ${LOCAL_CROPS_DIR}"
open "${LOCAL_CROPS_DIR}" 2>/dev/null || true
