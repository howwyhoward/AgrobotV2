#!/usr/bin/env bash
###############################################################################
# model_sync.sh — Transfer Trained Models: HPC → Mac → AMD Edge
#
# This script bridges the two separate network legs:
#   Leg 1: HPC Greene → Mac   (scp over AnyConnect VPN)
#   Leg 2: Mac → AMD Edge     (rsync over Tailscale mesh)
#
# Why the relay through Mac?
#   HPC and AMD Edge have no direct connectivity:
#     - HPC is behind NYU's AnyConnect VPN (NYU-only network)
#     - AMD Edge is on Tailscale (separate mesh)
#   Mac is the one machine connected to both networks simultaneously.
#
# Usage (Mac terminal, AnyConnect connected):
#   bash tools/network/setup/model_sync.sh --run <SLURM_JOB_ID>
#   bash tools/network/setup/model_sync.sh --onnx ./models/detector_v1.onnx
#
# Prerequisites:
#   - AnyConnect connected to NYU-NET
#   - Tailscale running (`tailscale status`)
#   - ~/.ssh/config has `greene` host entry (set up by hpc_setup.sh)
###############################################################################

set -euo pipefail

NYU_NETID="${NYU_NETID:-YOUR_NETID}"
GREENE_SCRATCH="/scratch/${NYU_NETID}/agrobot"
LOCAL_MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/models"
AMD_EDGE_HOST="amd-edge"                  # Tailscale MagicDNS hostname
AMD_EDGE_MODELS_DIR="/opt/agrobot/models" # Destination on AMD edge

mkdir -p "${LOCAL_MODELS_DIR}"

usage() {
    echo "Usage:"
    echo "  model_sync.sh --run <SLURM_JOB_ID>       Sync outputs of a completed SLURM job"
    echo "  model_sync.sh --onnx <local_file.onnx>   Push a specific ONNX file to AMD edge"
    echo "  model_sync.sh --all                        Sync entire models/ directory to AMD edge"
    exit 1
}

# ── Helpers ───────────────────────────────────────────────────────────────────
check_anyconnect() {
    if ! ping -c 1 -W 2 greene.hpc.nyu.edu &>/dev/null; then
        echo "ERROR: Cannot reach Greene. Is AnyConnect connected to NYU-NET?"
        exit 1
    fi
    echo "  ✓ AnyConnect: connected to NYU-NET"
}

check_tailscale() {
    if ! tailscale ping --c=1 "${AMD_EDGE_HOST}" &>/dev/null; then
        echo "ERROR: Cannot reach ${AMD_EDGE_HOST} via Tailscale."
        echo "  Run: tailscale status"
        exit 1
    fi
    echo "  ✓ Tailscale: ${AMD_EDGE_HOST} reachable"
}

# ── Leg 1: HPC → Mac ─────────────────────────────────────────────────────────
sync_from_hpc() {
    local job_id="$1"
    local remote_output="${GREENE_SCRATCH}/outputs/job_${job_id}"

    echo ""
    echo "── Leg 1: HPC Greene → Mac (scp over AnyConnect) ──────────────────"
    check_anyconnect

    echo "  Pulling from: greene:${remote_output}/"
    echo "  Saving to:    ${LOCAL_MODELS_DIR}/"

    # rsync is safer than scp for directories: resumes interrupted transfers.
    rsync -avz --progress \
        "greene:${remote_output}/" \
        "${LOCAL_MODELS_DIR}/job_${job_id}/"

    echo "  ✓ Models pulled to ${LOCAL_MODELS_DIR}/job_${job_id}/"
}

# ── Leg 2: Mac → AMD Edge ─────────────────────────────────────────────────────
sync_to_amd() {
    local source_path="$1"

    echo ""
    echo "── Leg 2: Mac → AMD Edge (rsync over Tailscale) ───────────────────"
    check_tailscale

    echo "  Pushing: ${source_path}"
    echo "  To:      ${AMD_EDGE_HOST}:${AMD_EDGE_MODELS_DIR}/"

    # Create destination directory on AMD edge if it doesn't exist.
    ssh "${AMD_EDGE_HOST}" "mkdir -p ${AMD_EDGE_MODELS_DIR}"

    rsync -avz --progress \
        "${source_path}" \
        "${AMD_EDGE_HOST}:${AMD_EDGE_MODELS_DIR}/"

    echo "  ✓ Models pushed to ${AMD_EDGE_HOST}:${AMD_EDGE_MODELS_DIR}/"
}

# ── Main ──────────────────────────────────────────────────────────────────────
[ $# -eq 0 ] && usage

case "$1" in
    --run)
        [ -z "${2:-}" ] && { echo "ERROR: Provide a SLURM job ID."; usage; }
        echo "Syncing SLURM job ${2} outputs..."
        sync_from_hpc "$2"
        sync_to_amd "${LOCAL_MODELS_DIR}/job_${2}/"
        ;;
    --onnx)
        [ -z "${2:-}" ] && { echo "ERROR: Provide a path to an ONNX file."; usage; }
        echo "Pushing ONNX file to AMD edge..."
        check_tailscale
        sync_to_amd "$2"
        ;;
    --all)
        echo "Syncing entire models/ directory to AMD edge..."
        check_tailscale
        sync_to_amd "${LOCAL_MODELS_DIR}/"
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "Done. Verify on AMD edge:"
echo "  ssh ${AMD_EDGE_HOST} ls ${AMD_EDGE_MODELS_DIR}/"
