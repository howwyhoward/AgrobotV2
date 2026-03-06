#!/usr/bin/env bash
###############################################################################
# run_rocm.sh — Run the ROCm Docker image with correct GPU device access [EDGE]
#
# Why this script? Ubuntu base images do not define a "render" (or "video")
# group. Using --group-add render fails with "Unable to find group render".
# Device nodes (/dev/kfd, /dev/dri/renderD*) are owned by a host-specific GID
# (e.g. 992). The container process must be in that GID to open the devices.
# This script reads the host GIDs and passes them to docker run so GPU access
# works on any AMD edge machine.
#
# Usage (from repo root on the edge box):
#   ./deployment/docker/run_rocm.sh [command and args...]
# Example:
#   ./deployment/docker/run_rocm.sh
#   ./deployment/docker/run_rocm.sh bash
#   ./deployment/docker/run_rocm.sh ros2 launch perception perception.launch.py
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE="${IMAGE:-agrobot-tom-v2/rocm:latest}"

# Host GID that owns /dev/kfd (and usually the DRI render node)
KFD_GID=""
if [ -c /dev/kfd ]; then
  KFD_GID=$(stat -c '%g' /dev/kfd 2>/dev/null || true)
fi
# Optional: host "video" group for /dev/dri/card*
VIDEO_GID=""
if getent group video &>/dev/null; then
  VIDEO_GID=$(getent group video | cut -d: -f3)
fi

GROUP_ADD_ARGS=()
[ -n "$KFD_GID" ] && GROUP_ADD_ARGS+=(--group-add "$KFD_GID")
[ -n "$VIDEO_GID" ] && [ "$VIDEO_GID" != "$KFD_GID" ] && GROUP_ADD_ARGS+=(--group-add "$VIDEO_GID")

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  "${GROUP_ADD_ARGS[@]}" \
  -v "${REPO_ROOT}:/workspace" \
  -w /workspace \
  "$IMAGE" \
  "$@"
