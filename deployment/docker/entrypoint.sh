#!/bin/bash
###############################################################################
# entrypoint.sh — Container Entrypoint
#
# Why do we need this instead of just `source` in the Dockerfile?
#
# `RUN source /opt/ros/jazzy/setup.bash` in a Dockerfile only affects that
# single RUN layer — the environment is NOT inherited by subsequent layers or
# by processes started with CMD/ENTRYPOINT.
#
# The correct pattern is: source in the entrypoint script, then exec the
# user's command with `exec "$@"`. The `exec` replaces the shell process with
# the user's command (PID 1), which is critical for signal handling (SIGTERM,
# SIGINT) to work correctly in Docker.
###############################################################################

set -e

# Source ROS 2 Jazzy environment
source /opt/ros/jazzy/setup.bash

# Source the local workspace overlay if it has been built.
# This adds our custom packages on top of the base ROS 2 installation.
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
    echo "[agrobot] Sourced local ROS 2 workspace overlay."
fi

echo "[agrobot] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[agrobot] ROS_DISTRO=${ROS_DISTRO}"
echo "[agrobot] Container ready."

# Replace this shell process with the user's command.
exec "$@"
