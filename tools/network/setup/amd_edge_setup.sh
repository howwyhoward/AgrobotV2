#!/usr/bin/env bash
###############################################################################
# amd_edge_setup.sh — Tailscale + ROS 2 Setup for AMD Edge Microcomputer
#
# Run this ONCE on the AMD edge machine (96GB RAM, ROCm).
# NOTE: Run on the AMD edge machine, NOT your Mac.
#
# Prerequisites: Ubuntu 24.04 installed, sudo access, internet connection.
#
# What this does:
#   1. Installs Tailscale via the official apt repository (system-wide, with root)
#   2. Tags the machine as tag:robot
#   3. Installs the ROS 2 DDS profile as a systemd environment file
#      so every ROS 2 process gets the correct DDS config automatically
#   4. Creates a systemd service for the agrobot perception stack (Sprint 4)
###############################################################################

set -euo pipefail

REPO_ROOT="${HOME}/AgrobotV2"   # Adjust if your AMD edge clone path differs
DDS_PROFILE="${REPO_ROOT}/tools/network/ros2_dds_profile.xml"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agrobot TOM v2 — AMD Edge Machine Network Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Install Tailscale (system-wide via apt) ───────────────────────────
echo "[1/4] Installing Tailscale..."

if ! command -v tailscale &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sudo sh
    echo "      Tailscale installed."
else
    echo "      Tailscale already installed: $(tailscale version | head -1)"
fi

# ── Step 2: Authenticate and tag as robot ─────────────────────────────────────
echo ""
echo "[2/4] Authenticating Tailscale..."
echo ""
echo "      Generate an auth key for this machine:"
echo "      https://login.tailscale.com/admin/settings/keys"
echo "      → New auth key → Reusable: No → Tags: tag:robot → Create key"
echo ""
echo "      Then run:"
echo "        sudo tailscale up --authkey=<YOUR_AUTH_KEY> --hostname=amd-edge --advertise-tags=tag:robot"
echo ""

TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "NOT_YET_AUTHENTICATED")
echo "      Current Tailscale IP: ${TAILSCALE_IP}"

# ── Step 3: Install DDS profile as a systemd environment file ─────────────────
# A systemd EnvironmentFile is the cleanest way to inject env vars into every
# process started by systemd on this machine — including our future
# agrobot.service unit (Sprint 4).
echo ""
echo "[3/4] Installing ROS 2 DDS environment as systemd EnvironmentFile..."

sudo tee /etc/agrobot-ros2.env > /dev/null <<EOF
# Agrobot TOM v2 — ROS 2 System Environment
# Managed by tools/network/setup/amd_edge_setup.sh
# Loaded by all agrobot systemd services via EnvironmentFile=

ROS_DOMAIN_ID=42
ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
FASTRTPS_DEFAULT_PROFILES_FILE=${DDS_PROFILE}
EOF

echo "      Written to /etc/agrobot-ros2.env"

# Also add to /etc/environment for interactive shells (non-systemd processes).
for var in "ROS_DOMAIN_ID=42" "ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET" "FASTRTPS_DEFAULT_PROFILES_FILE=${DDS_PROFILE}"; do
    key="${var%%=*}"
    if ! grep -q "^${key}=" /etc/environment 2>/dev/null; then
        echo "${var}" | sudo tee -a /etc/environment > /dev/null
    fi
done
echo "      Updated /etc/environment"

# ── Step 4: Create the agrobot perception systemd service (Sprint 4 stub) ─────
# We create the unit file now (disabled) so Sprint 4 can just `systemctl enable` it.
echo ""
echo "[4/4] Creating agrobot-perception.service systemd unit (disabled until Sprint 4)..."

sudo tee /etc/systemd/system/agrobot-perception.service > /dev/null <<'SERVICE_EOF'
# agrobot-perception.service — Autostart the tomato detector at boot (Sprint 4)
# Enable with: sudo systemctl enable --now agrobot-perception

[Unit]
Description=Agrobot TOM v2 Perception Pipeline
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=agrobot
EnvironmentFile=/etc/agrobot-ros2.env
# Source ROS 2 setup via bash -c so the environment is properly inherited.
ExecStartPre=/bin/bash -c "source /opt/ros/jazzy/setup.bash"
ExecStart=/bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    source /ros2_ws/install/setup.bash && \
    ros2 launch agrobot_perception perception.launch.py"
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

sudo systemctl daemon-reload
echo "      Service unit created (not enabled — Sprint 4 will enable it)."
echo ""
echo "Done. Update ros2_dds_profile.xml with IP: ${TAILSCALE_IP}"
echo ""
echo "      To verify Tailscale mesh connectivity from Mac:"
echo "        tailscale ping amd-edge"
echo "        tailscale ping hpc-greene"
