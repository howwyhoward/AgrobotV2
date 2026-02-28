#!/usr/bin/env bash
###############################################################################
# amd_edge_setup.sh — Tailscale + ROS 2 + ROCm Setup for GMKtec EVO-X2
#
# Hardware: AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151), 96GB unified RAM
#
# Run this ONCE on the AMD edge machine.
# NOTE: Run on the AMD edge machine, NOT your Mac.
#
# Prerequisites:
#   - Ubuntu 24.04 LTS installed (wipe Windows 11 Pro that ships with the device)
#
#   - KERNEL (CRITICAL — two valid paths, pick one):
#     Option A (recommended): Ubuntu OEM kernel >= 6.14.0-1018
#       sudo apt install linux-oem-24.04 && sudo reboot
#       Verify: uname -r   (should show 6.14.x-NNNN-oem)
#
#     Option B: Upstream kernel >= 6.18.4
#       (Use mainline-ppa or compile from source — more complex)
#
#     WHY: Strix Halo (gfx1151) requires two specific kernel patches merged
#     in 6.18.4 upstream (backported into linux-oem-24.04 >= 6.14.0-1018).
#     Without them, ROCm compute workloads fail to initialize or crash.
#     Source: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html
#
#   - ROCm 7.2.0 installed on the HOST (not just in Docker):
#     https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
#     ROCm 6.4.x is UNSTABLE on Strix Halo. Use 7.2.0 minimum.
#
#   - sudo access, internet connection
#
# What this does:
#   1. Installs Tailscale via the official apt repository (system-wide, with root)
#   2. Tags the machine as tag:robot
#   3. Configures GPU memory via amd-ttm (exposes full ~90GB to GPU workloads)
#   4. Installs the ROS 2 DDS profile as a systemd environment file
#   5. Creates a systemd service for the agrobot perception stack (Sprint 4)
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

# ── Step 1b: Verify kernel version ───────────────────────────────────────────
KERNEL_VER=$(uname -r)
echo "      Running kernel: ${KERNEL_VER}"
if echo "${KERNEL_VER}" | grep -q "oem"; then
    echo "      ✓ OEM kernel detected — Strix Halo patches should be present."
    echo "      Verify package version: dpkg -l linux-oem-24.04 | grep 6.14.0"
else
    echo "      ⚠️  WARNING: Not running an OEM kernel."
    echo "      For stable ROCm 7.x on gfx1151, run:"
    echo "        sudo apt install linux-oem-24.04 && sudo reboot"
    echo "      Then re-run this script."
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

# ── Step 3: Configure GPU memory via amd-ttm ─────────────────────────────────
# The GMKtec EVO-X2 has 96GB unified RAM shared between CPU and GPU.
# ROCm accesses this through the GPU Translation Table (GTT).
# By default, GTT is capped at ~50% of total RAM (~48GB).
# We increase it to 90GB so DINOv2 + SAM2 + Qwen-VL can all fit simultaneously.
#
# amd-ttm is AMD's official tool for this configuration.
# Source: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html
echo ""
echo "[3/5] Configuring GPU memory (amd-ttm)..."

if ! command -v pipx &>/dev/null; then
    sudo apt-get install -y pipx
    pipx ensurepath
fi

if ! pipx list 2>/dev/null | grep -q "amd-debug-tools"; then
    pipx install amd-debug-tools
fi

echo "      Current GTT configuration:"
~/.local/bin/amd-ttm 2>/dev/null || pipx run amd-debug-tools amd-ttm

echo ""
echo "      Setting usable GPU memory to 90GB (leaves ~6GB for OS + CPU)..."
~/.local/bin/amd-ttm --set 90 2>/dev/null || pipx run amd-debug-tools amd-ttm --set 90
echo "      ✓ amd-ttm set to 90GB. A reboot is required for this to take effect."
echo "      The script will continue — reboot after it completes."

# ── Step 4: Install DDS profile as a systemd environment file ─────────────────
# A systemd EnvironmentFile is the cleanest way to inject env vars into every
# process started by systemd on this machine — including our future
# agrobot.service unit (Sprint 4).
echo ""
echo "[4/5] Installing ROS 2 DDS environment as systemd EnvironmentFile..."

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

# ── Step 5: Create the agrobot perception systemd service (Sprint 4 stub) ─────
# We create the unit file now (disabled) so Sprint 4 can just `systemctl enable` it.
echo ""
echo "[5/5] Creating agrobot-perception.service systemd unit (disabled until Sprint 4)..."

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
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "REBOOT REQUIRED for amd-ttm memory config to take effect."
echo ""
echo "After reboot, verify GPU is working:"
echo "  rocm-smi                        ← should show gfx1151 GPU"
echo "  python3 -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "From your Mac, verify Tailscale connectivity:"
echo "  tailscale ping amd-edge"
