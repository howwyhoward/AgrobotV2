#!/usr/bin/env bash
###############################################################################
# mac_setup.sh — Tailscale + ROS 2 Network Setup for Mac M2
#
# Run this ONCE on your Mac terminal from the AgrobotV2 repo root.
# Prerequisites: Homebrew installed, internet access.
#
# What this does:
#   1. Installs Tailscale via Homebrew Cask
#   2. Authenticates and tags this machine as tag:dev
#   3. Writes the shell environment snippet to ~/.zshrc / ~/.bashrc
#   4. Prints the Tailscale IP so you can fill in ros2_dds_profile.xml
###############################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DDS_PROFILE="${REPO_ROOT}/tools/network/ros2_dds_profile.xml"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agrobot TOM v2 — Mac M2 Network Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Install Tailscale ─────────────────────────────────────────────────
if ! command -v tailscale &>/dev/null; then
    echo "[1/4] Installing Tailscale..."
    brew install --cask tailscale
    echo "      Tailscale installed. Open the Tailscale menu bar app and log in."
    echo "      Then re-run this script."
    exit 0
else
    echo "[1/4] Tailscale already installed: $(tailscale version 2>/dev/null | head -1)"
fi

# ── Step 2: Authenticate and tag this machine ─────────────────────────────────
echo ""
echo "[2/4] Checking Tailscale connection status..."
if ! tailscale status &>/dev/null; then
    echo "      Tailscale is not running. Start it from the menu bar and log in."
    echo "      Then re-run this script."
    exit 1
fi

TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "NOT_CONNECTED")
TAILSCALE_HOSTNAME=$(tailscale status --json 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','unknown'))" \
    2>/dev/null || echo "unknown")

echo "      Tailscale IP:       ${TAILSCALE_IP}"
echo "      Tailscale hostname: ${TAILSCALE_HOSTNAME}"
echo ""
echo "      ACTION REQUIRED: In the Tailscale admin panel, tag this machine as 'tag:dev'."
echo "      URL: https://login.tailscale.com/admin/machines"

# ── Step 3: Write ROS 2 environment to shell config ──────────────────────────
echo ""
echo "[3/4] Writing ROS 2 + DDS environment variables to ~/.zshrc..."

SHELL_SNIPPET=$(cat <<EOF

# ── Agrobot TOM v2 — ROS 2 + Tailscale Network Config ────────────────────────
# Added by tools/network/setup/mac_setup.sh
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_PROFILE}"
# ─────────────────────────────────────────────────────────────────────────────
EOF
)

# Append to zshrc if not already present.
if ! grep -q "Agrobot TOM v2" ~/.zshrc 2>/dev/null; then
    echo "${SHELL_SNIPPET}" >> ~/.zshrc
    echo "      Written to ~/.zshrc"
else
    echo "      Already present in ~/.zshrc — skipping."
fi

# Also write to .bashrc for bash users.
if ! grep -q "Agrobot TOM v2" ~/.bashrc 2>/dev/null; then
    echo "${SHELL_SNIPPET}" >> ~/.bashrc
fi

# ── Step 4: Update the DDS profile with this machine's IP ────────────────────
echo ""
echo "[4/4] Your Tailscale IP is: ${TAILSCALE_IP}"
echo ""
echo "      NEXT STEPS:"
echo "      ┌─────────────────────────────────────────────────────────────────┐"
echo "      │ 1. Edit tools/network/ros2_dds_profile.xml                      │"
echo "      │    Replace 100.AAA.AAA.AAA with: ${TAILSCALE_IP}"
echo "      │                                                                 │"
echo "      │ 2. Run mac_setup.sh again after you have the HPC and AMD IPs   │"
echo "      │    to verify connectivity:                                      │"
echo "      │      tailscale ping hpc-greene                                  │"
echo "      │      tailscale ping amd-edge                                    │"
echo "      │                                                                 │"
echo "      │ 3. Once all three IPs are in ros2_dds_profile.xml,             │"
echo "      │    rebuild the Docker image to bake the profile in:            │"
echo "      │      ./compose.sh build dev                                     │"
echo "      └─────────────────────────────────────────────────────────────────┘"
echo ""
echo "Done."
