#!/usr/bin/env bash
###############################################################################
# hpc_setup.sh — Tailscale + ROS 2 Setup for NYU HPC Greene
#
# Run this ONCE on a Greene login node or compute node.
# NOTE: Run on the HPC, NOT your Mac.
#
# NYU HPC specifics:
#   - Greene uses SLURM for job scheduling.
#   - You do NOT have root/sudo access on compute nodes.
#   - Tailscale can run in "userspace networking" mode without root.
#   - We install Tailscale as a user-space binary in ~/bin/.
#
# What this does:
#   1. Downloads the Tailscale userspace binary (no root needed)
#   2. Starts tailscaled in userspace mode
#   3. Authenticates with an auth key (you generate this in the Tailscale admin)
#   4. Writes a SLURM module file to auto-setup the environment in batch jobs
###############################################################################

set -euo pipefail

REPO_ROOT="${HOME}/AgrobotV2"   # Adjust if your HPC clone path differs
DDS_PROFILE="${REPO_ROOT}/tools/network/ros2_dds_profile.xml"
TAILSCALE_DIR="${HOME}/.local/tailscale"
TAILSCALE_BIN="${HOME}/bin/tailscale"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agrobot TOM v2 — NYU HPC Greene Network Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Download Tailscale (userspace mode, no root) ──────────────────────
echo "[1/4] Downloading Tailscale userspace binary..."
mkdir -p "${HOME}/bin" "${TAILSCALE_DIR}"

TAILSCALE_VERSION="1.80.2"
TAILSCALE_ARCHIVE="tailscale_${TAILSCALE_VERSION}_amd64.tgz"
TAILSCALE_URL="https://pkgs.tailscale.com/stable/${TAILSCALE_ARCHIVE}"

if [ ! -f "${TAILSCALE_BIN}" ]; then
    curl -fsSL "${TAILSCALE_URL}" -o "/tmp/${TAILSCALE_ARCHIVE}"
    tar -xzf "/tmp/${TAILSCALE_ARCHIVE}" -C /tmp/
    cp "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale" "${TAILSCALE_BIN}"
    cp "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" "${HOME}/bin/tailscaled"
    chmod +x "${TAILSCALE_BIN}" "${HOME}/bin/tailscaled"
    rm -rf "/tmp/tailscale_${TAILSCALE_VERSION}_amd64" "/tmp/${TAILSCALE_ARCHIVE}"
    echo "      Tailscale ${TAILSCALE_VERSION} installed to ~/bin/"
else
    echo "      Tailscale already installed at ${TAILSCALE_BIN}"
fi

# ── Step 2: Start tailscaled in userspace mode ────────────────────────────────
# Userspace networking: Tailscale creates a virtual tun device in user space.
# No kernel module, no root. Perfect for HPC environments.
echo ""
echo "[2/4] Starting tailscaled (userspace mode)..."

if ! pgrep -x tailscaled &>/dev/null; then
    nohup "${HOME}/bin/tailscaled" \
        --state="${TAILSCALE_DIR}/tailscaled.state" \
        --socket="${TAILSCALE_DIR}/tailscaled.sock" \
        --tun=userspace-networking \
        &>/tmp/tailscaled.log &
    sleep 3
    echo "      tailscaled started. PID: $(pgrep -x tailscaled || echo 'unknown')"
else
    echo "      tailscaled already running."
fi

# ── Step 3: Authenticate ──────────────────────────────────────────────────────
echo ""
echo "[3/4] Authenticating with Tailscale..."
echo ""
echo "      You need a Tailscale auth key. Generate one here:"
echo "      https://login.tailscale.com/admin/settings/keys"
echo "      → New auth key → Reusable: No → Tags: tag:hpc → Create key"
echo ""
echo "      Then run:"
echo "        ${TAILSCALE_BIN} --socket=${TAILSCALE_DIR}/tailscaled.sock \\"
echo "          up --authkey=<YOUR_AUTH_KEY> --hostname=hpc-greene --advertise-tags=tag:hpc"
echo ""

TAILSCALE_IP=$(${TAILSCALE_BIN} --socket="${TAILSCALE_DIR}/tailscaled.sock" ip -4 2>/dev/null || echo "NOT_YET_AUTHENTICATED")
echo "      Current Tailscale IP: ${TAILSCALE_IP}"

# ── Step 4: Write the environment to .bashrc + SLURM preamble ─────────────────
echo ""
echo "[4/4] Writing ROS 2 + DDS environment to ~/.bashrc..."

SHELL_SNIPPET=$(cat <<EOF

# ── Agrobot TOM v2 — ROS 2 + Tailscale Network Config (HPC Greene) ───────────
# Added by tools/network/setup/hpc_setup.sh
export PATH="${HOME}/bin:\${PATH}"
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_PROFILE}"
# Start tailscaled if not running (useful in interactive sessions).
if ! pgrep -x tailscaled &>/dev/null; then
    nohup "${HOME}/bin/tailscaled" \
        --state="${TAILSCALE_DIR}/tailscaled.state" \
        --socket="${TAILSCALE_DIR}/tailscaled.sock" \
        --tun=userspace-networking \
        &>/tmp/tailscaled.log &
fi
# ─────────────────────────────────────────────────────────────────────────────
EOF
)

if ! grep -q "Agrobot TOM v2" ~/.bashrc 2>/dev/null; then
    echo "${SHELL_SNIPPET}" >> ~/.bashrc
    echo "      Written to ~/.bashrc"
else
    echo "      Already present in ~/.bashrc — skipping."
fi

# Write a SLURM job preamble template for batch jobs that need ROS 2 / Tailscale.
cat > "${REPO_ROOT}/tools/network/setup/slurm_ros2_preamble.sh" <<'SLURM_EOF'
#!/usr/bin/env bash
# slurm_ros2_preamble.sh — Source this at the top of any SLURM job script
# that needs ROS 2 or Tailscale networking.
#
# Usage in your SLURM script:
#   source ~/AgrobotV2/tools/network/setup/slurm_ros2_preamble.sh

# Load ROS 2 Jazzy (assumes it is installed or available via Singularity).
# On Greene you'll likely use a Singularity container — adjust as needed.
source /opt/ros/jazzy/setup.bash 2>/dev/null || true

export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE="${HOME}/AgrobotV2/tools/network/ros2_dds_profile.xml"

# Ensure tailscaled is running on compute node.
if ! pgrep -x tailscaled &>/dev/null; then
    nohup "${HOME}/bin/tailscaled" \
        --state="${HOME}/.local/tailscale/tailscaled.state" \
        --socket="${HOME}/.local/tailscale/tailscaled.sock" \
        --tun=userspace-networking \
        &>/tmp/tailscaled.log &
    sleep 5
fi
SLURM_EOF

echo ""
echo "      Created: tools/network/setup/slurm_ros2_preamble.sh"
echo "      Use this at the top of SLURM batch scripts that need ROS 2."
echo ""
echo "Done. Update ros2_dds_profile.xml with IP: ${TAILSCALE_IP}"
