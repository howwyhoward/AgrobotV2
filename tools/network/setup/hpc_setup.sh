#!/usr/bin/env bash
###############################################################################
# hpc_setup.sh — NYU HPC Greene Access Setup (AnyConnect VPN)
#
# WHY NO TAILSCALE ON HPC:
#   NYU HPC Greene requires Cisco AnyConnect VPN (NYU-NET Traffic Only).
#   AnyConnect routes all traffic through NYU's servers and blocks WireGuard
#   UDP ports. Greene compute nodes also have no direct internet egress —
#   they reach the outside world through NYU's HTTP proxy only.
#
#   Tailscale requires UDP connectivity to its coordination server and direct
#   WireGuard UDP between peers. Neither is available on Greene.
#
# WHAT WE USE INSTEAD:
#   - SSH over AnyConnect for interactive access and file transfer
#   - scp/rsync over AnyConnect for pulling trained model artifacts to Mac
#   - A model_sync.sh script to relay those artifacts to the AMD edge via Tailscale
#
# Run this ONCE on your Mac terminal (NOT on the HPC).
# Prerequisites: AnyConnect is connected (as shown in your screenshot).
###############################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ── You must fill these in ────────────────────────────────────────────────────
# Your NYU NetID (e.g., kh1234)
NYU_NETID="${NYU_NETID:-YOUR_NETID}"
# Greene login node hostname (standard)
GREENE_HOST="greene.hpc.nyu.edu"
# Your scratch directory on Greene (replace <netid> with yours)
GREENE_SCRATCH="/scratch/${NYU_NETID}/agrobot"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agrobot TOM v2 — NYU HPC Greene Access Setup"
echo "  (AnyConnect VPN — no Tailscale)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "${NYU_NETID}" = "YOUR_NETID" ]; then
    echo "ERROR: Set your NYU NetID first:"
    echo "  export NYU_NETID=kh1234"
    echo "  bash tools/network/setup/hpc_setup.sh"
    exit 1
fi

# ── Step 1: Verify AnyConnect is active ──────────────────────────────────────
echo "[1/4] Checking AnyConnect VPN status..."
if ! ping -c 1 -W 2 "${GREENE_HOST}" &>/dev/null; then
    echo ""
    echo "      Cannot reach ${GREENE_HOST}."
    echo "      Make sure Cisco AnyConnect is connected to NYU-NET."
    echo "      Open the AnyConnect app → Connect to: vpn.nyu.edu"
    exit 1
fi
echo "      ✓ ${GREENE_HOST} is reachable over AnyConnect."

# ── Step 2: Write SSH config for frictionless Greene access ──────────────────
echo ""
echo "[2/4] Writing SSH config for Greene..."

SSH_CONFIG_BLOCK=$(cat <<EOF

# ── Agrobot: NYU HPC Greene ────────────────────────────────────────────────
# Added by tools/network/setup/hpc_setup.sh
# Usage: ssh greene   (AnyConnect must be connected)
Host greene
    HostName ${GREENE_HOST}
    User ${NYU_NETID}
    # Multiplex connections: first SSH opens a control socket.
    # Subsequent `ssh greene` or `scp greene:...` reuse it instantly — no password re-prompt.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
    # Forward your SSH agent so you can git clone / push from Greene.
    ForwardAgent yes
    ServerAliveInterval 60
# ──────────────────────────────────────────────────────────────────────────
EOF
)

mkdir -p ~/.ssh && chmod 700 ~/.ssh

if ! grep -q "NYU HPC Greene" ~/.ssh/config 2>/dev/null; then
    echo "${SSH_CONFIG_BLOCK}" >> ~/.ssh/config
    chmod 600 ~/.ssh/config
    echo "      Written to ~/.ssh/config"
    echo "      You can now use: ssh greene"
else
    echo "      Already present in ~/.ssh/config — skipping."
fi

# ── Step 3: Create scratch directory and clone repo on Greene ─────────────────
echo ""
echo "[3/4] Setting up Greene scratch directory..."
echo ""
echo "      Run these commands manually on Greene (after: ssh greene):"
echo ""
echo "      ┌─────────────────────────────────────────────────────────┐"
echo "      │ mkdir -p ${GREENE_SCRATCH}                              │"
echo "      │ cd ${GREENE_SCRATCH}                                    │"
echo "      │ git clone <your-repo-url> AgrobotV2                    │"
echo "      │                                                         │"
echo "      │ # Load the Singularity/Apptainer container with ROS 2  │"
echo "      │ module load singularity/3.11.4                          │"
echo "      └─────────────────────────────────────────────────────────┘"

# ── Step 4: Write SLURM environment preamble ──────────────────────────────────
echo ""
echo "[4/4] Writing SLURM preamble for training jobs..."

cat > "${REPO_ROOT}/tools/network/setup/slurm_ros2_preamble.sh" <<'SLURM_EOF'
#!/usr/bin/env bash
###############################################################################
# slurm_ros2_preamble.sh — Source at the top of any SLURM batch script.
#
# Greene specifics:
#   - No Tailscale. No WireGuard. Use scp/rsync to transfer artifacts.
#   - ROS 2 is available inside a Singularity container.
#   - CUDA 12.x is available via the module system.
#
# Usage in your SLURM script:
#   source ~/agrobot/tools/network/setup/slurm_ros2_preamble.sh
###############################################################################

# CUDA + cuDNN from Greene's module system
module purge
module load cuda/12.4.1
module load cudnn/8.9.7.29-cuda12

# Python environment (Singularity container in Sprint 2)
export PYTHONPATH="${HOME}/agrobot:${PYTHONPATH:-}"

# ROS 2 environment — available inside Singularity only, not on bare SLURM nodes.
# Use `singularity exec` wrappers for ros2 commands in batch jobs.
export ROS_DOMAIN_ID=42
SLURM_EOF

echo "      Created: tools/network/setup/slurm_ros2_preamble.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "DONE. Summary of your HPC workflow:"
echo ""
echo "  1. Connect AnyConnect → vpn.nyu.edu → NYU-NET"
echo "  2. ssh greene                    ← uses the SSH config above"
echo "  3. sbatch <your_training_job>    ← Sprint 2 SLURM scripts"
echo "  4. scp greene:<path> ./models/   ← pull trained weights to Mac"
echo "  5. rsync ./models/ amd-edge:...  ← push to robot via Tailscale"
echo ""
echo "  See tools/network/setup/model_sync.sh for steps 4–5 automated."
