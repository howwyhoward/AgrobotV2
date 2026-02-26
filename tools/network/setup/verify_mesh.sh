#!/usr/bin/env bash
###############################################################################
# verify_mesh.sh — Verify Tailscale Mesh Connectivity
#
# Run on your Mac AFTER all three machines have joined the tailnet.
# This script pings all peers and checks ROS 2 DDS reachability.
#
# Usage (Mac terminal):
#   bash tools/network/setup/verify_mesh.sh
###############################################################################

set -euo pipefail

PEERS=("hpc-greene" "amd-edge")   # MagicDNS names — adjust if yours differ

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Agrobot TOM v2 — Tailscale Mesh Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Tailscale Status ───────────────────────────────────────────────────────
echo "[1] Tailscale status:"
tailscale status
echo ""

# ── 2. Ping each peer ─────────────────────────────────────────────────────────
echo "[2] Pinging peers..."
for peer in "${PEERS[@]}"; do
    if tailscale ping --c=3 "${peer}" &>/dev/null; then
        LATENCY=$(tailscale ping --c=1 "${peer}" 2>&1 | grep -oE '[0-9]+ms' | head -1 || echo "unknown")
        echo "    ✓ ${peer} — reachable (${LATENCY})"
    else
        echo "    ✗ ${peer} — NOT reachable (not on tailnet yet?)"
    fi
done
echo ""

# ── 3. SSH connectivity ───────────────────────────────────────────────────────
echo "[3] SSH check (10s timeout)..."
for peer in "${PEERS[@]}"; do
    if ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no \
       "${peer}" "echo ok" &>/dev/null; then
        echo "    ✓ ${peer} — SSH works"
    else
        echo "    ✗ ${peer} — SSH failed (check ~/.ssh/config or Tailscale SSH setting)"
    fi
done
echo ""

# ── 4. ROS_DOMAIN_ID check ────────────────────────────────────────────────────
echo "[4] ROS_DOMAIN_ID on peers (requires SSH + ROS 2 to be sourced)..."
for peer in "${PEERS[@]}"; do
    REMOTE_DOMAIN=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "${peer}" \
        "echo \$ROS_DOMAIN_ID" 2>/dev/null || echo "unreachable")
    if [ "${REMOTE_DOMAIN}" = "42" ]; then
        echo "    ✓ ${peer} — ROS_DOMAIN_ID=42 ✓"
    else
        echo "    ! ${peer} — ROS_DOMAIN_ID='${REMOTE_DOMAIN}' (expected 42 — run the setup script on that machine)"
    fi
done
echo ""

echo "Verification complete."
echo "If all peers are reachable and ROS_DOMAIN_ID=42, you are ready for Sprint 2."
