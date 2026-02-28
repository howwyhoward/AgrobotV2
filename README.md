# Agrobot TOM v2

Autonomous tomato-picking robot perception pipeline built on ROS 2 Jazzy, Bazel, and Docker.

**Full design doc & roadmap:** [Google Doc](https://docs.google.com/document/d/1PyNI9mKTuDRYljvWCO47Ucky7pXPXgNRf4o25j-nc1A/edit?usp=sharing)

---

## Quick Start (Local Mac)

```bash
# Prerequisites (one-time)
brew install bazelisk

# 1. Verify Bazel sees the monorepo
bazel query //...

# 2. Start the dev container
./compose.sh run --rm dev bash

# 3. Inside the container — build the ROS 2 workspace
ln -sfn /workspace/perception /ros2_ws/src/agrobot_perception
cd /ros2_ws && colcon build --packages-select agrobot_perception --symlink-install
source install/setup.bash

# 4. Smoke-test the perception node
ros2 run agrobot_perception tomato_detector --ros-args -p publish_debug_image:=false
```

---

## Repo Structure

```
AgrobotV2/
├── perception/          # ROS 2 tomato detection node (DINOv2 + SAM2)
├── planning/            # Robot arm motion planning
├── simulation/          # NVIDIA Isaac Sim / Replicator SDG scripts (HPC)
├── deployment/
│   ├── docker/          # Dockerfile.dev · Dockerfile.cuda · Dockerfile.rocm
│   └── compose/         # docker-compose for local dev
└── tools/
    ├── platforms/        # Bazel platform targets (local/mac, CUDA, ROCm)
    └── network/          # Tailscale ACL + ROS 2 DDS unicast profile
```

---

## Sprint 1 Documentation

### Task 4: Networking — Tailscale Mesh + HPC Access

#### Why Tailscale?

Without a VPN, the three project machines exist on completely isolated networks:

- The **local Mac** is behind a home or office NAT router — no public IP.
- **Greene HPC** is behind NYU's institutional firewall, reachable only via VPN.
- The **AMD edge robot** is on-site, on a separate local subnet.

ROS 2's default communication layer (DDS) uses multicast UDP for peer discovery. Multicast is a Layer-2 (Ethernet) protocol — it physically cannot cross subnet boundaries or VPN tunnels. Without intervention, `ros2 topic list` on the local Mac would be completely silent even if the AMD edge robot was publishing sensor data.

Tailscale solves this by building a **WireGuard-based mesh VPN** between enrolled machines. Each device gets a stable `100.x.x.x` address and a MagicDNS hostname (e.g., `amd-edge`). Because it is a peer-to-peer mesh (not hub-and-spoke), the local Mac talks directly to the AMD edge at WireGuard-level latency — not routed through a central server.

#### Why Not Tailscale on the HPC?

NYU HPC Greene requires **Cisco AnyConnect VPN** (NYU-NET Traffic Only). AnyConnect is hub-and-spoke and routes all traffic through NYU's servers. It conflicts with Tailscale's WireGuard UDP in two ways:

1. AnyConnect frequently blocks non-TCP traffic, killing WireGuard tunnels.
2. Greene compute nodes have no direct internet egress — they reach the outside world through NYU's HTTP proxy only, making Tailscale's control plane unreachable from within a SLURM job.

The correct solution is a **split network architecture**:

```
Local Mac ←── Cisco AnyConnect ──→ Greene HPC
               (SSH + scp/rsync)      (training only, no live ROS 2)

Local Mac ←──── Tailscale mesh ───→ AMD Edge Robot
             (WireGuard peer-to-peer)  (ROS 2 DDS, real-time)

Greene HPC  ✗  AMD Edge              (no direct path — Mac relays model files)
```

The local Mac is the single machine on both networks simultaneously, making it the natural relay for model artifacts after a training run.

#### Making ROS 2 Work Over Tailscale

Even with Tailscale connected, ROS 2 DDS still fails because it tries to use multicast. The fix is switching FastDDS (ROS 2 Jazzy's DDS engine) from multicast to **unicast discovery**. Instead of broadcasting to the subnet, each machine directly contacts its peers at their known Tailscale IPs.

This is configured in `tools/network/ros2_dds_profile.xml` and activated via the environment variable `FASTRTPS_DEFAULT_PROFILES_FILE`. The DDS port for `ROS_DOMAIN_ID=42` is calculated as: `7400 + (250 × 42) = 17900`.

#### Files and What They Do

| File | Where to run | What it does |
|---|---|---|
| `tools/network/tailscale_acl.json` | Tailscale admin panel (web) | Defines which machines exist (`tag:dev`, `tag:robot`) and which ports are open. Paste into `login.tailscale.com/admin/acls`. |
| `tools/network/ros2_dds_profile.xml` | All machines | Disables DDS multicast, replaces with unicast to known Tailscale IPs. **Fill in `100.AAA.AAA.AAA` (local Mac) and `100.CCC.CCC.CCC` (AMD edge) before using.** |
| `tools/network/setup/mac_setup.sh` | Local Mac (one-time) | Installs Tailscale, writes `ROS_DOMAIN_ID`, `ROS_AUTOMATIC_DISCOVERY_RANGE`, and `FASTRTPS_DEFAULT_PROFILES_FILE` to `~/.zshrc`. |
| `tools/network/setup/hpc_setup.sh` | Local Mac (one-time) | Writes SSH `ControlMaster` config for frictionless Greene access (`ssh greene`). Generates `slurm_ros2_preamble.sh` for batch jobs. |
| `tools/network/setup/amd_edge_setup.sh` | AMD edge machine (one-time) | Installs Tailscale, configures GPU memory via `amd-ttm`, writes ROS 2 env to `/etc/agrobot-ros2.env`, creates the `agrobot-perception.service` systemd stub. |
| `tools/network/setup/model_sync.sh` | Local Mac | Two-leg relay: pulls trained model artifacts from HPC via `rsync` over AnyConnect, then pushes them to the AMD edge via `rsync` over Tailscale. |
| `tools/network/setup/verify_mesh.sh` | Local Mac | Pings AMD edge via Tailscale and checks `ROS_DOMAIN_ID` matches across machines. |

#### AMD Edge Machine — Hardware-Specific Requirements

The GMKtec EVO-X2 (AMD Ryzen AI Max+ 395, Radeon 8060S) requires specific configuration that differs from a standard AMD GPU machine:

| Concern | Detail |
|---|---|
| **GPU architecture** | `gfx1151` (RDNA 3.5 / Strix Halo) — distinct from desktop RDNA 3 (`gfx1100`) |
| **ROCm version** | 7.2.0 minimum. ROCm 6.4.x is rated unstable for this chip by AMD's official docs. |
| **Kernel requirement** | Ubuntu OEM kernel `>= 6.14.0-1018` (`sudo apt install linux-oem-24.04`) or upstream `>= 6.18.4`. Two specific kernel patches must be present for ROCm compute workloads to initialize. |
| **Memory architecture** | 96GB is **unified** RAM shared between CPU and GPU. There is no discrete VRAM. ROCm accesses it via the GTT (GPU Translation Table), capped at ~50% by default. Run `amd-ttm --set 90` (done by `amd_edge_setup.sh`) to expose 90GB to GPU workloads. |
| **`HSA_OVERRIDE_GFX_VERSION`** | Must be `11.5.1` (matching `gfx1151`). Using `11.0.x` (the desktop RDNA 3 value) causes a memory access fault on first GPU tensor allocation. |

#### Setup Steps

**One-time on the local Mac:**

```bash
# 1. Run the Mac setup script (installs Tailscale, sets env vars)
bash tools/network/setup/mac_setup.sh

# 2. Run the HPC setup script (configures SSH for Greene)
export NYU_NETID=<your_netid>
bash tools/network/setup/hpc_setup.sh

# 3. Note your Tailscale IP from the script output, then fill it in:
#    tools/network/ros2_dds_profile.xml → replace 100.AAA.AAA.AAA
```

**One-time on the AMD edge machine** (run by on-site Test & Integration team):

```bash
# Prerequisites: Ubuntu 24.04 installed, OEM kernel installed and booted
sudo apt install linux-oem-24.04 && sudo reboot

# After reboot — verify kernel is OEM (should show "-oem" in output)
uname -r

# Install ROCm 7.2.0 from AMD's official guide:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/

# Clone the repo and run the setup script
git clone https://github.com/howwyhoward/AgrobotV2
cd AgrobotV2
bash tools/network/setup/amd_edge_setup.sh
# → installs Tailscale, runs amd-ttm, writes DDS env, creates systemd service stub

# Reboot for memory config to take effect
sudo reboot

# Verify GPU is visible
rocm-smi   # should show gfx1151, ~90GB GTT memory
```

**Fill in the DDS profile and rebuild the dev image:**

```bash
# On the local Mac — after both machines have Tailscale IPs:
# Edit tools/network/ros2_dds_profile.xml
#   replace 100.AAA.AAA.AAA → local Mac's Tailscale IP  (run: tailscale ip -4)
#   replace 100.CCC.CCC.CCC → AMD edge's Tailscale IP   (run: tailscale ip -4 on edge)

# Rebuild the dev container to bake in the updated profile
./compose.sh build dev

# Verify the full mesh
bash tools/network/setup/verify_mesh.sh
```

**Transferring trained models after an HPC training run:**

```bash
# Local Mac terminal — AnyConnect must be connected, Tailscale must be running
export NYU_NETID=<your_netid>
bash tools/network/setup/model_sync.sh --run <SLURM_JOB_ID>
# → pulls from Greene's /scratch, relays to AMD edge automatically
```

---

## Hardware Targets

| Environment | Hardware | Accelerator | Dockerfile |
|---|---|---|---|
| Local dev | Local Mac (Apple Silicon) | Apple MPS | `Dockerfile.dev` |
| Training | NYU HPC Greene (A100 / L4) | CUDA 12.4 | `Dockerfile.cuda` |
| Edge / Robot | GMKtec EVO-X2 (Ryzen AI Max+ 395, 96GB) | ROCm 7.2.0 | `Dockerfile.rocm` |
