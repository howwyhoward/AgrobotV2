# Agrobot TOM v2

Autonomous tomato-picking robot perception pipeline built on ROS 2 Jazzy, Bazel, and Docker.

**Roadmap:** [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md) · **Sprint checklist:** [docs/GRAND_PLAN.md](docs/GRAND_PLAN.md) · **Reproduce eval numbers:** [REPRODUCE.md](REPRODUCE.md)

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

## Quick Start (NucBox)

```bash
cd ~/AgrobotV2
./deployment/docker/run_rocm.sh bash
```

---

## Repo Structure

```
AgrobotV2/
├── docs/               # Grand plan, roadmap, failure modes (Sprint 3)
├── perception/         # ROS 2 tomato detection node (DINOv2 + SAM2)
├── planning/           # Robot arm motion planning (Sprint 4)
├── simulation/         # Isaac Sim / Replicator SDG (deferred; NVIDIA-only)
├── data/               # val_list.txt, README; large datasets gitignored
├── models/             # Model weights (gitignored); see models/README.md
├── deployment/
│   ├── docker/         # Dockerfile.dev · Dockerfile.rocm
│   └── compose/        # docker-compose for local dev
└── tools/
    ├── platforms/      # Bazel platform targets (Mac, ROCm)
    └── network/        # Tailscale ACL + ROS 2 DDS unicast profile
```
