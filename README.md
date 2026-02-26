# Agrobot TOM v2

Autonomous tomato-picking robot perception pipeline built on ROS 2 Jazzy, Bazel, and Docker.

**Full project documentation:** [Design Doc & Roadmap](https://docs.google.com/document/d/1PyNI9mKTuDRYljvWCO47Ucky7pXPXgNRf4o25j-nc1A/edit?usp=sharing)

---

## Quick Start

```bash
# 1. Install prerequisites (Mac M2)
brew install bazelisk

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
├── perception/         # ROS 2 — tomato detection node (DINOv2 + SAM2)
├── planning/           # ROS 2 — motion & task planning
├── simulation/         # NVIDIA Isaac Sim / Replicator SDG scripts (HPC)
├── deployment/
│   ├── docker/         # Dockerfile.dev · Dockerfile.cuda · Dockerfile.rocm
│   └── compose/        # docker-compose for local dev
└── tools/
    ├── platforms/       # Bazel platform targets (M2, CUDA, ROCm)
    └── network/         # Tailscale ACL + DDS unicast profile
```

---

## Hardware Targets

| Environment | Accelerator | Dockerfile |
|---|---|---|
| Mac M2 (local dev) | Apple MPS | `Dockerfile.dev` |
| NYU HPC Greene (training) | NVIDIA CUDA 12.4 | `Dockerfile.cuda` |
| AMD Edge (robot) | ROCm 6 | `Dockerfile.rocm` |
