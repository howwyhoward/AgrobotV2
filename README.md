# AgrobotV2

Autonomous agricultural robot platform built on ROS 2 Jazzy. Modular monorepo covering perception, planning, and simulation with a Dockerized development environment and Bazel build system.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Apple Silicon / M2+)
- [Bazel](https://bazel.build/) (version pinned in `.bazelversion`)

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/howwyhoward/AgrobotV2.git
cd AgrobotV2

# 2. Configure your environment
cp .env.example .env
# Edit .env — set AGROBOT_ROOT to the absolute path of this repo

# 3. Start the dev container
docker compose -f deployment/compose/docker-compose.yml up dev

# 4. Open an interactive shell inside the container
docker compose -f deployment/compose/docker-compose.yml run dev bash
```

---

## Project Structure

```
AgrobotV2/
├── perception/     # ROS 2 package — tomato detection & sensor processing
├── planning/       # ROS 2 package — motion & task planning
├── simulation/     # Simulation environments
├── deployment/
│   ├── docker/     # Dockerfiles (dev, CUDA, ROCm)
│   └── compose/    # docker-compose for local development
└── tools/          # Bazel toolchain & platform configs
```

---

## Docker Targets

| Target | Use case |
|---|---|
| `Dockerfile.dev` | Local development on Apple Silicon |
| `Dockerfile.cuda` | NVIDIA GPU inference (HPC / cloud) |
| `Dockerfile.rocm` | AMD GPU inference (edge hardware) |

---

## Building with Bazel

```bash
# Build all targets
bazel build //...

# Run tests
bazel test //...
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `AGROBOT_ROOT` | Absolute path to the repo root (used by docker-compose) |
| `ROS_DOMAIN_ID` | ROS 2 DDS domain — default `42` |
