# 🍅 Agrobot v2 Project Roadmap

**Objective:** Establish the "Silicon Pipeline," prototype the perception stack, and achieve competition readiness via a Bazel-managed monorepo on **Mac** and **AMD Edge (NucBox)**.  
*(HPC/Greene is optional and deferred; training and inference run on NucBox when HPC is unavailable.)*

---

## Phase 1: Foundation & Prototype (Weeks 1–2)

### Sprint 1: Infrastructure & Environment Parity — **Done**

- Bazel monorepo, Docker ROS 2 Jazzy, Tailscale mesh (Mac ↔ NucBox).
- RealSense D456 driver and USB passthrough in dev and ROCm images; default camera topic set.
- **2-machine pivot:** No HPC dependency; model sync is Mac ↔ NucBox over Tailscale (`model_sync.sh --pull` / `--all`).

---

## Phase 2: Sim-to-Real & Zero-Shot Prototype (Weeks 2–3)

### Sprint 2: Zero-Shot Pipeline, Eval, 3D & Rigor — **In progress**

**Goal:** Prototype DINOv2+SAM2 zero-shot detection, add evaluation and depth→3D, and prepare for fine-tuning on NucBox.

**AI/Systems (Mac + NucBox):**

- **Done:** Replace placeholder with DINOv2+SAM2 zero-shot node (MPS on Mac host, CPU in Docker). First time the robot can “see” a tomato in code.
- **Done:** Dependencies: torch, torchvision, sam2; DINOv2 via torch.hub, SAM2 checkpoint at `models/sam2/`.
- **Done:** Eval pipeline: `perception/eval/run_eval.py` — latency (mean/p99) and mAP@0.5 when ground truth provided; `REPRODUCE.md` for reproducibility.
- **Done:** Depth→3D: optional `depth_topic` / `depth_camera_info_topic`; publish `/agrobot/detections_3d` for the arm.
- **Done:** CI: `.github/workflows/ci.yml` — Bazel build + `//perception:image_utils_test`.
- **Done:** Data: Laboro Tomato val split in `data/val_list.txt` (161 images); `data/README.md`; dataset dirs gitignored.
- **Done:** Ablation path: `--detector dino_only` for DINOv2-only comparison.
- **Deferred:** NVIDIA Isaac Sim / synthetic data generation (requires HPC or NVIDIA GPU). Use real data (Laboro Tomato, KUTomaData) for now; revisit if HPC access is restored.
- **Next:** Curate Laboro Tomato (and optionally KUTomaData) on NucBox (`data/Laboro-Tomato/`); run SAM2 fine-tuning on NucBox; re-run eval and document in REPRODUCE.md.

**On-Site (Test/Integration):**

- Confirm D456 stability (0 frame drops over 10 mins) with `realsense2_camera` + perception launch.
- Pull Sprint 2 ROCm Docker image; run `./deployment/docker/run_rocm.sh` and verify perception in container.

---

## Phase 3: Robot Deployment & Readiness (Weeks 3–4)

### Sprint 3: Model Fine-Tuning & Spatial Mapping

**Goal:** Convert 2D detections to 3D reliably and optimize for AMD Radeon hardware.

**AI/Systems:**

- **Fine-tuning:** Train SAM2 mask decoder on NucBox (ROCm) using Laboro Tomato / KUTomaData (hybrid real data; synthetic deferred until Isaac Sim is available).
- **Edge optimization:** Convert .pt to ONNX; optimize with MIGraphX for Radeon 8060S; target **>30 FPS** or p99 &lt; 33 ms.
- **3D perception:** Deploy 3D Gaussian Splatting (e.g. ROSplat) for a real-time “digital twin” of crop rows; use for consistent 3D tomato positions.
- **Safety/fallback:** Document failure modes (low light, occlusion, no detections, sensor dropout); implement “no pick” / skip behavior in code.

**Test/Integration (On-Site):**

- **Calibration:** Checkerboard calibration; report `camera_info`.
- **Hardware:** Mount D456 to robot arm; verify cable slack.
- **Benchmark:** ROCm inference; report FPS and memory via `rocm-smi`.

---

## Phase 4: Resident VLMs & Competition Readiness (Weeks 5–6)

### Sprint 4: Resident VLMs & Competition Readiness

**Goal:** Deploy a reasoning layer and achieve >30 FPS full-loop performance.

**The Reasoning Layer:**

- **VLM deployment:** Load quantized Qwen-VL (or Florence-2) into 96GB unified memory on NucBox.
- **Failure analysis:** VLM produces structured reports (e.g. “Tomato ID-402 occluded by stem; skipping”).

**Performance & Monitoring:**

- **Foxglove:** Live visualization of 3DGS and SAM2 masks over Tailscale.
- **Tuning:** MIOpen kernel profiling to meet the >30 FPS perception-to-action target.
- **Automation:** Promote `agrobot-perception.service` to full systemd auto-start.

**Final Competition Drill:**

- **HIL testing:** Full chain (Camera → Detector → 3DGS → VLM → Arm).
- **Success metrics:** Run 5 consecutive pick cycles; record success rate and collision data.

---

## Reference

- **Step-by-step checklist:** [docs/GRAND_PLAN.md](GRAND_PLAN.md)
- **Context & tags:** `.cursor/rules/agrobot-context.mdc`
- **Reproducibility:** [REPRODUCE.md](../REPRODUCE.md)
