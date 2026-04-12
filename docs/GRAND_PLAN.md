# Agrobot TOM v2 — Grand Plan

Single source of truth for the perception pipeline roadmap. Use this to see where you are, what addresses the “employer-grade” gaps, and what to do next step by step.

---

## Where You Are Now

| Phase   | Sprint   | Status             | What's done |
|---------|----------|--------------------|-------------|
| Phase 1 | Sprint 1 | Done               | Bazel, Docker ROS 2 Jazzy, Tailscale, RealSense, 2-machine pivot |
| Phase 2 | Sprint 2 | Done               | Zero-shot DINOv2+SAM2, eval, depth-to-3D, CI, ablation, data-driven embedding |
| Phase 3 | Sprint 3 | Done (partial)     | ONNX export, SAM2 fine-tune, SAM2 AMG detector (mAP 0 to 0.023), failure modes + watchdog. MIGraphX GPU blocked, see docs/SPRINT3_ROCM_ISSUE.md |
| Phase 4 | Sprint 4 | In progress        | Qwen-VL deployment, HIL testing |

**You are in Sprint 4.**

---

## Do the Weaknesses Get Addressed?

| Weakness | Addressed by | Sprint | Deliverable |
|----------|--------------|--------|-------------|
| No evaluation story | Eval pipeline | 2 | `perception/eval/` script: run on val set → mAP@0.5 (or P/R), latency mean/p99; `REPRODUCE.md` |
| 2D only (no 3D for arm) | Depth fusion | 2 | Node or util: RealSense depth + 2D detections → 3D bbox/point per tomato; publish 3D detections |
| No CI | CI workflow | 2 | `.github/workflows/ci.yml`: Bazel build + `//perception:image_utils_test` (+ optional ruff) |
| No technical hook | Ablation / baseline | 2 | One comparison (e.g. DINOv2-only vs DINOv2+SAM2) on same val set; document in REPRODUCE.md |
| Safety / fallback | Failure-mode design | 3 | Short doc + code: no detections → “no pick”; low confidence → skip; sensor dropout → safe default |

So: **yes**, the grand plan below addresses all five. Evaluation, 3D, CI, and ablation are in **Sprint 2**; safety/fallback is in **Sprint 3**.

---

## Full Tech Stack (What We Use and Where)

| Layer | Tech | Role |
|-------|------|------|
| Build & env | Bazel, Docker, pip (requirements.in/lock) | Hermetic builds, Mac vs NucBox parity |
| Runtime | ROS 2 Jazzy, FastDDS (unicast), vision_msgs | Perception graph, DDS over Tailscale |
| Network | Tailscale | Mac ↔ NucBox mesh; no HPC on mesh |
| Sensors | Intel RealSense D456 (RGB-D) | Color + depth; realsense2_camera driver |
| Detection | DINOv2 (ViT-B/14) + SAM2 (Hiera small) | Zero-shot → fine-tuned; box + mask |
| 3D | RealSense depth → 3D bbox (Sprint 2); ROSplat 3DGS (Sprint 3) | Per-tomato 3D for arm; then full-scene 3DGS |
| Acceleration | ROCm (NucBox), MIGraphX (Sprint 3), MPS (Mac host) | Inference on NucBox; optional ONNX on edge |
| Language | Qwen-VL quantized (Sprint 4) | Natural-language pick policy, ripeness/condition |
| Quality | pytest (image_utils), CI (Sprint 2), eval script, REPRODUCE.md | Tests, numbers, reproducibility |

---

## Phased Plan — Step by Step

### Phase 1 — Infrastructure (Done)

- [x] Bazel monorepo, MODULE.bazel, platforms (Mac, ROCm)
- [x] Docker: Dockerfile.dev (Mac), Dockerfile.rocm (NucBox)
- [x] ROS 2 Jazzy, DDS unicast profile, ROS_DOMAIN_ID=42
- [x] Tailscale Mac ↔ NucBox; verify_mesh.sh
- [x] RealSense D456 in compose + ROCm run script; default camera topic
- [x] 2-machine pivot (no HPC); model_sync.sh Mac ↔ NucBox

---

### Phase 2 — Sprint 2: Detection + Eval + 3D + Rigor

**Goal:** Employer-grade detection pipeline: numbers, 3D, CI, one ablation.

#### 2.1 NucBox and data (do first)

- [ ] **S2.1.1** On NucBox: plug RealSense, enter ROCm container, run `realsense2_camera` + `perception.launch.py`; confirm detections and topic rate.
- [ ] **S2.1.2** On NucBox (or Mac): download Laboro Tomato and/or KUTomaData into `data/` (or NucBox `~/AgrobotV2/data/`); document in `data/README.md`.
- [ ] **S2.1.3** Define a small **val split** (e.g. 100–500 images) and list in `data/val_list.txt` or equivalent so the eval script is reproducible.

#### 2.2 Evaluation pipeline

- [ ] **S2.2.1** Add `perception/eval/` (or `tools/eval/`): script that loads val images, runs the detector, computes **mAP@0.5** (or precision/recall at a threshold). Output: one number + optional per-image CSV.
- [ ] **S2.2.2** Same script (or separate): **latency** — mean and p99 ms per frame (preprocess + detect). Log or print.
- [ ] **S2.2.3** Add **REPRODUCE.md** at repo root: how to run eval, expected range of numbers, dataset/split, model weights (DINOv2 hub, SAM2 path).

#### 2.3 Depth → 3D (2D weakness fixed)

- [ ] **S2.3.1** Subscribe to RealSense **depth** (aligned to color) in the detector node or a small fusion node.
- [ ] **S2.3.2** For each 2D detection, read depth at bbox center (or median in a small patch); back-project to **camera-frame 3D** (x, y, z).
- [ ] **S2.3.3** Publish **3D detections** (e.g. custom message or vision_msgs-style) so the planner/arm gets 3D targets. Keep 2D detections for compatibility.

#### 2.4 CI

- [ ] **S2.4.1** Add `.github/workflows/ci.yml`: on push/PR, run Bazel build (e.g. `//perception/...`) and `bazel test //perception:image_utils_test`. Optional: ruff.
- [ ] **S2.4.2** Ensure `bazel test //perception:image_utils_test` passes in Docker (dev image).

#### 2.5 One ablation (technical hook)

- [ ] **S2.5.1** Implement a **DINOv2-only** path (no SAM2): same proposal step, boxes from patches only, no mask refinement.
- [ ] **S2.5.2** Run eval script for (a) DINOv2-only and (b) DINOv2+SAM2 on the same val set. Record mAP/latency in REPRODUCE.md.
- [ ] **S2.5.3** One-sentence “why we use SAM2” (e.g. “SAM2 refines boxes and gives masks; mAP +X%, latency +Y ms”).

#### 2.6 Fine-tuning (NucBox)

- [ ] **S2.6.1** Run SAM2 fine-tuning on NucBox (Laboro Tomato / KUTomaData) using the training deps in Dockerfile.rocm. Save checkpoint to `models/sam2/`.
- [ ] **S2.6.2** Re-run eval with fine-tuned weights; update REPRODUCE.md with before/after numbers.

**Sprint 2 exit criteria:** Eval script runs and produces mAP + latency; 3D detections published; CI green; one ablation documented; fine-tuned model optional but recommended.

---

### Phase 3 — Sprint 3: Performance + 3DGS + Safety

**Goal:** Real-time on NucBox, 3D Gaussian scene, and explicit failure handling.

#### 3.1 Throughput and latency

- [ ] **S3.1.1** MIGraphX/ONNX path for DINOv2 (and optionally SAM2) on ROCm; target **>30 FPS** or p99 &lt; 33 ms.
- [ ] **S3.1.2** Latency breakdown (preprocess, DINOv2, SAM2, postprocess) in logs or a small profiling script.

#### 3.2 ROSplat 3DGS

- [ ] **S3.2.1** Integrate **ROSplat** (or equivalent 3DGS-for-ROS): consume RGB-D + poses, build 3D Gaussian map online.
- [ ] **S3.2.2** Use 3DGS for **3D tomato positions** (e.g. project 2D detections into 3DGS/depth) so the arm has a single, consistent 3D representation.

#### 3.3 Safety and fallback

- [ ] **S3.3.1** Add **docs/FAILURE_MODES.md**: low light, occlusion, no detections, sensor dropout → what the system does (e.g. “no pick”, retry, safe default).
- [ ] **S3.3.2** In code: if no detections or all below confidence threshold → publish empty detections and (if needed) a “no pick” or “skip” signal for the planner.

**Sprint 3 exit criteria:** >30 FPS on NucBox; ROSplat 3D map + 3D tomato outputs; failure-mode doc and code behavior defined.

---

### Phase 4 — Sprint 4: VLM + HIL

**Goal:** Natural-language pick policy and human-in-the-loop testing.

#### 4.1 Qwen-VL

- [ ] **S4.1.1** Deploy **quantized Qwen-VL** on NucBox (ROCm); fit in 96GB with DINOv2+SAM2.
- [ ] **S4.1.2** VLM input: crop(s) from 2D detections (and optionally 3D context). Output: “which tomato” or “ripe/damaged” or short explanation.
- [ ] **S4.1.3** Integrate with perception: operator says “pick the reddest” → VLM selects from current detections → planner gets 3D target.

#### 4.2 HIL testing

- [ ] **S4.2.1** Define HIL protocol: human confirms or overrides picks; log for analysis.
- [ ] **S4.2.2** Run on real row (or representative setup); document results and failure cases.

**Sprint 4 exit criteria:** Qwen-VL drives pick selection from language; HIL runs and is documented.

---

## Next Actions (Start Here)

You are in **Sprint 4**. Sprints 1–3 are complete (MIGraphX GPU deferred — see
`docs/SPRINT3_ROCM_ISSUE.md`). Current perception mAP: **0.377** (S4.12 config).

See `docs/SPRINT4_ARCHITECTURE.md` for the full Sprint 4 perception architecture and
per-step mAP progression.

### Immediate (Howard)

1. **tomato_spatial_node** — Build NODE 2: clip PointCloud2 to each bbox, fit
   sphere (RANSAC algebraic LS), publish `/agrobot/tomato_spatial` JSON so
   Dani's arm planner has 3D centroid + sphere extents. ✅ *Done.*
2. **E6 LoRA DINOv2** — Fine-tune DINOv2 last 4 blocks with LoRA rank-8 on Mac
   (MPS) since ROCm is blocked. Rebuild `query_embedding_k4.pt`. Eval delta
   expected +5–10 mAP points. See `perception/tools/finetune_dino_lora.py`.
3. **Qwen-VL** (S4.1): Deploy quantized Qwen2.5-VL-3B on NucBox. Subscribe to
   `/agrobot/tomato_spatial` (use `clipped_image`). Publish pick selection to
   `/agrobot/pick_target`. This satisfies the Sprint 4 exit criterion.
4. **HIL** (S4.2): Run 5 consecutive pick cycles. Document in `docs/HIL_RESULTS.md`.

---

## Reference: Key Files

| Purpose | Path |
|--------|------|
| Context & roadmap | `.cursor/rules/agrobot-context.mdc` |
| This plan | `docs/GRAND_PLAN.md` |
| Reproducibility | `REPRODUCE.md` (to add in Sprint 2) |
| Failure modes | `docs/FAILURE_MODES.md` (Sprint 3) |
| Model weights | `models/README.md` |
| CI | `.github/workflows/ci.yml` (Sprint 2) |
