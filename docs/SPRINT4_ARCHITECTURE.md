# Sprint 4 — Full Perception Architecture

**Sprint 4 result:** mAP=0.377 on Laboro Tomato val (+121% vs Sprint 3). Full 5-node
picking pipeline implemented and verified end-to-end on NucBox with RealSense D456.

---

## 1. The Complete Pipeline

```
RealSense D456
  │
  ├─ /camera/.../color/image_raw        ──────────────────────────────┐
  ├─ /camera/.../depth/image_rect_raw   ──────────────────────────────┤ NODE 1
  ├─ /camera/.../depth/camera_info      ──────────────────────────────┤
  └─ /camera/.../depth/color/points     ──────────────────────────────┤
                                                                       │
                                                    ┌──────────────────▼─────────────────┐
                                                    │  tomato_detector_node  (NODE 1)    │
                                                    │  SAM2 AMG → 784 mask proposals     │
                                                    │  DINOv2 → scores each mask         │
                                                    │  NMS + confidence filter            │
                                                    └──────────────────┬─────────────────┘
                                                                       │ /agrobot/detections
                                                                       │ /agrobot/safe_to_pick
                                                                       │ /agrobot/debug_image
                                        ┌──────────────────────────────▼───────────────────┐
                                        │  tomato_spatial_node  (NODE 2)                   │
                                        │  clip PointCloud2 to each bbox                   │
                                        │  strip background (z_min + 10cm window)          │
                                        │  RANSAC sphere fit → centroid + radius           │
                                        │  JPEG crop per tomato (for VLM)                  │
                                        └──────────────────────────────┬───────────────────┘
                                                                       │ /agrobot/tomato_spatial
                              ┌────────────────────────────────────────▼─────────────────────┐
                              │  tomato_tracker_node  (NODE 2b)                              │
                              │  Hungarian bipartite matching (scipy.linear_sum_assignment)  │
                              │  persistent_id stable across frames                         │
                              │  EMA centroid smoothing (α=0.4, converges in ~4 frames)     │
                              │  /agrobot/mark_picked suppresses picked tomatoes             │
                              └────────────────────────────────────────┬─────────────────────┘
                                                                       │ /agrobot/tomato_tracks
          ┌────────────────────────────────────────────────────────────▼──────────────────────┐
          │  qwen_vl_node  (NODE 3)                                                           │
          │  Qwen2.5-VL-3B-Instruct (bfloat16, CPU, ~6GB RAM)                                │
          │  multi-image prompt: all JPEG crops → "which tomato should be picked?"            │
          │  reasoning-first response → parse persistent_id → publish PoseStamped             │
          └────────────────────────────────────────────────────────────┬──────────────────────┘
                                                                       │ /agrobot/pick_target
                                                                       │   (geometry_msgs/PoseStamped)
                                                                       │   frame: camera_color_optical_frame
                                                                       ▼
                                                              Dani's arm planner (NODE 4)
```

---

## 2. NODE 1: Detector — How It Works

### Why this architecture

The naive approach (DINOv2 patch proposals) was Sprint 2. DINOv2's patch size is 14px.
A tomato at 0.5m subtends ~60px — its bounding box from patch grid has IoU < 0.3 with
the real tomato, collapsing mAP to near zero. The fix: **SAM2 proposes, DINOv2 scores.**

SAM2's Automatic Mask Generator produces pixel-precise contour masks. DINOv2 — running
once per frame — scores each mask by cosine similarity in feature space. Architecture
swap alone took mAP from 0 to 0.022 (S3.4).

### The full scoring pipeline per frame

```
Input: float32 (3, 518, 518) — ImageNet-normalised, letterboxed
                │
    ┌───────────┴────────────┐
    │                        │
    ▼                        ▼
DINOv2 ViT-B/14         SAM2 AMG (pts=28)
  forward_features()      uniform 28×28 grid
  patch_tokens            → 784 candidate masks
  (1369, 768)             decoded by SAM2 decoder
  L2-normalised           pred_iou per mask
    │                        │
    └───────────┬────────────┘
                │   shared per mask
                ▼
    [1] _mask_to_patch_coverage(seg)
        → w_i = fraction of 14×14 patch cell covered by mask, ∈ [0,1]
        → NOT a hard 0/1 threshold — boundary patches of round tomatoes
          contribute proportionally rather than being silently excluded

    [2] Coverage-weighted cosine similarity vs k=4 prototypes
        sim_k = Σ(w_i × cos(f_i, q_k)) / Σ(w_i)    for each prototype k
        tomato_sim = max(sim_1, sim_2, sim_3, sim_4)
        → Each prototype = one ripeness mode: green/yellow/red/occluded
        → max-over-k: green tomato matches prototype 0 even if prototype 2 is red

    [3] Contrastive negative suppression
        neg_sim = coverage-weighted cosine vs background_embedding
        dino_sim = tomato_sim − λ × neg_sim       (λ = 1.0)
        → background_embedding = mean DINOv2 patch embedding outside GT boxes

    [4] Score fusion
        score = α × dino_sim + (1−α) × pred_iou   (α = 0.7)
        → pred_iou is SAM2's self-assessed mask shape quality
        → 30% SAM2 vote penalises semantically high but geometrically poor masks

    [5] Filter: score ≥ 0.35 → keep
        Box NMS at IoU=0.5, cap at 30 detections
```

### Design choices and why they matter

| Choice | Alternative considered | Why this is better |
|---|---|---|
| SAM2 proposes, DINOv2 scores | DINOv2 proposes | DINOv2 14px grid → IoU too low for mAP; SAM2 gives contour-level masks |
| Float coverage weights (E1) | Binary patch in/out | Binary discards all boundary patches of round tomatoes; float retains proportional contribution |
| k=4 k-means prototypes (E2) | Single mean embedding | Single mean averages green/yellow/red into a vector that represents no mode well; k=4 separates ripeness stages |
| max-over-k scoring | Average-over-k | Average penalises green tomatoes against red-biased prototypes; max selects best matching mode |
| Contrastive negative | No negative | Without it, background foliage (similar hue/texture to green tomatoes) scores too high |
| SAM2 fine-tuned with point prompts (E4) | Box-prompt fine-tune | Runtime AMG uses point prompts; box-prompt fine-tune creates a train/test mismatch |
| Score fusion at α=0.7 (E4) | Pure DINOv2 score | pred_iou filters geometrically fragmented masks that DINOv2 scores high |

### Score formula

$$\text{score} = \alpha \left(\max_k \frac{\sum_i w_i \cos(f_i, q_k)}{\sum_i w_i} - \lambda \frac{\sum_i w_i \cos(f_i, n)}{\sum_i w_i}\right) + (1-\alpha) \cdot \text{pred\_iou}$$

| Symbol | S4.12 value | Meaning |
|---|---|---|
| $f_i$ | — | L2-normalised DINOv2 patch token at position $i$ |
| $w_i$ | $[0, 1]$ | Fraction of 14×14 patch cell $i$ inside SAM2 mask |
| $q_k$ | k=4 cluster centres | k-means++ prototypes from training tomato patches |
| $n$ | background mean | Mean DINOv2 embedding of non-tomato training patches |
| $\alpha$ | **0.7** | DINOv2 weight in fusion |
| $\lambda$ | **1.0** | Contrastive suppression strength |

---

## 3. NODE 2: Spatial — Sphere Fitting

### What it solves

The detector publishes 2D bboxes in 518×518 letterboxed space. Dani's arm planner
needs a 3D target in metres. The depth image gives per-pixel Z values, but a single
pixel sample is noisy. The point cloud gives all 3D points in the camera frame.

### Processing per detection

```
Detection bbox (x1,y1,x2,y2) in 518×518 space
PointCloud2 → Nx3 numpy (XYZ in camera frame, metres)
                │
    [1] clip_points_to_bbox()
        project each 3D point forward: u = X/Z × fx + cx → map to 518 space
        keep only points whose 2D projection falls inside the detection bbox

    [2] filter_cluster_by_depth()
        z_min = nearest point in cluster (= tomato surface)
        keep only points with Z ≤ z_min + 0.10m
        → strips background wall/shelf points that inflate the sphere fit
        → tomato surface is always the nearest geometry in its bbox

    [3] fit_sphere_ransac()
        4-point random subsamples → algebraic LS sphere fit
        linearise: 2·cx·x + 2·cy·y + 2·cz·z + d = x² + y² + z²
        count inliers within 1.5cm of sphere surface
        refine on all consensus inliers
        → robust to noisy boundary points

    [4] compute_extents()
        axis-aligned AABB of inlier cluster
        width/height/depth for gripper aperture sizing

    [5] crop_color_to_bbox()
        reverse letterbox transform → crop in original camera resolution
        cv2.imencode → base64 JPEG for Qwen-VL
```

### Design choices

| Choice | Why |
|---|---|
| RANSAC sphere fit (not centroid of depth pixels) | Single pixel is noisy; sphere fit uses all surface points and is robust to outliers |
| z_min + 10cm depth window before fitting | Without this, background points at higher Z inflate the sphere radius to 10–20cm; tomato surface is always the nearest geometry |
| Algebraic LS (not iterative) | O(n) solve via lstsq; fast enough for CPU, numerically stable |
| base64 JPEG in JSON | Self-contained message — Qwen-VL node needs no additional camera subscription |

---

## 4. NODE 2b: Tracker — Persistent IDs

### What it solves

`tomato_spatial_node` is stateless — `tomato_id` is just detection order (0, 1, 2...),
resetting every frame. Qwen-VL would then say "pick tomato 0" which refers to a
different physical tomato every cycle. The tracker assigns stable IDs.

### Algorithm

```
On each /agrobot/tomato_spatial message:

1. Build cost matrix: C[i][j] = Euclidean distance(track_i.centroid, detection_j.centroid)
   Set C[i][j] = 1e9 when distance > 8cm (threshold)

2. scipy.optimize.linear_sum_assignment(C)
   → globally optimal one-to-one assignment (O(n³), trivial for n < 15)
   → Hungarian algorithm, not greedy nearest-neighbour

3. For matched pairs with cost < threshold:
   EMA update: centroid = 0.4 × new + 0.6 × old
   → convergence in ~4 frames (age ≥ 3 → smoothed=True)

4. Unmatched detections → new track with next persistent_id
   Unmatched tracks → missed_frames++ → drop after 3 missed
```

### Why Hungarian over greedy nearest-neighbour

Greedy claims the locally closest match first, which can steal a track from a better
global assignment. With 5+ clustered tomatoes this causes ID-swaps mid-demo. Hungarian
minimises total distance across all pairs simultaneously — order-invariant, optimal.

### /agrobot/mark_picked

The arm planner publishes `{"persistent_id": N}` to this topic after a successful pick.
The tracker immediately removes that track from the registry and suppresses it from
future `/agrobot/tomato_tracks` messages. Without this, Qwen-VL would keep
recommending the same empty location on the next detection cycle.

---

## 5. NODE 3: Qwen-VL — Language-Conditioned Pick Policy

### Why VLM over a heuristic

A heuristic (pick closest, pick highest score) cannot distinguish a ripe red tomato
from an unripe green one at the same depth. Qwen2.5-VL-3B takes all JPEG crops and
reasons about ripeness, size, and accessibility in natural language:

> "Tomato 0 appears red and fully ripe at 0.38m; tomato 1 is green and further away.
> 0"

### Inference flow

```
/agrobot/tomato_tracks (age ≥ 3, smoothed=True only)
    │
    for each tomato: decode clipped_image (base64 → PIL)
    │
    build multi-image prompt:
      "Policy: prefer red/ripe, then larger, then closer.
       Tomato 0 (z=0.38m, r=4.9cm): [IMAGE]
       Tomato 1 (z=0.64m, r=3.5cm): [IMAGE]
       First explain your reasoning in one sentence,
       then on a new line reply with ONLY the tomato number."
    │
    Qwen2.5-VL-3B.generate(max_new_tokens=64, do_sample=False)
    │
    parse: re.findall(r"\b(\d+)\b", response) → read LAST number
           (first number may appear in reasoning sentence)
    │
    fallback: if parse fails → pick min(z) smoothed tomato
    │
    publish /agrobot/pick_target (geometry_msgs/PoseStamped)
            frame_id: camera_color_optical_frame
            position: EMA-smoothed centroid from tracker
```

### Design choices

| Choice | Why |
|---|---|
| bfloat16 on CPU | ROCm blocked on gfx1151; 3B × 2 bytes = 6GB, fits in 96GB NucBox |
| Only smoothed tracks (age ≥ 3) | Sphere-fit centroid jitters ±3cm at 0.5m; EMA needs ≥3 frames to converge |
| clipped_image in JSON | No additional camera subscription for Qwen-VL; spatial node bundles the crop |
| Reasoning-first prompt | Models produce better final answers when forced to reason before deciding; easier to debug |
| Read LAST number in response | First number may appear in reasoning ("Tomato 0 is closer"); last number is the answer line |
| do_sample=False (greedy) | Deterministic — same scene gives same pick every time; important for HIL reproducibility |

---

## 6. E6 — LoRA DINOv2 (in progress)

### What it changes

Only DINOv2's last 4 transformer blocks get low-rank adapter weights. Everything
else — SAM2, scoring formula, NMS, spatial, tracker, VLM — is identical.

### How LoRA works

Standard attention projection: `h = W₀ × x` (W₀ frozen, 86M total params)

With LoRA (rank=8):
```
h = W₀ × x + (B × A) × x × (α/r)

A ∈ R^{8 × 768}   trainable, Gaussian init
B ∈ R^{768 × 8}   trainable, ZERO init → zero output at epoch 0
                    (model starts identical to pretrained DINOv2)
```

Trainable params: ~147K (0.17% of 87M). The frozen 99.83% preserves universal
visual representations — no catastrophic forgetting on 643 images.

### Training signal (NT-Xent contrastive loss)

```
anchor   = DINOv2 patch from inside GT tomato box
positive = same patch with 20% feature dropout (different "view")
negatives = patches from outside GT boxes (background, leaves, stems)

L = -log( exp(cos(anchor, positive)/0.07) / Σ exp(cos(anchor, z_k)/0.07) )
```

Temperature τ=0.07 is very low — hard contrast, strongly penalises confusions.
After training: `cos(tomato_patch, query)` increases; `cos(background_patch, query)`
decreases. Since the entire scoring pipeline is cosine similarity, every downstream
layer benefits automatically.

**Critical:** after LoRA training, the query embedding must be rebuilt with
`--dino-lora-path models/dino_lora.pt`. The k-means centroids are in the old
feature space and will give wrong scores against the adapted DINOv2.

### Expected mAP gain

| Scenario | mAP | Rationale |
|---|---|---|
| Conservative | 0.39–0.40 | Feature space already decent; marginal separation gain |
| Expected | 0.40–0.43 | LoRA improves tomato/background separation; k=4 centroids in better space |
| Optimistic | 0.43–0.46 | Background false positives were the dominant failure mode |

Recall ceiling (~0.62) is set by SAM2 proposal density, not feature quality — LoRA
improves precision, not recall. Breaking 0.46 requires more proposals (pts=32+)
combined with LoRA-improved precision.

---

## 7. mAP Progression

| Sprint | Change | mAP | Δ |
|---|---|---|---|
| S3.10 | AMG pts=20, box-prompt FT, single-mean query | 0.170 | baseline |
| E1+E2+E4 | Soft coverage + k=4 prototypes + point-prompt FT | 0.287 | +69% |
| pts=24, max=30 | More proposals | 0.328 | +14% |
| Score fusion | α=0.7, conf=0.35 | 0.360 | +10% |
| pts=28 | 784 proposals (28×28 grid) | **0.377** | +5% |
| E6 LoRA | DINOv2 adapters, NT-Xent loss | TBD | expected +5–10% |

The steep early gains fixed fundamental mismatches. Later gains show diminishing
returns from proposal density — the ceiling is now feature discriminability, which
only LoRA can address without more data.

---

## 8. File Map

| File | Role | Status |
|---|---|---|
| `detectors/sam2_amg_detector.py` | NODE 1 — SAM2 AMG + DINOv2 scoring | **Active, S4.12 final** |
| `detectors/dino_sam2_detector.py` | Sprint 2 — DINOv2 proposals + SAM2 | Kept as ablation baseline |
| `detectors/sam2_semantic_detector.py` | E3 — heatmap-guided point prompts | mAP=0.114, slower path |
| `tomato_detector_node.py` | NODE 1 ROS 2 node | **Active** |
| `tomato_spatial_node.py` | NODE 2 — sphere fitting, PointCloud2 | **Active** |
| `tomato_tracker_node.py` | NODE 2b — Hungarian tracking, EMA | **Active** |
| `qwen_vl_node.py` | NODE 3 — Qwen2.5-VL-3B pick selection | **Active** |
| `utils/pointcloud_utils.py` | Sphere fit, depth filter, crop (pure numpy) | **Active** |
| `tools/finetune_sam2_polygon.py` | Point-prompt SAM2 fine-tune (E4) | **Active** |
| `tools/build_query_embedding.py` | k-prototype query + negative builder | **Active** |
| `tools/finetune_dino_lora.py` | LoRA DINOv2 (E6) | **Active — training on NucBox CPU** |
| `tools/compile_migraphx.py` | ONNX → MIGraphX .mxr (E8) | Blocked on ROCm gfx1151 |
| `tools/save_spatial_crops.py` | Save JPEG crops to disk (Mac inspection) | **Active** |
| `tools/test_qwen_vl.py` | Standalone VLM smoke-test on Mac | **Active** |
| `eval/run_eval.py` | mAP + latency evaluation harness | **Active** |
