# Reproducibility — Agrobot TOM v2

---

## Live ROS 2 pipeline — NucBox [4 terminals]

Open four separate terminal windows on the NucBox. Run them in order.

### Terminal 1 — Camera
```bash
./deployment/docker/run_rocm.sh bash

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42

ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  pointcloud.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30
```

Wait for `RealSense Node Is Up!` before starting Terminal 2.

**Fallback (USB 2, or driver rejects 30 FPS):** drop the two `*_profile` lines — driver defaults to ~15 FPS:

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  pointcloud.enable:=true
```

**Note:** Higher camera FPS does **not** speed up the detector (~15–20 s/frame on CPU); it only makes `/camera/...` topics smoother for debugging and other nodes.

If a profile fails, list supported modes after the camera is running: `ros2 param describe /camera/camera rgb_camera.color_profile`.

### Terminal 2 — Detector (NODE 1)
```bash
docker exec -it $(docker ps -q) bash

source /opt/ros/jazzy/setup.bash
colcon build --packages-select agrobot_perception --symlink-install
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=42

ros2 launch agrobot_perception perception.launch.py \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  depth_camera_info_topic:=/camera/camera/depth/camera_info
```

Wait for `TomatoDetectorNode initialized` before proceeding.

> `AGROBOT_FORCE_CPU=1`, `HIP_VISIBLE_DEVICES=-1`, `ROCR_VISIBLE_DEVICES=-1` are baked
> into the launch file via `additional_env` — no need to export them manually.
>
> `colcon build` only needed once per container session (or after code changes).
> Do NOT set `PYTHONPATH` — it breaks `ros2`.

### Terminal 3 — Spatial Node (NODE 2)
```bash
docker exec -it $(docker ps -q) bash

source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=42

ros2 run agrobot_perception tomato_spatial
```

Wait for both of these log lines before proceeding:
```
[INFO] [tomato_spatial]: TomatoSpatialNode initialized. detections='/agrobot/detections' pointcloud='/camera/camera/depth/color/points'
[INFO] [tomato_spatial]: Camera intrinsics cached: fx=... fy=... cx=... cy=... res=640×480
```

Then when the detector fires (~every 17s) you will see:
```
[INFO] [tomato_spatial]: Published 2/2 tomato spatial estimates.
```

> The spatial node subscribes to `/agrobot/detections` (from NODE 1), the point cloud,
> and the color image. It fits a sphere to the point cloud cluster inside each bbox
> and publishes 3D centroid + radius + JPEG crop for each tomato.

### Terminal 4 — Verify
```bash
docker exec -it $(docker ps -q) bash

source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=42

# Camera: expect ~8–15 Hz with color+depth+pointcloud on USB 3
ros2 topic hz /camera/camera/color/image_raw

# Detections every ~17s (SAM2+DINOv2 CPU inference time)
ros2 topic echo /agrobot/detections

# Arm gate signal
ros2 topic echo /agrobot/safe_to_pick

# Sphere-fit 3D localization (NODE 2) — pretty-print both tomatoes
cat > /tmp/show_spatial.py << 'EOF'
import sys, json
raw = sys.stdin.read()
start = raw.index('[')
end = raw.rindex(']') + 1
for t in json.loads(raw[start:end]):
    print(f"\n--- Tomato {t['tomato_id']} ---")
    print(f"  centroid : x={t['centroid']['x']:+.3f}  y={t['centroid']['y']:+.3f}  z={t['centroid']['z']:.3f} m")
    print(f"  radius   : {t['sphere']['radius']*100:.1f} cm")
    print(f"  size     : {t['sphere']['width']*100:.1f} w x {t['sphere']['height']*100:.1f} h x {t['sphere']['depth']*100:.1f} d cm")
    print(f"  score    : {t['confidence']:.3f}")
    print(f"  has_jpeg : {'yes' if t['clipped_image'] else 'no'}")
EOF

ros2 topic echo --full-length /agrobot/tomato_spatial --once | python3 /tmp/show_spatial.py
```

### Terminal 5 — Save JPEG crops to disk (optional, for Mac visualization)
```bash
# On NucBox host first: git pull (gets tools/save_spatial_crops.py)
# Then enter container:
docker exec -it $(docker ps -q) bash

source /opt/ros/jazzy/setup.bash
source /workspace/install/setup.bash
export ROS_DOMAIN_ID=42

python3 tools/save_spatial_crops.py
```

Each time the detector fires, JPEGs are saved to `eval_reports/spatial_crops/` with filenames encoding depth, radius, and score:
```
20260408_143022_f0001_t0_z0.38m_r4.9cm_s0.60.jpg
20260408_143022_f0001_t1_z0.64m_r3.5cm_s0.48.jpg
```

**Pull to Mac over Tailscale (run from Mac terminal):**
```bash
bash tools/pull_crops.sh
# Finder opens automatically to eval_reports/spatial_crops/
```

### Expected output per processed frame (2 tomatoes in scene)
```
# /agrobot/detections
detections:
  - bbox: center=(cx,cy) size=(w,h)  score: 0.6–0.9  label: tomato
  - bbox: center=(cx,cy) size=(w,h)  score: 0.6–0.9  label: tomato

# /agrobot/safe_to_pick
data: True

# /agrobot/tomato_spatial (Terminal 4 pretty-print)
--- Tomato 0 ---
  centroid : x=-0.062  y=+0.012  z=0.382 m
  radius   : 4.9 cm
  size     : 6.7 w x 4.3 h x 6.7 d cm
  score    : 0.597
  has_jpeg : yes

--- Tomato 1 ---
  centroid : x=+0.309  y=+0.003  z=0.642 m
  radius   : 3.5 cm
  size     : 2.6 w x 6.3 h x 3.9 d cm
  score    : 0.480
  has_jpeg : yes

# Spatial node log (Terminal 3)
[INFO] [tomato_spatial]: Published 2/2 tomato spatial estimates.
```

### Known warnings (all safe to ignore)
| Warning | Cause |
|---|---|
| `xFormers is not available` | Optional attention library, no impact |
| `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory` | ROCm driver gap, CPU fallback active |
| `cannot import name '_C' from 'sam2'` | SAM2 C++ extension skipped, results unaffected |
| `Device connected using a 2.1 port. Reduced performance expected.` | Plug into USB 3 port for full 30fps |
| `get_xu(ctrl=1) failed! Device or resource busy` | IMU bandwidth issue on USB 2.1, IMU unused |

---

## Quick start (eval only, no camera)

## Current best (mAP 0.377 — Sprint 4, S4.12)

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 28 \
  --max-detections 30 \
  --confidence 0.35 \
  --nms-iou 0.5 \
  --dino-weight 0.7 \
  --query-embedding models/query_embedding_k4.pt \
  --negative-embedding models/negative_embedding.pt \
  --negative-weight 1.0 \
  --visualize-dir eval_reports/s4_final
```

**Result:** mAP=0.377 | Precision=0.640 | Recall=0.616 | ~19 s/frame (CPU)

**Requirements:** `query_embedding_k4.pt`, `negative_embedding.pt`, `sam2_tomato_finetuned.pt` (point-prompt version, Sprint 4).

View report: `cd eval_reports/s4_final && python3 -m http.server 8000` → http://localhost:8000 (GT=green, TP=cyan, FP=red)

> **NucBox:** `AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=""` required (ROCm blocked on gfx1151). See [docs/SPRINT3_ROCM_ISSUE.md](docs/SPRINT3_ROCM_ISSUE.md).

---

## Models

| Model | Path | How to get |
|-------|------|-----------|
| DINOv2 ViT-B/14 | `~/.cache/torch/hub/` | Auto-downloaded on first run |
| SAM2.1 hiera-small | `models/sam2/sam2.1_hiera_small.pt` | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| Query embedding (k=4) | `models/query_embedding_k4.pt` | `build_query_embedding.py --num-prototypes 4` |
| Negative embedding | `models/negative_embedding.pt` | `build_query_embedding.py --output-negative` |
| Fine-tuned SAM2 (point prompts) | `models/sam2/sam2_tomato_finetuned.pt` | `finetune_sam2_polygon.py` (Sprint 4 version) |

---

## One-time setup

```bash
# Step 1 — SAM2 fine-tune with point prompts (~60-90 min on NucBox CPU)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/finetune_sam2_polygon.py \
  --coco-json data/Laboro-Tomato/annotations/train.json \
  --train-images data/Laboro-Tomato/train/images \
  --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \
  --output models/sam2/sam2_tomato_finetuned.pt \
  --epochs 5

# Step 2 — k=4 prototype query + background-mean negative (~8 min)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding_k4.pt \
  --num-prototypes 4 \
  --output-negative models/negative_embedding.pt

# Step 3 — Mine hard negatives under the k=4 query (~25 min)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/mine_hard_negatives.py \
  --val-list data/val_list.txt --gt-csv data/val_gt.csv \
  --output models/hard_negative_embedding.pt \
  --query-embedding models/query_embedding_k4.pt \
  --amg-points 20 --confidence 0.2 --negative-weight 0.0 --nms-iou 0.5

# Step 4 (E6) — LoRA DINOv2 fine-tune on Mac MPS (~2-4 h for 643 images, 10 epochs)
# No AGROBOT_FORCE_CPU → auto-selects MPS on Apple Silicon
PYTHONPATH=perception \
  python3 perception/tools/finetune_dino_lora.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/dino_lora.pt \
  --epochs 10 --rank 8 --lora-blocks 4

# Step 5 (E6) — Rebuild query embedding with LoRA backbone (~10 min on MPS)
PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding_lora_k4.pt \
  --num-prototypes 4 \
  --output-negative models/negative_embedding_lora.pt \
  --dino-lora-path models/dino_lora.pt

# Step 6 (E6) — Eval with LoRA to measure mAP delta (run on NucBox)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 28 \
  --max-detections 30 \
  --confidence 0.35 \
  --nms-iou 0.5 \
  --dino-weight 0.7 \
  --query-embedding models/query_embedding_lora_k4.pt \
  --negative-embedding models/negative_embedding_lora.pt \
  --negative-weight 1.0 \
  --dino-lora-path models/dino_lora.pt \
  --visualize-dir eval_reports/e6_lora

# Sync Mac ↔ NucBox (run from Mac)
bash tools/network/setup/model_sync.sh --pull   # NucBox → Mac
bash tools/network/setup/model_sync.sh --all    # Mac → NucBox

# Rebuild val_gt.csv if labels change
python3 perception/tools/build_val_gt_csv.py \
  --val-images data/Laboro-Tomato/val/images \
  --val-labels data/Laboro-Tomato/val/labels \
  --output data/val_gt.csv --val-list data/val_list.txt
```

---

## Results — Laboro Tomato val (161 images, 1,996 GT)

| Sprint | Config | mAP | prec | rec | Mean ms |
|--------|--------|-----|------|-----|---------|
| S3.4a | sam2_amg pts=8 | 0.022 | 0.19 | 0.13 | 2017 |
| S3.4b | sam2_amg pts=12 | 0.023 | 0.14 | 0.20 | 3665 |
| S3.6 | + contrastive λ=1.0 | 0.060 | 0.35 | 0.19 | ~3990 |
| S3.7 | pts=16 | 0.091 | 0.34 | 0.28 | 6536 |
| S3.8 | + nms=0.5 | 0.112 | 0.42 | 0.28 | 6329 |
| S3.9 | + polygon FT (box prompts) | 0.139 | 0.52 | 0.27 | 7061 |
| S3.10 | pts=20, single-mean query | 0.170 | 0.50 | 0.34 | 9458 |
| S4.1 | k4 + hard-neg λ=1.2 (poisoned) | 0.070 | 0.79 | 0.09 | 9392 |
| S4.2 | E1+E2+E4: k4 query + bg-mean neg λ=1.0, pts=20, max=20 | 0.287 | 0.72 | 0.42 | 9393 |
| S4.3 | k4 + hard-neg λ=0.5 | 0.135 | 0.31 | 0.50 | 9391 |
| S4.4 | pts=24, max=30, k4 + bg-mean λ=1.0 | 0.328 | 0.70 | 0.49 | 13377 |
| S4.5 | sam2_semantic top-k=48 (under-prompted) | 0.083 | 0.41 | 0.22 | 1772 |
| S4.6 | sam2_semantic top-k=128 | 0.114 | 0.36 | 0.34 | 3716 |
| S4.7 | dino_weight=0.7, conf=0.20 | 0.134 | 0.25 | 0.60 | 13225 |
| S4.8 | dino_weight=1.0, conf=0.15 | 0.335 | 0.66 | 0.54 | 14578 |
| S4.9 | dino_weight=0.7, conf=0.35, pts=24 | 0.360 | 0.68 | 0.56 | 13968 |
| S4.10 | dino_weight=0.7, conf=0.30, pts=24 | <0.360 | — | — | — |
| **S4.12** | **dino_weight=0.7, conf=0.35, pts=28, max=30** | **0.377** | **0.64** | **0.62** | **19081** |
| E6 | + LoRA DINOv2 rank=8, lora_blocks=4 (steps 4-6 above) | TBD | — | — | — |
| S4.13 | pts=32, max=35 (recall-chasing) | 0.378 | 0.61 | 0.67 | 21883 |

### Key insight (S3.4)

DINOv2 proposals snap to a 14px grid → coarse boxes → IoU < 0.5. Fix: **SAM2 AMG proposes, DINOv2 scores**. Architecture swap alone: mAP 0 → 0.022.

### Sprint 4 progression — what each change delivered

| Step | Change | Cumulative mAP |
|------|--------|----------------|
| E4 | Point-prompt SAM2 fine-tune + E1 soft coverage | baseline |
| E2 | k=4 k-means prototypes | 0.287 (+69% vs S3) |
| pts=24, max=30 | More proposals, cap lifted | 0.328 |
| Score fusion | dino_weight=0.7 + conf=0.35 | 0.360 |
| pts=28 | 784 proposals, 18.5px grid spacing | **0.377 (+121% vs S3)** |

**Next unlock:** E6 LoRA DINOv2 (needs ROCm gfx1151 fix). See [docs/SPRINT3_ROCM_ISSUE.md](docs/SPRINT3_ROCM_ISSUE.md) and [docs/SPRINT4_ARCHITECTURE.md](docs/SPRINT4_ARCHITECTURE.md).
