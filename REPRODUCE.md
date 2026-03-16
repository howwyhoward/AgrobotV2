# Reproducibility — Agrobot TOM v2

## Quick start

| Environment | Command |
|-------------|---------|
| Mac (Docker) | `./compose.sh run --rm dev bash` |
| NucBox (ROCm) | `./deployment/docker/run_rocm.sh bash` |

---

## Current best (mAP 0.170)

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 20 \
  --confidence 0.2 \
  --negative-weight 1.0 \
  --nms-iou 0.5
```

Add `--visualize-dir eval_reports/run_001` for HTML report + annotated images (GT=green, TP=cyan, FP=red). View: `cd eval_reports/run_001 && python3 -m http.server 8000` → open http://localhost:8000

**Requirements:** `query_embedding.pt`, `negative_embedding.pt`, `sam2_tomato_finetuned.pt`. ~9.5 s/frame on NucBox CPU.

> **NucBox:** `AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=""` required (ROCm blocked on gfx1151). See [docs/SPRINT3_ROCM_ISSUE.md](docs/SPRINT3_ROCM_ISSUE.md).

---

## Models

| Model | Path | How to get |
|-------|------|-----------|
| DINOv2 ViT-B/14 | `~/.cache/torch/hub/` | Auto-downloaded on first run |
| SAM2.1 hiera-small | `models/sam2/sam2.1_hiera_small.pt` | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| Query embedding | `models/query_embedding.pt` | `build_query_embedding.py` |
| Negative embedding | `models/negative_embedding.pt` | `build_query_embedding.py --output-negative` |
| Fine-tuned SAM2 | `models/sam2/sam2_tomato_finetuned.pt` | `finetune_sam2_polygon.py` |

---

## One-time setup

```bash
# Query + negative embeddings (~6 min each)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt \
  --output-negative models/negative_embedding.pt

# SAM2 polygon fine-tune (~60 min)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/finetune_sam2_polygon.py \
  --coco-json data/Laboro-Tomato/annotations/train.json \
  --train-images data/Laboro-Tomato/train/images \
  --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \
  --output models/sam2/sam2_tomato_finetuned.pt \
  --epochs 5

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
| S3.9 | + polygon FT | 0.139 | 0.52 | 0.27 | 7061 |
| **S3.10** | **pts=20** | **0.170** | **0.50** | **0.34** | **9458** |

### Key insight (S3.4)

DINOv2 proposals snap to a 14px grid → coarse boxes → IoU < 0.5. Fix: **SAM2 AMG proposes, DINOv2 scores**. Architecture swap alone: mAP 0 → 0.022.

### S3.8–S3.10 progression

| Step | Change | Effect |
|------|--------|--------|
| S3.8 | NMS IoU=0.5 | Suppress duplicates → higher precision |
| S3.9 | SAM2 polygon fine-tune | Round tomato masks → better IoU |
| S3.10 | pts=20 (400 proposals) | Higher recall in clutter/occlusion |

**Next:** pts=24 or conf=0.15. See [docs/MAP_IMPROVEMENT_ROADMAP.md](docs/MAP_IMPROVEMENT_ROADMAP.md).
