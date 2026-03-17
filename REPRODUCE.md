# Reproducibility — Agrobot TOM v2

## Quick start

| Environment | Command |
|-------------|---------|
| Mac (Docker) | `./compose.sh run --rm dev bash` |
| NucBox (ROCm) | `./deployment/docker/run_rocm.sh bash` |

---

## Current best (mAP 0.287 — Sprint 4)

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 20 \
  --confidence 0.2 \
  --nms-iou 0.5 \
  --query-embedding models/query_embedding_k4.pt \
  --negative-embedding models/negative_embedding.pt \
  --negative-weight 1.0
```

Add `--visualize-dir eval_reports/sprint4` for HTML report + annotated images (GT=green, TP=cyan, FP=red). View: `cd eval_reports/sprint4 && python3 -m http.server 8000` → open http://localhost:8000

**Requirements:** `query_embedding_k4.pt`, `negative_embedding.pt`, `sam2_tomato_finetuned.pt` (point-prompt version). ~9.4 s/frame on NucBox CPU.

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
| S3.10 | pts=20 | 0.170 | 0.50 | 0.34 | 9458 |
| S4.1 | E1+E2+E4: soft coverage + k=4 prototypes + point-prompt FT | 0.070 | 0.79 | 0.09 | 9392 |
| **S4.2** | **E1+E2+E4: k4 query + background-mean neg λ=1.0** | **0.287** | **0.72** | **0.42** | **9393** |

### Key insight (S3.4)

DINOv2 proposals snap to a 14px grid → coarse boxes → IoU < 0.5. Fix: **SAM2 AMG proposes, DINOv2 scores**. Architecture swap alone: mAP 0 → 0.022.

### Sprint 4 progression

| Step | Change | Effect |
|------|--------|--------|
| E4 | SAM2 fine-tuned with **point** prompts (matches AMG runtime) | Better mask shapes at IoU@0.5 |
| E2 | k=4 k-means prototypes (green/yellow/red/occluded) | Precision +0.22, recall +0.08 |
| E1 | Soft coverage-weighted cosine scoring | Better calibration for small tomatoes |
| S4.1 (failed) | Hard negatives mined under wrong (single-mean) query at λ=1.2 | Over-suppression: recall 0.09 |
| **S4.2** | **k4 query + background-mean neg λ=1.0** | **mAP 0.170 → 0.287 (+69%)** |

**Next:** E3 semantic detector, pts=24, or conf=0.15 sweep to push recall above 0.50. See [docs/SPRINT4_ARCHITECTURE.md](docs/SPRINT4_ARCHITECTURE.md).
