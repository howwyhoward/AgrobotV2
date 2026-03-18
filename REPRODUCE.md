# Reproducibility — Agrobot TOM v2

## Quick start

| Environment | Command |
|-------------|---------|
| Mac (Docker) | `./compose.sh run --rm dev bash` |
| NucBox (ROCm) | `./deployment/docker/run_rocm.sh bash` |

---

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
  --negative-weight 1.0
```

**Requirements:** `query_embedding_k4.pt`, `negative_embedding.pt`, `sam2_tomato_finetuned.pt` (point-prompt version). ~19 s/frame on NucBox CPU.

Add `--visualize-dir eval_reports/s4_best` for HTML report. View: `cd eval_reports/s4_best && python3 -m http.server 8000` → http://localhost:8000

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
