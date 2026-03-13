# Reproducibility — Agrobot TOM v2

## Quick start (NucBox, inside ROCm container)

```bash
./deployment/docker/run_rocm.sh bash
```

**Current best detector (`sam2_amg` + contrastive):**
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --confidence 0.2 \
  --detector sam2_amg \
  --amg-points 16 \
  --negative-weight 1.0
```
Requires `models/query_embedding.pt` and `models/negative_embedding.pt` (build with `--output-negative`). ~6.5 s/frame on CPU.

**Legacy detector (research baseline):**
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --confidence 0.3
```

> **NucBox env vars required:** `AGROBOT_FORCE_CPU=1` bypasses GPU selection (pre-built ROCm wheels fault on gfx1151). `HIP_VISIBLE_DEVICES=""` prevents SAM2 HIP kernel init. GPU path blocked by kernel ABI conflict — see [docs/SPRINT3_ROCM_ISSUE.md](docs/SPRINT3_ROCM_ISSUE.md).

---

## Models required

| Model | Path | How to get |
|-------|------|-----------|
| DINOv2 ViT-B/14 | `~/.cache/torch/hub/` | Auto-downloaded by `torch.hub` on first run |
| SAM2.1 hiera-small | `models/sam2/sam2.1_hiera_small.pt` | `curl -L -o models/sam2/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` |
| Query embedding | `models/query_embedding.pt` | Built from Laboro Tomato training set — see below |
| Negative embedding | `models/negative_embedding.pt` | Required for best. Built with `--output-negative` for contrastive scoring |
| Fine-tuned SAM2 decoder | `models/sam2/sam2_tomato_finetuned.pt` | Run `finetune_sam2_polygon.py` — auto-loaded when present |

---

## One-time setup

### Build query embedding — ~6 min on NucBox CPU
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt
```

### Build negative embedding (contrastive scoring) — required for best, ~6 min
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt \
  --output-negative models/negative_embedding.pt
```

### Fine-tune SAM2 decoder on COCO polygon masks — ~60 min on NucBox CPU
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/finetune_sam2_polygon.py \
  --coco-json data/Laboro-Tomato/annotations/train.json \
  --train-images data/Laboro-Tomato/train/images \
  --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \
  --output models/sam2/sam2_tomato_finetuned.pt \
  --epochs 5
```

### Sync models Mac ↔ NucBox
```bash
bash tools/network/setup/model_sync.sh --pull   # NucBox → Mac
bash tools/network/setup/model_sync.sh --all    # Mac → NucBox
```

### Rebuild val_gt.csv (if labels change — already committed)
```bash
python3 perception/tools/build_val_gt_csv.py \
  --val-images data/Laboro-Tomato/val/images \
  --val-labels data/Laboro-Tomato/val/labels \
  --output data/val_gt.csv --val-list data/val_list.txt
```

---

## Results — Laboro Tomato val set (161 images, 1,996 GT boxes)

| Sprint | Detector | Config | Device | Mean ms | p99 ms | mAP@0.5 | Notes |
|--------|----------|--------|--------|---------|--------|---------|-------|
| S1 | dino_sam2 | conf=0.3 | Mac CPU | 820 | 1487 | — | RGB prior, 0 detections |
| S2 baseline | dino_sam2 | conf=0.2 | NucBox CPU | 345 | 384 | — | RGB prior, SAM2 loaded |
| S2 ablation | dino_only | conf=0.2 | NucBox CPU | 362 | 395 | — | No SAM2 |
| S2.6 | dino_sam2 | conf=0.3 | NucBox CPU | 1032 | 1262 | 0.0000 | Data-driven embedding, 1,439 detections |
| S3.0 | dino_sam2 | conf=0.3 + GT | NucBox CPU | 1055 | 1286 | **0.0000** | First mAP measurement; coarse 14px proposals fail IoU@0.5 |
| S3.1 | — | DINOv2 ONNX export | Mac CPU | — | — | — | 346 MB opset-17 graph |
| S3.2 | — | MIGraphX GPU | NucBox ROCm 6.4 | — | — | — | **Blocked**: gfx1151 kernel ABI conflict — needs ROCm 7.3 |
| S3.3 | dino_sam2 | Polygon fine-tune | NucBox CPU | — | — | 0.0000 | Decoder overfit; root cause was proposal geometry, not mask quality |
| **S3.4a** | **sam2_amg** | pts=8, conf=0.3 | NucBox CPU | 2017 | 2493 | **0.0222** | Architecture fix: SAM2 proposes, DINOv2 scores. First real mAP. prec=0.19, rec=0.13 |
| **S3.4b** | **sam2_amg** | pts=12, conf=0.2 | NucBox CPU | 3665 | 4321 | **0.0233** | Baseline: DINOv2-only scoring. prec=0.14, rec=0.20 |
| **S3.6** | **sam2_amg** | pts=12, conf=0.2, neg λ=0.5 | NucBox CPU | ~3970 | ~4680 | **0.0509** | Contrastive. prec=0.29, rec=0.19 |
| S3.6 ablation | sam2_amg | pts=12, conf=0.2, neg λ=0.6 | NucBox CPU | ~4320 | ~5525 | **0.0550** | prec=0.31, rec=0.19 |
| S3.6 ablation | sam2_amg | pts=12, conf=0.2, neg λ=1.0 | NucBox CPU | ~3990 | ~5220 | **0.0601** | prec=0.35, rec=0.19 |
| **S3.7** | **sam2_amg** | pts=16, conf=0.2, **neg λ=1.0** | NucBox CPU | 6536 | 8869 | **0.0907** | Best: 256 proposals. prec=0.34, rec=0.28 |
| S3 GPU target | sam2_amg + MIGraphX | pts=32 | NucBox ROCm 7.x | <100 | <100 | >0.15 | After ROCm 7.3 — 1024 proposals, full GPU acceleration |

### Key insight: why mAP was 0 until S3.4

DINOv2 proposals snap to a 14px patch grid. A small tomato spans 2–3 patches → coarse bounding box → IoU < 0.5 against pixel-precise GT. SAM2 fine-tuning cannot fix spatially misaligned proposals. The fix was swapping roles: **SAM2 AMG generates pixel-precise proposals, DINOv2 scores them for tomatoness**. mAP jumped from 0 to 0.022 with no weight changes — architecture was the bottleneck.

### S3.6: Contrastive negative embedding

**Change:** Score = tomato_sim − λ × negative_sim. Negative embedding built from patches *outside* GT boxes (leaves, stems, background). Suppresses false positives that score high on tomato but also high on negative.

**Sweep:** λ=0.5 → 0.051 mAP; λ=0.6 → 0.055; λ=1.0 → 0.060 (best at pts=12).

### S3.7: More proposals (pts=16)

**Change:** 256 SAM2 masks per frame (vs 144). Higher recall.

**Result:** mAP 0.091, prec 0.34, rec 0.28. ~4× improvement over baseline (S3.4b).

### Proof of concept

**Foundation models can do agricultural object detection with minimal training.** SAM2 for pixel-precise proposals, DINOv2 for semantic scoring, contrastive negative embedding to suppress leaf/stem. No detector training — only query + negative embeddings from patches. Runs on CPU (~6.5 s/frame). Deployable levels (50%+ mAP) would require fine-tuning or a trained detector.
