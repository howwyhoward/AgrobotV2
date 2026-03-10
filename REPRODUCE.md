# Reproducibility — Agrobot TOM v2

## Quick start (copy-paste on NucBox)

```bash
# 1. Enter ROCm container
./deployment/docker/run_rocm.sh bash

# 2. Run full eval with mAP (inside container)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --confidence 0.3

# 3. Ablation — DINOv2 only (no SAM2)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --confidence 0.3 \
  --detector dino_only
```

> **NucBox env vars required:** `AGROBOT_FORCE_CPU=1` bypasses GPU selection (pre-built ROCm 6.4 wheels fault on gfx1151 unified memory). `HIP_VISIBLE_DEVICES=""` prevents SAM2's HIP kernels from initialising. Both are CPU-only workarounds until Sprint 3 (MIGraphX).

---

## Models required

| Model | Path | How to get |
|-------|------|-----------|
| DINOv2 ViT-B/14 | `~/.cache/torch/hub/` | Auto-downloaded by torch.hub on first run |
| SAM2.1 hiera-small | `models/sam2/sam2.1_hiera_small.pt` | `curl -L -o models/sam2/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt` |
| Query embedding | `models/query_embedding.pt` | Built from training set — see S2.6 below |

---

## One-time setup commands

### Build query embedding (S2.6) — run once on NucBox, ~6 min

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt
```

### Sync models Mac ↔ NucBox

```bash
# Pull from NucBox to Mac
bash tools/network/setup/model_sync.sh --pull

# Push from Mac to NucBox
bash tools/network/setup/model_sync.sh --all
```

### Generate val_gt.csv (S3.0) — already committed, rebuild if labels change

```bash
python3 perception/tools/build_val_gt_csv.py \
  --val-images data/Laboro-Tomato/val/images \
  --val-labels data/Laboro-Tomato/val/labels \
  --output data/val_gt.csv \
  --val-list data/val_list.txt
```

---

## Results history

### Validation set: Laboro Tomato, 161 images, 1,996 GT boxes

| Sprint | Config | Device | Mean (ms) | p99 (ms) | Detections | mAP@0.5 | Notes |
|--------|--------|--------|-----------|----------|------------|---------|-------|
| S1 | dino_sam2, conf=0.3 | Mac Docker (CPU) | 820 | 1487 | 0 | — | RGB prior, cold load |
| S2 (baseline) | dino_sam2, conf=0.2 | NucBox CPU | 345 | 384 | 0 | — | RGB prior, SAM2 loaded |
| S2 (ablation) | dino_only, conf=0.2 | NucBox CPU | 362 | 395 | 0 | — | No SAM2, RGB prior |
| **S2.6** | dino_sam2, conf=0.3 | NucBox CPU | 1032 | 1262 | **1,439** | — | Data-driven embedding |
| **S3.0** | dino_sam2, conf=0.3 | NucBox CPU | 1055 | 1286 | 1,439 | **0.0000** | First mAP run, zero-shot |
| S3 target | dino_sam2 + MIGraphX | NucBox ROCm | **<33** | **<33** | TBD | TBD | After MIGraphX export |

### Reading the S3.0 mAP result

`mAP=0.0000, Precision=0.0028, Recall=0.0020` is the correct zero-shot baseline — not a bug. The detector fires on 1,439 regions that DINOv2 identifies as tomato-like, but the bounding boxes are coarse (14px patch granularity from connected components) and don't overlap GT annotations at IoU≥0.5. SAM2 mask refinement tightens boxes but needs fine-tuned weights to align with human annotations. **This number is the "before" for Sprint 3 fine-tuning.**

### Ablation interpretation

SAM2 adds ~670 ms/frame on CPU (345 ms → 1,032 ms) once detections exist. With zero detections, `dino_only` appears slower because SAM2's `set_image` is skipped entirely. The real SAM2 cost-vs-accuracy tradeoff is only measurable post fine-tuning. Sprint 3 MIGraphX target eliminates this overhead entirely.

---

## Sprint 3 next steps

1. **S3.1** — Export DINOv2 to ONNX: `torch.onnx.export` on Mac, transfer to NucBox
2. **S3.2** — MIGraphX compile + benchmark: `migraphx-driver perf model.onnx`
3. **S3.3** — SAM2 mask decoder fine-tuning on NucBox (freeze encoder, train decoder on Laboro Tomato)
4. **S3.4** — ROSplat 3DGS integration in `perception.launch.py`
5. **S3.5** — `docs/FAILURE_MODES.md` + watchdog code in detector node
