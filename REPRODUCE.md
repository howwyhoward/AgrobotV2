# Reproducibility

How to reproduce evaluation numbers for the Agrobot TOM v2 perception pipeline.

## Environment

- **Dev (Mac):** Docker image `agrobot-tom-v2/dev:latest`. Run `./compose.sh run --rm dev bash`, then from `/workspace` run eval (see below).
- **NucBox:** Docker image `agrobot-tom-v2/rocm:latest` via `./deployment/docker/run_rocm.sh`.

## Model weights

- **DINOv2:** Fetched automatically by `torch.hub` on first run; cached under `~/.cache/torch/hub/`.
- **SAM2:** Place `sam2.1_hiera_small.pt` at `models/sam2/sam2.1_hiera_small.pt`. Download from `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt`.
- **Query embedding:** Built from Laboro Tomato training set by `perception/tools/build_query_embedding.py`. Saved to `models/query_embedding.pt`. If absent, detector falls back to hardcoded RGB prior (zero detections on most real images).

## Validation set

1. Add image paths to `data/val_list.txt` (one path per line, relative to repo root).
2. Optional: add `data/val_gt.csv` with columns `image_path,x1,y1,x2,y2,label` for mAP@0.5.

See [data/README.md](data/README.md).

## Run evaluation

From repo root (or inside container at `/workspace`). **Use `python3` in Docker** (no `python` alias).

```bash
# Latency only (no ground truth)
PYTHONPATH=perception python3 perception/eval/run_eval.py --val-list data/val_list.txt

# Lower confidence to get more detections on zero-shot (try if Total: 0)
PYTHONPATH=perception python3 perception/eval/run_eval.py --val-list data/val_list.txt --confidence 0.2

# With ground truth → mAP@0.5
PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --gt-csv data/val_gt.csv

# DINOv2-only ablation (no SAM2 refinement)
PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --detector dino_only

# Write per-image CSV
PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --output-csv data/eval_results.csv
```

### NucBox (ROCm container) — force CPU

Pre-built ROCm 6.4 PyTorch wheels target discrete RDNA 3 (gfx1100) and fault on
gfx1151 (Strix Halo) unified memory. Set `AGROBOT_FORCE_CPU=1` to bypass GPU
selection until Sprint 3 (MIGraphX path compiles natively for gfx1151).

```bash
# [NUCBOX] inside ROCm container
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --confidence 0.2

AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --confidence 0.2 --detector dino_only
```

## Expected output

- **Latency:** Mean and p99 ms per frame (preprocess + detect). Typical zero-shot: tens to hundreds of ms depending on device (CPU vs MPS vs ROCm).
- **mAP@0.5:** Only when `--gt-csv` is provided. Single class (tomato).
- **Ablation:** Compare `--detector dino_sam2` vs `--detector dino_only` on the same val list; document in this file after you run (e.g. "dino_sam2: mAP=X, mean Y ms; dino_only: mAP=Z, mean W ms").

## Recorded runs (Laboro Tomato val, 161 images)

| Config | Device | Frames | Mean (ms) | p99 (ms) | Detections | Notes |
|--------|--------|--------|-----------|----------|------------|-------|
| dino_sam2, conf=0.3 | Mac Docker (CPU) | 159 | 820 | 1487 | 0 | First run, cold model load |
| dino_sam2, conf=0.2 | NucBox Docker (CPU) | 159 | 361 | 395 | 0 | SAM2 not loaded (ckpt missing) |
| dino_only, conf=0.2 | NucBox Docker (CPU) | 159 | 365 | 397 | 0 | SAM2 not loaded |
| dino_sam2, conf=0.2 | NucBox Docker (CPU, SAM2 loaded) | 159 | 345 | 384 | 0 | `HIP_VISIBLE_DEVICES=""` |
| dino_only, conf=0.2 | NucBox Docker (CPU, SAM2 loaded) | 159 | 362 | 395 | 0 | `HIP_VISIBLE_DEVICES=""` |
| dino_sam2, conf=0.3 | NucBox Docker (CPU, data-driven embedding) | 159 | 1032 | 1262 | **1439** | S2.6 query embedding from 643 train images |

**Ablation note (zero-shot baseline):** With 0 detections, `dino_sam2` appears faster than `dino_only` because SAM2's `set_image` + `predict` calls are only invoked when proposals exist. With detections present (post S2.6), SAM2 adds ~670 ms per frame on CPU (345 ms → 1032 ms). That overhead disappears in Sprint 3 with MIGraphX on ROCm.

**S2.6 result:** Replacing the 4 hardcoded RGB patches with a mean embedding from 136,655 tomato patches across 643 training images produced **1,439 detections** on 159 val frames at conf=0.3. The prior was the problem, not the architecture.

**Zero detections explained:** The zero-shot query is built from 4 hardcoded red/orange RGB patches. Laboro Tomato val includes green/yellow tomatoes, variable lighting, and partial occlusion — the cosine similarity of those patches against the red/orange prior falls below 0.2 on all 159 frames. This is the expected pre-fine-tuning baseline. Sprint 2.6 replaces the hardcoded prior with a mean embedding computed from the Laboro Tomato training set.

**Sprint 3 target:** ROCm + MIGraphX inference on NucBox (gfx1151 native). Expected: <33 ms mean (>30 FPS). The CPU baseline of ~350 ms = the "before" in the optimization story.

## Updating this file

After running eval on your val set, paste the actual numbers here so others can compare.

---

## S2.6 — Build data-driven query embedding

Run once to replace the hardcoded RGB prior with a mean DINOv2 patch embedding
computed from the 643 Laboro Tomato training images. Takes ~5–8 min on CPU.

```bash
# [NUCBOX] inside ROCm container (or [MAC] host)
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt
```

Smoke-test with 10 images first to confirm it runs:

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/build_query_embedding.py \
  --train-images data/Laboro-Tomato/train/images \
  --train-labels data/Laboro-Tomato/train/labels \
  --output models/query_embedding.pt \
  --max-images 10
```

After building, re-run eval — you should see non-zero detections:

```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --confidence 0.3
```

Add the result row to the Recorded runs table above.
