# Reproducibility

How to reproduce evaluation numbers for the Agrobot TOM v2 perception pipeline.

## Environment

- **Dev (Mac):** Docker image `agrobot-tom-v2/dev:latest`. Run `./compose.sh run --rm dev bash`, then from `/workspace` run eval (see below).
- **NucBox:** Docker image `agrobot-tom-v2/rocm:latest` via `./deployment/docker/run_rocm.sh`.

## Model weights

- **DINOv2:** Fetched automatically by `torch.hub` on first run; cached under `~/.cache/torch/hub/`.
- **SAM2:** Place `sam2.1_hiera_small.pt` at `models/sam2/sam2.1_hiera_small.pt`. See [models/README.md](models/README.md).

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
AGROBOT_FORCE_CPU=1 PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --confidence 0.2

AGROBOT_FORCE_CPU=1 PYTHONPATH=perception python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt --confidence 0.2 --detector dino_only
```

## Expected output

- **Latency:** Mean and p99 ms per frame (preprocess + detect). Typical zero-shot: tens to hundreds of ms depending on device (CPU vs MPS vs ROCm).
- **mAP@0.5:** Only when `--gt-csv` is provided. Single class (tomato).
- **Ablation:** Compare `--detector dino_sam2` vs `--detector dino_only` on the same val list; document in this file after you run (e.g. "dino_sam2: mAP=X, mean Y ms; dino_only: mAP=Z, mean W ms").

## Recorded runs (Laboro Tomato val, 161 images)

| Config | Device | Frames | Mean (ms) | p99 (ms) | Detections |
|--------|--------|--------|-----------|----------|------------|
| dino_sam2, conf=0.3 | Mac Docker (CPU) | 159 | 820 | 1487 | 0 |
| dino_sam2, conf=0.2 | NucBox Docker (CPU) | 159 | 361 | 395 | 0 |
| dino_only, conf=0.2 | NucBox Docker (CPU) | 159 | 365 | 397 | 0 |

**Ablation note (zero-shot):** `dino_sam2` and `dino_only` show near-identical latency because SAM2 was not loaded (checkpoint absent). Latency delta will be meaningful once `models/sam2/sam2.1_hiera_small.pt` is present. Add the post-SAM2 row below when available.

**Zero detections:** Zero-shot uses a hardcoded red/orange RGB prior. The Laboro Tomato val set includes green/yellow fruit, variable lighting, and partial occlusion — the prior does not match enough DINOv2 patches above threshold at conf=0.2. This is the expected pre-fine-tuning baseline. Detection counts and mAP will be populated after Sprint 2.6 (SAM2 fine-tuning on NucBox).

**Sprint 3 target:** ROCm + MIGraphX inference on NucBox. Expected: <33 ms mean (>30 FPS). Add that row here when Sprint 3 eval runs.

## Updating this file

After running eval on your val set, paste the actual numbers here so others can compare.
