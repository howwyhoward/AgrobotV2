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

## Expected output

- **Latency:** Mean and p99 ms per frame (preprocess + detect). Typical zero-shot: tens to hundreds of ms depending on device (CPU vs MPS vs ROCm).
- **mAP@0.5:** Only when `--gt-csv` is provided. Single class (tomato).
- **Ablation:** Compare `--detector dino_sam2` vs `--detector dino_only` on the same val list; document in this file after you run (e.g. "dino_sam2: mAP=X, mean Y ms; dino_only: mAP=Z, mean W ms").

## Recorded run (Laboro Tomato val, 161 images)

| Config        | Frames | Mean (ms) | p99 (ms) | Detections |
|---------------|--------|-----------|----------|------------|
| dino_sam2, conf=0.3, CPU (Docker) | 159 | 820 | 1487 | 0 |

**Zero detections:** Zero-shot uses a hardcoded red/orange tomato prior; Laboro Tomato val images may not match well at default threshold. Try `--confidence 0.2` or run SAM2 fine-tuning (Sprint 2.6) to get meaningful counts and mAP.

## Updating this file

After running eval on your val set, paste the actual numbers here so others can compare.
