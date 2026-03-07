#!/usr/bin/env python3
"""
run_eval.py — Run the tomato detector on a validation set and report metrics.

Sprint 2: mAP@0.5 (if ground truth provided), latency mean/p99, and per-image
detection counts. Used for ablation (DINOv2-only vs DINOv2+SAM2) and REPRODUCE.md.

Usage (from repo root):
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt --gt-csv data/val_gt.csv
  PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt --detector dino_only

Inside Docker:
  cd /workspace && PYTHONPATH=perception python perception/eval/run_eval.py --val-list data/val_list.txt

Requires: AGROBOT_ROOT or run from repo root so models/sam2/ and torch.hub resolve.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "perception" / "agrobot_perception").exists():
        env_root = os.environ.get("AGROBOT_ROOT")
        if env_root:
            return Path(env_root)
        raise RuntimeError(
            "Cannot find repo root. Set AGROBOT_ROOT or run from AgrobotV2/."
        )
    return root


def _setup_path(repo_root: Path) -> None:
    perception_dir = repo_root / "perception"
    if str(perception_dir) not in sys.path:
        sys.path.insert(0, str(perception_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detector on val set; report latency and optionally mAP@0.5."
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        required=True,
        help="Path to val_list.txt (one image path per line, relative to repo root).",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Optional: val_gt.csv with columns image_path,x1,y1,x2,y2,label for mAP.",
    )
    parser.add_argument(
        "--detector",
        choices=["dino_sam2", "dino_only"],
        default="dino_sam2",
        help="dino_sam2 = DINOv2 + SAM2 refinement; dino_only = DINOv2 proposals only.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default 0.3).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional: write per-image detections to CSV.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup frames before timing (default 2).",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    _setup_path(repo_root)
    os.chdir(repo_root)

    from agrobot_perception.utils.image_utils import preprocess_for_dino
    from agrobot_perception.detectors.dino_sam2_detector import (
        DINOv2SAM2Detector,
        _select_device,
    )
    import cv2

    # Build val image paths
    val_list_path = args.val_list if args.val_list.is_absolute() else repo_root / args.val_list
    if not val_list_path.exists():
        print(f"ERROR: Val list not found: {val_list_path}", file=sys.stderr)
        sys.exit(1)

    image_paths = []
    with open(val_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = repo_root / p
            image_paths.append(p)

    if not image_paths:
        print("No image paths in val list. Add paths to data/val_list.txt.", file=sys.stderr)
        sys.exit(1)

    # Load detector (dino_only = no SAM2 refinement)
    use_sam2 = args.detector == "dino_sam2"
    detector = DINOv2SAM2Detector(
        device=_select_device(),
        confidence_threshold=args.confidence,
        use_sam2=use_sam2,
    )
    input_size = (518, 518)

    # Load ground truth if provided
    gt_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    if args.gt_csv:
        gt_path = args.gt_csv if args.gt_csv.is_absolute() else repo_root / args.gt_csv
        if gt_path.exists():
            with open(gt_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = row.get("image_path", "").strip()
                    if not img_path:
                        continue
                    try:
                        x1, y1, x2, y2 = (
                            float(row["x1"]), float(row["y1"]),
                            float(row["x2"]), float(row["y2"]),
                        )
                    except KeyError:
                        continue
                    key = str(Path(img_path).resolve())
                    gt_by_image.setdefault(key, []).append((x1, y1, x2, y2))

    # Run inference and collect latencies + detections
    latencies_ms = []
    all_detections: list[tuple[Path, list[dict]]] = []

    for i, img_path in enumerate(image_paths):
        if not img_path.exists():
            print(f"Skip missing: {img_path}", file=sys.stderr)
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"Skip unreadable: {img_path}", file=sys.stderr)
            continue

        preprocessed = preprocess_for_dino(bgr, input_size=input_size)

        if i < args.warmup:
            detector.detect(preprocessed)
            continue

        t0 = time.perf_counter()
        dets = detector.detect(preprocessed)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
        all_detections.append((img_path, dets))

    if not latencies_ms:
        print("No frames timed. Add valid image paths to val_list.txt.", file=sys.stderr)
        sys.exit(1)

    # Latency stats
    latencies_ms.sort()
    n = len(latencies_ms)
    mean_ms = sum(latencies_ms) / n
    p99_ms = latencies_ms[int(0.99 * n)] if n >= 100 else latencies_ms[-1]

    print("── Latency ──")
    print(f"  Frames: {n}")
    print(f"  Mean:   {mean_ms:.2f} ms")
    print(f"  p99:    {p99_ms:.2f} ms")
    print(f"  Detector: {args.detector}")

    # mAP@0.5 if we have GT
    if gt_by_image:
        from eval.metrics import compute_ap_iou_threshold
        ap, prec, rec = compute_ap_iou_threshold(all_detections, gt_by_image, iou_threshold=0.5)
        print("── mAP@0.5 ──")
        print(f"  mAP:        {ap:.4f}")
        print(f"  Precision:  {prec:.4f}")
        print(f"  Recall:     {rec:.4f}")
    else:
        total_dets = sum(len(d) for _, d in all_detections)
        print("── Detections ──")
        print(f"  Total: {total_dets} (no GT → no mAP)")

    if args.output_csv:
        out_path = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "num_detections", "latency_ms"])
            for (img_path, dets), lat in zip(all_detections, latencies_ms):
                w.writerow([str(img_path), len(dets), f"{lat:.2f}"])
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
