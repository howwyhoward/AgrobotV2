#!/usr/bin/env python3
"""
build_query_embedding.py — Build a DINOv2 tomato query embedding from labeled training images.

Replaces the hardcoded red/orange RGB prior in dino_sam2_detector.py with a
mean patch embedding computed from actual tomato pixels in the Laboro Tomato
training set. The resulting vector is saved to models/query_embedding.pt and
loaded automatically by DINOv2SAM2Detector at startup.

Why this matters:
  The RGB prior fires on red/orange patches. Laboro Tomato has green and yellow
  tomatoes, variable lighting, and cluttered backgrounds — the prior misses almost
  everything. A data-driven embedding encodes shape, texture, and context from
  real tomato images, not just colour.

Usage (from repo root):
  # [MAC] host (uses MPS)
  PYTHONPATH=perception python3 perception/tools/build_query_embedding.py \
    --train-images data/Laboro-Tomato/train/images \
    --train-labels data/Laboro-Tomato/train/labels \
    --output models/query_embedding.pt

  # [NUCBOX] inside ROCm container
  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
    python3 perception/tools/build_query_embedding.py \
    --train-images data/Laboro-Tomato/train/images \
    --train-labels data/Laboro-Tomato/train/labels \
    --output models/query_embedding.pt

Runtime: ~3–8 min on CPU for 643 images (DINOv2 forward pass per image).
Output:  models/query_embedding.pt — float32 tensor of shape (768,), L2-normalised.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → float32 CHW tensor, ImageNet-normalised, resized to 518×518."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE)).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


def _yolo_to_pixel(cx: float, cy: float, w: float, h: float,
                   img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO normalised cx,cy,w,h → pixel x1,y1,x2,y2."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (
        max(0, x1), max(0, y1),
        min(img_w - 1, x2), min(img_h - 1, y2),
    )


def _boxes_to_patch_mask(
    boxes_pixel: list[tuple[int, int, int, int]],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Return a (37,37) boolean mask of DINOv2 patch grid cells that overlap any GT box.

    The 518×518 input is letterboxed from (orig_h, orig_w). Each patch covers
    14×14 pixels in the 518×518 space. We project GT boxes into that space and
    mark all overlapping patch cells.
    """
    scale = min(_DINO_INPUT_SIZE / orig_w, _DINO_INPUT_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (_DINO_INPUT_SIZE - new_w) // 2
    pad_y = (_DINO_INPUT_SIZE - new_h) // 2

    mask = np.zeros((_DINO_GRID, _DINO_GRID), dtype=bool)

    for x1, y1, x2, y2 in boxes_pixel:
        # Scale box into 518×518 letterboxed space
        sx1 = int(x1 * scale) + pad_x
        sy1 = int(y1 * scale) + pad_y
        sx2 = int(x2 * scale) + pad_x
        sy2 = int(y2 * scale) + pad_y

        # Convert to patch grid coordinates
        gx1 = max(0, sx1 // _DINO_PATCH_SIZE)
        gy1 = max(0, sy1 // _DINO_PATCH_SIZE)
        gx2 = min(_DINO_GRID - 1, sx2 // _DINO_PATCH_SIZE)
        gy2 = min(_DINO_GRID - 1, sy2 // _DINO_PATCH_SIZE)

        if gx2 >= gx1 and gy2 >= gy1:
            mask[gy1:gy2 + 1, gx1:gx2 + 1] = True

    return mask


def build_embedding(
    train_images_dir: Path,
    train_labels_dir: Path,
    output_path: Path,
    max_images: int = 0,
) -> None:
    device = _select_device()
    logger.info("Device: %s", device)

    logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
    dino = torch.hub.load("facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True)
    dino.eval().to(device)
    logger.info("DINOv2 loaded.")

    image_files = sorted(train_images_dir.glob("*.jpg")) + \
                  sorted(train_images_dir.glob("*.png"))
    if max_images > 0:
        image_files = image_files[:max_images]

    logger.info("Found %d training images.", len(image_files))

    all_patch_embeddings: list[torch.Tensor] = []
    skipped = 0

    for i, img_path in enumerate(image_files):
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            skipped += 1
            continue

        orig_h, orig_w = bgr.shape[:2]

        # Parse YOLO labels: class cx cy w h
        boxes_pixel = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = (float(p) for p in parts[:5])
                boxes_pixel.append(_yolo_to_pixel(cx, cy, bw, bh, orig_w, orig_h))

        if not boxes_pixel:
            skipped += 1
            continue

        patch_mask = _boxes_to_patch_mask(boxes_pixel, orig_w, orig_h)
        if not patch_mask.any():
            skipped += 1
            continue

        tensor = _preprocess(bgr).unsqueeze(0).to(device)

        with torch.no_grad():
            features = dino.forward_features(tensor)

        # patch_tokens: (1, 1369, 768) → (37, 37, 768)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (1369, 768)
        patch_grid = patch_tokens.reshape(_DINO_GRID, _DINO_GRID, -1)  # (37, 37, 768)

        # Extract only the patches that overlap GT boxes
        tomato_patches = patch_grid[patch_mask]  # (N, 768)
        if tomato_patches.shape[0] == 0:
            skipped += 1
            continue

        all_patch_embeddings.append(tomato_patches.cpu())

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d images, collected %d patch vectors so far.",
                        i + 1, len(image_files),
                        sum(t.shape[0] for t in all_patch_embeddings))

    if not all_patch_embeddings:
        logger.error("No patch embeddings collected. Check --train-images and --train-labels paths.")
        sys.exit(1)

    logger.info("Skipped %d images (no label / unreadable / empty box).", skipped)

    # Stack all tomato patches → mean → L2-normalise → (768,)
    all_patches = torch.cat(all_patch_embeddings, dim=0)  # (Total_patches, 768)
    logger.info("Total tomato patch vectors: %d", all_patches.shape[0])

    mean_embedding = all_patches.mean(dim=0)  # (768,)
    query = F.normalize(mean_embedding, dim=0)  # L2-normalised

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(query, str(output_path))
    logger.info("Saved query embedding to %s (shape=%s)", output_path, query.shape)
    logger.info("Done. Re-run eval to get detections:")
    logger.info("  AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES=\"\" PYTHONPATH=perception "
                "python3 perception/eval/run_eval.py --val-list data/val_list.txt --confidence 0.2")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DINOv2 tomato query embedding from Laboro Tomato training set."
    )
    parser.add_argument(
        "--train-images", type=Path, required=True,
        help="Path to Laboro Tomato train/images/ directory.",
    )
    parser.add_argument(
        "--train-labels", type=Path, required=True,
        help="Path to Laboro Tomato train/labels/ directory (YOLO format .txt files).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/query_embedding.pt"),
        help="Output path for the query embedding tensor (default: models/query_embedding.pt).",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Limit number of images processed (0 = all). Useful for a quick smoke-test.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    train_images = args.train_images if args.train_images.is_absolute() \
        else repo_root / args.train_images
    train_labels = args.train_labels if args.train_labels.is_absolute() \
        else repo_root / args.train_labels
    output = args.output if args.output.is_absolute() else repo_root / args.output

    if not train_images.exists():
        logger.error("train-images dir not found: %s", train_images)
        sys.exit(1)
    if not train_labels.exists():
        logger.error("train-labels dir not found: %s", train_labels)
        sys.exit(1)

    build_embedding(train_images, train_labels, output, max_images=args.max_images)


if __name__ == "__main__":
    main()
