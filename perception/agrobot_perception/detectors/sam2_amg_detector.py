"""
sam2_amg_detector.py — SAM2 Automatic Mask Generator + DINOv2 semantic scoring.

Why this replaces the DINOv2-proposals approach:
  The previous architecture used DINOv2 patch similarity → connected components
  → bounding boxes as SAM2 prompts. The fundamental problem: connected components
  on a 37×37 patch grid produce boxes at 14px granularity that rarely achieve
  IoU≥0.5 against pixel-precise GT annotations. Fine-tuning SAM2's decoder
  cannot fix spatially misaligned proposals.

This architecture swaps the roles:

  OLD: DINOv2 (proposals) → SAM2 (refinement)
       DINOv2 proposes bad boxes → SAM2 makes them slightly better

  NEW: SAM2 AMG (proposals) → DINOv2 (scoring)
       SAM2 generates pixel-precise mask proposals from a dense point grid
       DINOv2 patch similarity scores each proposal for "tomatoness"

SAM2 Automatic Mask Generator (AMG):
  SAM2 AMG places a regular grid of points across the image (e.g. 8×8 = 64
  points), runs the mask decoder from each point as a prompt, and returns
  pixel-precise binary masks. Each mask is contiguous, pixel-aligned, and
  has a well-defined bounding box — none of the 14px grid quantisation
  problem.

DINOv2 scoring:
  For each SAM2 mask, extract the DINOv2 patch tokens whose receptive fields
  overlap the mask region. Compute mean cosine similarity to the tomato query
  embedding. This score measures "how tomato-like is this SAM2 mask?" and
  serves as the detection confidence.

Score fusion:
  SAM2 AMG returns predicted_iou (mask shape quality) and stability_score per mask.
  We fuse: score = dino_weight * DINOv2_similarity + (1 - dino_weight) * predicted_iou.
  Complementary signals: semantic (tomatoness) + geometric (mask quality).

NMS:
  Overlapping masks for the same tomato are suppressed. Keeps highest-scoring
  per cluster; reduces duplicate false positives.

Interface:
  detect(preprocessed_chw) → list of {"box", "score", "label", "mask"}
  Identical to PlaceholderDetector and DINOv2SAM2Detector — zero node changes.

Runtime on NucBox CPU (expected):
  SAM2 image encoder: ~200 ms (once per frame, shared across all masks)
  SAM2 mask decoder: ~30–50 ms × N_points (batched; points_per_side=8 → 64 masks)
  DINOv2 forward: ~350 ms (once per frame)
  Scoring: ~5 ms
  Total: ~2–4 s/frame on CPU. Acceptable for offline eval.
  With MIGraphX on ROCm (Sprint 3): target <100 ms/frame.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_DINO_PATCH_SIZE = 14
_DINO_INPUT_SIZE = 518
_DINO_GRID = _DINO_INPUT_SIZE // _DINO_PATCH_SIZE  # 37
_DEFAULT_SAM2_CKPT = "models/sam2/sam2.1_hiera_small.pt"
_DEFAULT_SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"
_DEFAULT_QUERY_EMB = "models/query_embedding.pt"

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


def _find_repo_root() -> Path:
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        if (candidate / "MODULE.bazel").exists():
            return candidate
    env_root = os.environ.get("AGROBOT_ROOT")
    if env_root:
        return Path(env_root)
    raise RuntimeError("Cannot find repo root. Set AGROBOT_ROOT or run from the repo.")


class SAM2AMGDetector:
    """Tomato detector using SAM2 AMG proposals scored by DINOv2 similarity.

    Implements the same interface as DINOv2SAM2Detector and PlaceholderDetector:
      detect(preprocessed_chw) → list[dict]
    where each dict has keys: box, score, label, mask.

    Args:
        device: Inference device. Auto-selected if None.
        confidence_threshold: Minimum fused score to keep a mask.
        max_detections: Maximum number of detections per frame.
        points_per_side: SAM2 AMG grid density. 8 → 64 masks, 16 → 256 masks.
            Higher = better recall, slower. Use 8 for CPU, 16+ for GPU.
        min_mask_area: Minimum mask area in pixels (518×518 space) to keep.
            Filters out tiny spurious SAM2 masks.
        dino_score_weight: Weight for DINOv2 in fusion. score = α*dino + (1-α)*pred_iou.
            Default 0.6 — semantic dominates; pred_iou boosts clean masks.
        nms_iou_threshold: IoU threshold for NMS. Overlapping detections above
            this are suppressed. 0 = disable NMS. Default 0.5.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        sam2_checkpoint: Optional[str] = None,
        confidence_threshold: float = 0.3,
        max_detections: int = 20,
        points_per_side: int = 8,
        min_mask_area: int = 100,
        dino_score_weight: float = 0.6,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        self._device = device or _select_device()
        self._conf_threshold = confidence_threshold
        self._max_detections = max_detections
        self._points_per_side = points_per_side
        self._min_mask_area = min_mask_area
        self._dino_score_weight = dino_score_weight
        self._nms_iou_threshold = nms_iou_threshold
        self._repo_root = _find_repo_root()

        ckpt = sam2_checkpoint or str(self._repo_root / _DEFAULT_SAM2_CKPT)
        self._sam2_ckpt = Path(ckpt)

        self._dino: Optional[torch.nn.Module] = None
        self._amg = None
        self._query_embedding: Optional[torch.Tensor] = None

        self._load_models()
        self._load_query_embedding()

    def _load_models(self) -> None:
        # ── DINOv2 ────────────────────────────────────────────────────────────
        logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
        self._dino = torch.hub.load(
            "facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True
        )
        self._dino.eval().to(self._device)
        logger.info("DINOv2 loaded.")

        # ── SAM2 AMG ──────────────────────────────────────────────────────────
        if not self._sam2_ckpt.exists():
            logger.warning(
                "SAM2 checkpoint not found at %s. "
                "SAM2 AMG disabled — no detections will be produced.",
                self._sam2_ckpt,
            )
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2_model = build_sam2(
                _DEFAULT_SAM2_CFG, str(self._sam2_ckpt), device=self._device
            )

            # Load fine-tuned decoder weights if present alongside base checkpoint.
            finetuned = self._sam2_ckpt.parent / "sam2_tomato_finetuned.pt"
            if finetuned.exists():
                state = torch.load(str(finetuned), map_location=self._device)
                sam2_model.load_state_dict(state)
                logger.info("Loaded fine-tuned SAM2 weights from %s.", finetuned)

            self._amg = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=self._points_per_side,
                # Keep only confident, stable masks to reduce noise.
                pred_iou_thresh=0.70,
                stability_score_thresh=0.80,
                # Crop-based multi-scale augmentation — disable for speed on CPU.
                crop_n_layers=0,
                # Post-process: remove masks with <min_mask_area pixels.
                min_mask_region_area=self._min_mask_area,
            )
            logger.info(
                "SAM2 AMG loaded (points_per_side=%d).", self._points_per_side
            )
        except Exception as exc:
            logger.warning("SAM2 AMG failed to load (%s). No detections.", exc)
            self._amg = None

    def _load_query_embedding(self) -> None:
        query_path = self._repo_root / _DEFAULT_QUERY_EMB
        if query_path.exists():
            q = torch.load(str(query_path), map_location=self._device)
            self._query_embedding = F.normalize(q.float(), dim=0)
            logger.info("Loaded query embedding from %s.", query_path)
        else:
            logger.warning(
                "Query embedding not found at %s. "
                "Run perception/tools/build_query_embedding.py first.",
                query_path,
            )

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        """Detect tomatoes using SAM2 AMG proposals scored by DINOv2.

        Args:
            preprocessed_chw: float32 (3, 518, 518) ImageNet-normalised array
                               from preprocess_for_dino(). Same as DINOv2SAM2Detector.

        Returns:
            list of {"box": [x1,y1,x2,y2], "score": float, "label": str, "mask": np.ndarray}
        """
        if self._amg is None or self._query_embedding is None or self._dino is None:
            return []

        # ── Step 1: Reconstruct uint8 RGB image for SAM2 AMG ─────────────────
        # SAM2 AMG needs a uint8 HWC image, not the normalized CHW tensor.
        rgb_float = preprocessed_chw * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
        rgb_uint8 = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
        rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))  # CHW → HWC

        # ── Step 2: DINOv2 forward — patch tokens for scoring ─────────────────
        tensor = torch.from_numpy(preprocessed_chw).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._dino.forward_features(tensor)
        patch_tokens = features["x_norm_patchtokens"].squeeze(0)  # (1369, 768)
        patch_norms  = F.normalize(patch_tokens, dim=1)            # L2-normalised

        # ── Step 3: SAM2 AMG — generate pixel-precise mask proposals ──────────
        # generate() is already @torch.no_grad internally.
        try:
            masks_data = self._amg.generate(rgb_hwc)
        except Exception as exc:
            logger.warning("SAM2 AMG generate() failed: %s", exc)
            return []

        if not masks_data:
            return []

        # ── Step 4: Score each mask with DINOv2 patch similarity ──────────────
        detections = []
        for mask_info in masks_data:
            seg = mask_info["segmentation"]   # bool (518, 518)
            bbox_xywh = mask_info["bbox"]     # [x, y, w, h] in 518×518 space

            if seg.sum() < self._min_mask_area:
                continue

            # Map mask pixels → DINOv2 patch grid indices.
            # Each patch covers a 14×14 pixel block in the 518×518 image.
            # A patch at grid position (gr, gc) covers pixels
            #   rows [gr*14, (gr+1)*14), cols [gc*14, (gc+1)*14).
            # Downsample the mask to 37×37 by taking the majority vote in each cell.
            mask_grid = self._mask_to_patch_grid(seg)  # (37, 37) bool

            if mask_grid.sum() == 0:
                continue

            # Mean cosine similarity of patches inside the mask to query embedding.
            patch_indices = mask_grid.reshape(-1)       # (1369,) bool
            inside_patches = patch_norms[patch_indices]  # (N, 768)
            if inside_patches.shape[0] == 0:
                continue

            dino_sim = float((inside_patches @ self._query_embedding).mean().cpu())

            # Fuse with SAM2 predicted_iou when available (mask shape quality).
            pred_iou = mask_info.get("predicted_iou") or mask_info.get("pred_iou")
            if pred_iou is not None and isinstance(pred_iou, (int, float)):
                alpha = self._dino_score_weight
                score = alpha * dino_sim + (1.0 - alpha) * float(pred_iou)
            else:
                score = dino_sim

            if score < self._conf_threshold:
                continue

            # Box from SAM2's own bbox (pixel-precise, from mask contour).
            x, y, w, h = bbox_xywh
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": score,
                "label": "tomato",
                "mask": seg.astype(np.uint8),
            })

        # NMS to remove overlapping detections for the same tomato.
        if self._nms_iou_threshold > 0 and detections:
            detections = self._nms(detections)

        # Sort by score descending, cap at max_detections.
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections[: self._max_detections]

    def _nms(self, detections: list[dict]) -> list[dict]:
        """Apply Non-Maximum Suppression. Keeps highest-scoring per overlap cluster."""
        if len(detections) <= 1:
            return detections
        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)
        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        xywh = np.zeros_like(boxes)
        xywh[:, 0] = boxes[:, 0]  # x
        xywh[:, 1] = boxes[:, 1]  # y
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self._nms_iou_threshold,
        )
        if len(indices) == 0:
            return detections
        kept = np.asarray(indices).flatten()
        return [detections[int(i)] for i in kept]

    @staticmethod
    def _mask_to_patch_grid(seg: np.ndarray) -> torch.Tensor:
        """Downsample a 518×518 bool mask to the 37×37 DINOv2 patch grid.

        A patch cell is considered "inside" the mask if >50% of its 14×14
        pixels are True. This avoids counting edge patches that are only
        partially covered.
        """
        h, w = seg.shape  # should be 518, 518
        # Crop to exact 518×518 in case SAM2 returns a different size.
        seg_f = seg[:_DINO_GRID * _DINO_PATCH_SIZE, :_DINO_GRID * _DINO_PATCH_SIZE].astype(np.float32)
        # Reshape to (37, 14, 37, 14) and take mean over the 14×14 blocks.
        blocks = seg_f.reshape(_DINO_GRID, _DINO_PATCH_SIZE, _DINO_GRID, _DINO_PATCH_SIZE)
        patch_coverage = blocks.mean(axis=(1, 3))  # (37, 37)
        grid = torch.from_numpy(patch_coverage > 0.5)  # bool tensor
        return grid
