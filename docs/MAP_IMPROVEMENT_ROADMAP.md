# mAP Improvement Roadmap — Agrobot TOM v2

Prioritized experiments to improve detection mAP on CPU while GPU (ROCm 7.3) is blocked.

**Current best:** mAP 0.1701 (S3.10) — sam2_amg + polygon FT, pts=20, conf=0.2, neg=1.0, nms=0.5, prec=0.50, rec=0.34.

**Target:** Maximize mAP on Laboro Tomato val (161 images, 1,996 GT boxes). Deployable levels (50%+ mAP) would require fine-tuning or a trained detector.

---

## Why S3.3 SAM2 fine-tune gave 0.0000 — and why it might work now

S3.3 polygon fine-tune produced mAP 0.0000. Root cause: **proposal geometry**, not mask quality. The old architecture used DINOv2 patch proposals → 14px grid → coarse boxes. A perfect mask inside a misaligned box still has IoU < 0.5 against pixel-precise GT.

**Architecture changed in S3.4:** SAM2 AMG now proposes pixel-precise masks. DINOv2 only scores them. So SAM2 mask quality *directly* affects IoU — better masks → better boxes → higher mAP. **Re-running SAM2 polygon fine-tune is now justified.**

---

## Prioritized experiments

### Tier 1 — Quick wins (no retraining)

| # | Experiment | Command / change | Expected | Notes |
|---|------------|------------------|----------|-------|
| 1 | **NMS** | `--nms-iou 0.5` | +0.01–0.02 mAP | Suppress duplicate detections per tomato. |
| 2 | **Confidence sweep** | `--confidence 0.15` to `0.25` | Tune prec/rec tradeoff | Lower = more recall, more FP. |
| 3 | **Negative weight sweep** | `--negative-weight 0.8` to `1.2` | +0.01–0.03 | S3.6: λ=1.0 best at pts=12; re-check at pts=16. |
| 4 | **More proposals** | `--amg-points 20` or `24` | +0.01–0.03 | Higher recall. ~1.5–2× slower. |

**Baseline re-run (sanity check):**
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 16 \
  --confidence 0.2 \
  --negative-weight 1.0
```

**NMS + confidence sweep:**
```bash
# NMS
... --nms-iou 0.5

# Lower confidence for recall
... --confidence 0.15
```

---

### Tier 2 — Fine-tuning (CPU, ~1–2 h each)

| # | Experiment | Script | Expected | Notes |
|---|------------|--------|----------|-------|
| 5 | **SAM2 polygon fine-tune** | `finetune_sam2_polygon.py` | +0.02–0.05 | COCO polygon GT → round tomato masks. Now relevant with SAM2 AMG proposals. |
| 6 | **Longer SAM2 training** | Same, `--epochs 10` | Possible overfit | Monitor val mAP; stop if loss ↓ but mAP flat. |

**SAM2 polygon fine-tune (full run):**
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/tools/finetune_sam2_polygon.py \
  --coco-json data/Laboro-Tomato/annotations/train.json \
  --train-images data/Laboro-Tomato/train/images \
  --sam2-checkpoint models/sam2/sam2.1_hiera_small.pt \
  --output models/sam2/sam2_tomato_finetuned.pt \
  --epochs 5
```

**Eval with fine-tuned SAM2** (auto-loaded when `sam2_tomato_finetuned.pt` exists next to base ckpt):
```bash
AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" PYTHONPATH=perception \
  python3 perception/eval/run_eval.py \
  --val-list data/val_list.txt \
  --gt-csv data/val_gt.csv \
  --detector sam2_amg \
  --amg-points 16 \
  --confidence 0.2 \
  --negative-weight 1.0
```

---

### Tier 3 — Query / embedding refinement

| # | Experiment | Change | Expected | Notes |
|---|------------|--------|----------|-------|
| 7 | **Stricter query patches** | In `build_query_embedding.py`: only patches with >50% overlap with GT box | Less noise | May reduce recall if too strict. |
| 8 | **Hard-negative mining** | Build negative from patches near GT (e.g. 1–2 patch margin) | Better leaf/stem suppression | Laboro has dense foliage. |
| 9 | **Rebuild embeddings** | Re-run `build_query_embedding.py` with full train set | Sanity check | Ensure no drift from data changes. |

---

### Tier 4 — Larger changes (research)

| # | Experiment | Effort | Expected | Notes |
|---|------------|--------|----------|-------|
| 10 | **DINOv2 fine-tuning** | High | +0.05–0.15 | Train encoder for tomato patches. New training loop, LoRA or full fine-tune. |
| 11 | **Multi-query** | Medium | +0.02–0.05 | Separate embeddings for green vs red tomatoes; score = max. |
| 12 | **Data augmentation** | Medium | +0.02–0.05 | Augment train for query/negative embedding build. |

---

## Recommended order

1. ~~Baseline re-run~~ — Done.
2. ~~Tier 1.1 NMS~~ — S3.8: 0.112 mAP.
3. ~~Tier 2.5 SAM2 polygon fine-tune~~ — S3.9: 0.139 mAP.
4. ~~Tier 1.4 More proposals (pts=20)~~ — S3.10: **0.170 mAP** (current best).
5. **Next:** pts=24 (higher recall) or conf=0.15 (occluded tomatoes).
6. **Tier 3** — Query refinement if gains plateau.

---

## Recording results

Update `REPRODUCE.md` with each run. Latest: S3.10 (pts=20) = 0.1701 mAP. Next: pts=24, conf=0.15.

---

## Reference

- Eval: `perception/eval/run_eval.py`
- SAM2 polygon fine-tune: `perception/tools/finetune_sam2_polygon.py`
- Query embedding: `perception/tools/build_query_embedding.py`
- Detector: `perception/agrobot_perception/detectors/sam2_amg_detector.py`
- REPRODUCE.md: results table and commands
