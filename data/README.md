# Data Directory

Datasets and validation splits for the perception pipeline. Not committed to git (except this README and placeholder lists).

## Layout

- `val_list.txt` — One image path per line (relative to repo root). Pre-filled with **161** Laboro-Tomato val images. Used by `perception/eval/run_eval.py`.
- `val_gt.csv` — (Optional) Ground truth for mAP: `image_path,x1,y1,x2,y2,label`. If present, eval script computes mAP@0.5. Can be generated from `Laboro-Tomato/val/labels/` (YOLO format) with a small script.
- `Laboro-Tomato/` — Laboro Tomato dataset: `train/images`, `train/labels`, `val/images`, `val/labels`, `annotations/`.
- `kutomadata/` — KUTomaData (optional, download separately).

## Getting a validation set

1. Download [Laboro Tomato](https://github.com/laboroai/LaboroTomato) or [KUTomaData](https://github.com/kyushu-univ-fujimura-lab/KUTomaData) into a subdirectory here (or on NucBox under `~/AgrobotV2/data/`).
2. Create a val split (e.g. 100–500 images). List paths in `val_list.txt`.  
   **Already done:** `val_list.txt` contains all 161 Laboro-Tomato val images (`data/Laboro-Tomato/val/images/...`).

3. (Optional) For mAP, add `val_gt.csv` with columns: `image_path,x1,y1,x2,y2,label`.

## RealSense not required

Eval runs on image files. Live camera (S2.1.1) is only needed for end-to-end testing on NucBox.
