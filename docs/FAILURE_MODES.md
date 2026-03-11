# Failure Modes — Agrobot TOM v2 Perception

How the perception stack behaves under adverse conditions and what operators should do.

---

## Failure mode taxonomy

Every failure maps to one of three system responses:

| Response | What it means | Who handles it |
|----------|--------------|----------------|
| **NO_PICK** | Skip this pick cycle, try next frame | Planner reads `safe_to_pick=False` |
| **DEGRADE** | Continue with reduced information | Node logs warning, planner notified via detection count |
| **STOP** | System cannot safely operate | Operator intervention required |

---

## FM-1 — Camera dropout (no frames arriving)

**Symptom:** `/camera/image_raw` stops publishing. The perception node is alive but receives no input.

**Detection:** Watchdog timer in `TomatoDetectorNode` fires if no frame arrives within `watchdog_timeout_ms` (default: 2000 ms).

**Automatic behavior:**
1. Node publishes empty `Detection2DArray` on `/agrobot/detections`
2. Node publishes `safe_to_pick=False` on `/agrobot/safe_to_pick`
3. Node logs `WARN: No camera frame for Xms — watchdog triggered` at 1 Hz

**Response:** NO_PICK until camera recovers.

**Operator action:** Check USB cable (RealSense D456), verify `realsense2_camera` node is running (`ros2 node list`), restart camera node.

---

## FM-2 — Zero detections (camera running, no tomatoes found)

**Symptom:** Camera publishing, detector running, but `Detection2DArray` is empty.

**Detection:** `detection_count = 0` in `_image_callback`. Normal at the start of a row or between tomato clusters.

**Automatic behavior:**
1. Empty `Detection2DArray` published (explicit signal, not silence)
2. `safe_to_pick=False` published
3. No arm movement initiated

**Response:** NO_PICK. System continues scanning.

**Operator action:** None for short gaps (<5 s). If persistent, check lighting (FM-4) or lower `confidence_threshold` parameter.

---

## FM-3 — Low confidence (detections exist but below threshold)

**Symptom:** Detector finds candidate regions but all scores fall below `confidence_threshold`.

**Detection:** `raw_detections` non-empty but `detections` (filtered list) is empty.

**Automatic behavior:** Same as FM-2 — empty array + `safe_to_pick=False`.

**Response:** NO_PICK. Consider lowering threshold:
```bash
ros2 param set /agrobot/tomato_detector confidence_threshold 0.15
```

**Operator action:** Check scene lighting, verify camera is focused on crop row, not sky or floor.

---

## FM-4 — Low light / poor image quality

**Symptom:** Camera images are dark, blurry, or overexposed. DINOv2 patch features degrade.

**Detection:** No direct sensor — inferred from persistent FM-2/FM-3 conditions in normally tomato-dense scenes.

**Automatic behavior:** FM-2 or FM-3 response (no pick). DINOv2 features are not reliable below ~200 lux.

**Response:** DEGRADE → NO_PICK.

**Operator action:**
- Check and adjust greenhouse lighting
- Inspect camera exposure settings via `realsense2_camera` parameters
- If persistent, log the scene and report to the perception team

---

## FM-5 — Depth sensor failure (3D disabled)

**Symptom:** `/camera/aligned_depth_to_color/image_raw` stops publishing or produces invalid depth values.

**Detection:** `self._depth_image` is `None` (no depth callback received), or all depth values are 0 or `inf`.

**Automatic behavior:**
1. 2D detections (`/agrobot/detections`) continue publishing normally
2. 3D detections (`/agrobot/detections_3d`) stop publishing (conditional on depth being valid)
3. Node logs `WARN: No valid depth — 3D detections disabled`

**Response:** DEGRADE. System continues in 2D-only mode. The arm planner must fall back to a fixed reach depth or halt if it requires 3D.

**Operator action:** Check depth alignment settings in `realsense2_camera`. Restart with `align_depth.enable:=true`.

---

## FM-6 — Model load failure (SAM2 or DINOv2 not loaded)

**Symptom:** Model checkpoint missing or import fails at startup.

**Detection:** `DINOv2SAM2Detector.__init__` or `SAM2AMGDetector.__init__` logs a warning and falls back to `PlaceholderDetector` (returns zero detections).

**Automatic behavior:**
1. Node starts successfully (does not crash)
2. Every frame returns zero detections → FM-2 response
3. Node logs `WARN: SAM2 checkpoint not found` or `WARN: DINOv2 failed to load`

**Response:** NO_PICK until model is restored.

**Operator action:** Verify `models/sam2/sam2.1_hiera_small.pt` exists. Download if missing:
```bash
curl -L -o models/sam2/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

---

## FM-7 — Occlusion (partial tomato visibility)

**Symptom:** A tomato is partially hidden by a leaf, stem, or adjacent fruit. Detection confidence is reduced.

**Detection:** Detection exists but score is near threshold (0.20–0.35). SAM2 AMG may propose a partial mask.

**Automatic behavior:**
1. If score ≥ threshold: detection published, `safe_to_pick=True`
2. If score < threshold: FM-3 response

**Response:** DEGRADE → system picks the most confident detection. If the arm's grasp planner receives a detection with `score < 0.4`, it should execute a slower, more cautious grasp trajectory.

**Operator action:** None. The system handles partial occlusion within its detection confidence model.

---

## FM-8 — Sensor dropout after picking (post-grasp)

**Symptom:** During a pick, the arm occludes the camera field of view, causing a momentary FM-1 or FM-2 condition.

**Detection:** Watchdog triggers if arm motion causes >2 s without detections.

**Automatic behavior:** Watchdog suppresses `safe_to_pick` during arm motion window. The arm controller should signal "in motion" to the perception node to pause watchdog evaluation.

**Response:** NO_PICK for the duration of the arm motion. System resumes detection once the arm retracts.

**Operator action:** None during normal operation. If the arm does not retract and the timeout persists, E-stop.

---

## Summary table

| ID | Failure | Response | Recovery |
|----|---------|----------|----------|
| FM-1 | Camera dropout | NO_PICK | Check USB, restart camera node |
| FM-2 | Zero detections | NO_PICK | Automatic on next frame |
| FM-3 | Low confidence | NO_PICK | Lower threshold or check lighting |
| FM-4 | Low light | NO_PICK | Fix lighting, adjust exposure |
| FM-5 | Depth failure | DEGRADE (2D only) | Restart realsense with align_depth |
| FM-6 | Model load fail | NO_PICK | Restore checkpoint file |
| FM-7 | Occlusion | DEGRADE or NO_PICK | Automatic |
| FM-8 | Arm occlusion | NO_PICK (timed) | Automatic on arm retract |

---

## Reference

- Watchdog implementation: `perception/agrobot_perception/tomato_detector_node.py`
- Safe-to-pick topic: `/agrobot/safe_to_pick` (`std_msgs/Bool`)
- Detection topic: `/agrobot/detections` (`vision_msgs/Detection2DArray`)
- Detector fallback: `PlaceholderDetector` in `tomato_detector_node.py`
