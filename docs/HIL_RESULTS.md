# HIL Testing Results — Agrobot TOM v2

Hardware-In-the-Loop (HIL) testing documents the full camera→perception→VLM→arm
pipeline running on real hardware with a human operator in the loop.

Sprint 4 exit criterion: **5 consecutive pick cycles completed and documented.**

---

## Protocol

### Setup
- NucBox with RealSense D456 camera pointed at real tomatoes
- All 5 nodes running: detector → spatial → tracker → qwen_vl → (arm planner TBD)
- Human operator observes `/agrobot/pick_target` and `/agrobot/vlm_reasoning`
- Human confirms or overrides the VLM selection before arm moves

### Cycle definition
One cycle = one complete pass through the pipeline:
1. Camera frame captured
2. Detector fires (SAM2 AMG + DINOv2) → `/agrobot/detections`
3. Spatial node fits spheres → `/agrobot/tomato_spatial`
4. Tracker assigns persistent IDs → `/agrobot/tomato_tracks` (age ≥ 3)
5. Qwen-VL selects pick target → `/agrobot/pick_target`
6. Operator records: VLM reasoning, selected tomato, human judgement (agree/override)
7. (When arm planner available) Arm moves and gripper closes

### Pass criteria per cycle
- `/agrobot/detections` publishes ≥ 1 detection with score ≥ 0.35
- `/agrobot/tomato_spatial` publishes for all detections (sphere fit valid)
- `/agrobot/tomato_tracks` has ≥ 1 smoothed track (age ≥ 3)
- `/agrobot/pick_target` is published with `z` in [0.2, 1.5] m
- VLM reasoning is coherent (human readable, relevant to the image)

### Failure modes
- **FM-1**: No detections → `safe_to_pick=False`, no pick target published
- **FM-2**: Sphere fit invalid (radius > 0.12 m) → tomato skipped
- **FM-3**: VLM parse failure → heuristic fallback (closest smoothed tomato)
- **FM-4**: Qwen-VL not loaded → heuristic fallback logged

---

## Results

### Test environment
- Date: ___________
- Location: ___________
- Tomatoes: ___ in scene, variety: ___________
- Camera distance: ~___ m

### Cycle log

| Cycle | Detections | Spatial | Tracks | VLM pick | VLM reasoning | Human agree? | Notes |
|-------|-----------|---------|--------|----------|---------------|-------------|-------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |

**Column guide:**
- **Detections**: count published on `/agrobot/detections`
- **Spatial**: count published on `/agrobot/tomato_spatial`
- **Tracks**: count with `smoothed=True` on `/agrobot/tomato_tracks`
- **VLM pick**: `persistent_id` published on `/agrobot/pick_target`
- **VLM reasoning**: first sentence of `/agrobot/vlm_reasoning`
- **Human agree?**: Y / N / Override (with corrected pick)

### Summary

| Metric | Value |
|--------|-------|
| Cycles completed | / 5 |
| Detection success rate | % |
| Spatial success rate | % |
| VLM agreement rate | % |
| Mean detector latency | s |
| Mean VLM latency | s |
| Pipeline failures | |

---

## Verify commands (Terminal 6)

```bash
# Watch all pipeline topics live
ros2 topic echo /agrobot/detections &
ros2 topic echo /agrobot/pick_target &
ros2 topic echo /agrobot/vlm_reasoning

# Per-cycle snapshot (run after each cycle)
cat > /tmp/hil_snapshot.py << 'EOF'
import subprocess, json, datetime

def echo_once(topic, full=False):
    cmd = ["ros2", "topic", "echo", topic, "--once"]
    if full:
        cmd.insert(3, "--full-length")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.stdout.strip()

print(f"\n=== HIL Snapshot {datetime.datetime.now().strftime('%H:%M:%S')} ===")
print("\n-- Detections --")
print(echo_once("/agrobot/detections"))
print("\n-- Pick target --")
print(echo_once("/agrobot/pick_target"))
print("\n-- VLM reasoning --")
print(echo_once("/agrobot/vlm_reasoning"))
EOF
python3 /tmp/hil_snapshot.py
```

---

## Notes and failure cases

_Fill in after each test run._
