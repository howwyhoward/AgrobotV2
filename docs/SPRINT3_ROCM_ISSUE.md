# Sprint 3 — ROCm GPU Blocker (NucBox / gfx1151)

**Status: BLOCKED** — MIGraphX GPU benchmark deferred. CPU path fully functional.  
**Last updated:** March 2026  
**Affects:** S3.2 only (`migraphx-driver perf --gpu`). All other tasks unaffected.

---

## Current blocker (TL;DR)

**Kernel 6.14 is working.** WiFi, RealSense (RSUSB), ROCm, and `rocminfo` all succeed.

**MIGraphX GPU inference still segfaults** during "Compiling..." when building the ONNX
graph to gfx1151 kernels. Root cause: **MIOpen Conv2d compilation bug on gfx1151**
(Strix Halo). DINOv2's patch embedding uses Conv2d; MIOpen fails to compile it.
See [rocm-libraries#4070](https://github.com/ROCm/rocm-libraries/issues/4070).

**Fixed-batch export tested:** `dino_vitb14_patches_fixed.onnx` (no dynamic shapes)
still segfaults. Confirms this is upstream ROCm/MIOpen, not our ONNX export.

**No known workaround.** Wait for ROCm 7.3+ or try TheRock nightly builds.

---

## Root cause summary (from AMD platform investigation)

AMD sent their newest integrated GPU platform (Strix Halo, gfx1151). Two separate issues:

1. **Page fault (kernel 6.17):** ROCm 7.2's HIP memory allocator was built for older AMDKFD (AMD Kernel Fusion Driver) behavior. Kernel 6.17 changed that ABI; ROCm couldn't initialize the GPU. **Fix:** Switch to kernel 6.14, which uses the older AMDKFD ABI that ROCm 7.2 expects. `rocminfo` now works and the GPU is detected.

2. **MIOpen Conv2d (gfx1151):** ROCm 7.2 added gfx1151 support, but MIOpen's Conv2d path for this architecture still has bugs. The chip is newer than the current ROCm release; some ops (e.g. Conv2d in DINOv2's patch embedding) aren't fully validated. **Fix:** Wait for ROCm 7.3.

**Bottom line:** CPU for now. Real-time inference unlikely until AMD ships ROCm 7.3 with MIOpen fixes for gfx1151.

---

## What is blocked and why it matters

The Sprint 3 target is **<33 ms/frame** (>30 FPS) inference on the NucBox's Radeon
8060S using MIGraphX — AMD's graph compiler that compiles ONNX models to native
gfx1151 kernels. This is the path from "correct but slow" (~6.5 s CPU) to
"real-time" (goal: <33 ms GPU).

The ONNX model is exported and on NucBox (`models/dino_vitb14_patches.onnx`,
346 MB). MIGraphX 2.15 is installed in the Docker image. The pipeline is ready.
The blocker is MIOpen's Conv2d support on gfx1151, not kernel or ROCm init.

---

## Kernel 6.14 setup (resolved)

NucBox now runs **kernel 6.14.0-37-generic**. Resolution:

| Component | Solution |
|-----------|----------|
| ROCm 7.2 | ✅ Works on 6.14 (was broken on 6.17 with "Page not present") |
| RealSense D456 | ✅ Built librealsense from source with `-DFORCE_RSUSB_BACKEND=TRUE` (no DKMS) |
| WiFi (Mediatek) | ✅ `linux-modules-extra-6.14.0-37-generic` |

The original 3-way conflict (ROCm vs RealSense vs kernel) is resolved. The remaining
blocker is MIOpen Conv2d on gfx1151, which is independent of kernel version.

---

## What was tried (full log)

| Attempt | Result |
|---------|--------|
| `AGROBOT_FORCE_CPU=1` env var | ✅ Fixed PyTorch GPU fault. CPU eval works. |
| `HIP_VISIBLE_DEVICES=""` env var | ✅ Fixed SAM2 HIP kernel init fault. |
| ROCm 6.4 → ROCm 7.2 Docker rebuild | ✅ MIGraphX 2.15 installed. GPU detected by `rocm-smi`. |
| `migraphx-driver perf --gpu` on kernel 6.17 | ❌ `Page not present` during compilation. |
| `migraphx-driver perf --cpu` | ❌ `libmigraphx_cpu.so` not shipped in ROCm 7.2 apt. |
| Install kernel 6.14.0-37-generic | ⚠️ amdgpu DKMS compiled ✅, librealsense2-dkms failed ❌ |
| GRUB switch to 6.14 | ❌ Blocked — cannot remove running kernel (6.17 was active). GRUB numeric index `1>2` and grubenv edits did not take effect due to Ubuntu's HWE auto-boot logic. |
| ROCm 7.2 Docker on kernel 6.14 (never fully tested) | — Blocked by GRUB issue above. |
| Kernel 6.14 + librealsense RSUSB + linux-modules-extra | ✅ WiFi, RealSense, ROCm all work. |
| `migraphx-driver perf --gpu` on 6.14 | ❌ Segfault during "Compiling..." — MIOpen Conv2d bug on gfx1151 (rocm-libraries#4070). |
| `migraphx-driver perf` with `dino_vitb14_patches_fixed.onnx` (--fixed-batch) | ❌ Still segfaults. Confirms upstream, not our export. |

---

## MIGraphX segfault — confirmed upstream

With kernel 6.14 working, `rocminfo` and RealSense succeed. `migraphx-driver perf --gpu`
still segfaults during "Compiling...". This is **not** a kernel/ROCm init issue — it
occurs when MIGraphX compiles the ONNX graph to gfx1151 kernels.

**Root cause:** MIOpen Conv2d compilation failure on gfx1151 (Strix Halo). DINOv2 ViT
uses Conv2d in the patch embedding layer. See [rocm-libraries#4070](https://github.com/ROCm/rocm-libraries/issues/4070).

**Fixed-batch test (March 2026):** Exported `dino_vitb14_patches_fixed.onnx` with
`--fixed-batch` (no dynamic shapes) to rule out MIGraphX dynamic-shape issues. Result:
**still segfaults.** Confirms the blocker is upstream ROCm/MIOpen, not our ONNX export.

---

## Impact on the project

**Blocked:**
- `migraphx-driver perf --gpu` — the standalone MIGraphX benchmark
- Real-time GPU inference (>30 FPS) on NucBox

**Not blocked — everything else works:**
- Full detection pipeline (`run_eval.py`) on CPU — ~6.5 s/frame (sam2_amg + contrastive)
- SAM2 fine-tuning (S3.3) — CPU only, no GPU needed
- RealSense D456 camera — kernel 6.14 with RSUSB backend
- ROSplat 3DGS integration (S3.4) — no GPU dependency
- Failure modes doc + watchdog (S3.5) — no GPU dependency
- All Sprint 4 tasks (Qwen-VL, HIL)

**For demo/competition:** System runs correctly on CPU. At ~0.15 FPS (~6.5 s/frame), you
can pick tomatoes in slow-motion or pre-record detections. The architecture is correct.

---

## Resolution path — Wait for ROCm 7.3 (recommended)

AMD ships ROCm quarterly. ROCm 7.3 may fix the MIOpen Conv2d compilation bug on gfx1151.

**When ROCm 7.3 is released:**
1. Update one line in `Dockerfile.rocm`:
   ```dockerfile
   # Change:
   https://repo.radeon.com/rocm/apt/7.2 noble main
   # To:
   https://repo.radeon.com/rocm/apt/7.3 noble main
   ```
2. Rebuild the Docker image on NucBox:
   ```bash
   docker build --file deployment/docker/Dockerfile.rocm \
     --tag agrobot-tom-v2/rocm:latest --no-cache .
   ```
3. Run the benchmark:
   ```bash
   ./deployment/docker/run_rocm.sh bash
   migraphx-driver perf --onnx models/dino_vitb14_patches.onnx --gpu
   ```

Zero codebase changes. Zero risk. ETA: ~4–8 weeks.

---

## When GPU works: wire it into the eval pipeline

Once `migraphx-driver perf --gpu` produces a `Summary:` output, wire MIGraphX
into `run_eval.py` as the inference backend:

1. Record the `migraphx-driver` summary number in `REPRODUCE.md` (S3.2 row)
2. Add `MIGraphXDetector` class to `perception/agrobot_perception/detectors/`
   that loads `models/dino_vitb14_patches.onnx` via MIGraphX Python API
3. Pass `--detector migraphx` flag to `run_eval.py`
4. Record end-to-end latency and update `REPRODUCE.md` (S3 target row)

The detector interface (`detect()` → `list[dict]`) doesn't change — only the
backend swaps. No node changes required (thin node, thick library pattern).

---

## Reference

- [rocm-libraries#4070 — MIOpen Conv2d failure on gfx1151](https://github.com/ROCm/rocm-libraries/issues/4070) (primary blocker)
- [ROCm/ROCm#5824 — gfx1151 tracking issue](https://github.com/ROCm/ROCm/issues/5824)
- [AMD Strix Halo ROCm optimization guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [ROCm 7.x release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [MIGraphX Python API](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/python.html)
- [librealsense2 GitHub releases](https://github.com/IntelRealSense/librealsense/releases)
