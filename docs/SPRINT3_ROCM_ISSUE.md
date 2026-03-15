# Sprint 3 — ROCm GPU Blocker (NucBox / gfx1151)

**Status: BLOCKED** — MIGraphX GPU benchmark deferred. CPU path fully functional.  
**Last updated:** March 2026  
**Affects:** S3.2 only (`migraphx-driver perf --gpu`). All other tasks unaffected.

---

## What is blocked and why it matters

The Sprint 3 target is **<33 ms/frame** (>30 FPS) inference on the NucBox's Radeon
8060S using MIGraphX — AMD's graph compiler that compiles ONNX models to native
gfx1151 kernels. This is the path from "correct but slow" (1,032 ms CPU) to
"real-time" (goal: <33 ms GPU).

The ONNX model is exported and on NucBox (`models/dino_vitb14_patches.onnx`,
346 MB). MIGraphX 2.15 is installed in the Docker image. The pipeline is ready.
The only missing piece is a kernel version that satisfies all three drivers
simultaneously.

---

## The 3-way kernel conflict

The NucBox runs kernel **6.17.0-14-generic** (Ubuntu 24.04 HWE, shipped with machine).

| Driver | Kernel 6.14 | Kernel 6.17 | Why |
|--------|------------|------------|-----|
| ROCm 7.2 / MIGraphX GPU | ✅ | ❌ | ROCm 7.2 was released before 6.17. Its HIP memory allocator has hardcoded ABI assumptions that 6.17's AMDKFD changed. Result: `Page not present` fault during GPU kernel JIT compilation. |
| RealSense D456 (`librealsense2-dkms 1.3.28`) | ❌ | ✅ | Intel's UVC driver patch doesn't compile against 6.14 kernel headers. The DKMS build fails with exit code 2. |
| `amdgpu` DKMS (AMD's out-of-tree driver) | ✅ | ✅ (built-in) | AMD maintains a DKMS module that compiled cleanly for 6.14. |

**There is no kernel in Ubuntu 24.04's repos (as of March 2026) that satisfies all three.**

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
| `migraphx-driver perf --gpu` on 6.14 | ❌ Segfault during "Compiling..." — likely MIOpen Conv2d bug on gfx1151 (rocm-libraries#4070). |

---

## MIGraphX segfault (post-kernel fix)

With kernel 6.14 working, `rocminfo` and RealSense succeed. `migraphx-driver perf --gpu` still
segfaults during compilation. This is **not** a kernel/ROCm init issue — it occurs when
MIGraphX compiles the ONNX graph to gfx1151 kernels.

**Likely cause:** MIOpen Conv2d compilation failure on gfx1151 (Strix Halo). DINOv2 ViT
uses Conv2d in the patch embedding layer. See [rocm-libraries#4070](https://github.com/ROCm/rocm-libraries/issues/4070).

**Code workaround to try:** Export with `--fixed-batch` to rule out MIGraphX dynamic-shape
issues:
```bash
# [MAC] Re-export with fixed batch
PYTHONPATH=perception python3 perception/tools/export_dino_onnx.py \
  --output models/dino_vitb14_patches_fixed.onnx --fixed-batch
# Sync to NucBox, then:
migraphx-driver perf --onnx models/dino_vitb14_patches_fixed.onnx --gpu
```
If segfault persists, it is upstream ROCm/MIOpen, not the export.

---

## Impact on the project

**Blocked:**
- `migraphx-driver perf --gpu` — the standalone MIGraphX benchmark
- Real-time GPU inference (>30 FPS) on NucBox

**Not blocked — everything else works:**
- Full detection pipeline (`run_eval.py`) on CPU — 1,032 ms/frame, 1,439 detections
- SAM2 fine-tuning (S3.3) — CPU only, no GPU needed
- RealSense D456 camera — kernel 6.17 works fine
- ROSplat 3DGS integration (S3.4) — no GPU dependency
- Failure modes doc + watchdog (S3.5) — no GPU dependency
- All Sprint 4 tasks (Qwen-VL, HIL)

**For demo/competition:** System runs correctly on CPU. At 1 FPS (1,032 ms), you can
pick tomatoes in slow-motion or pre-record detections. The architecture is correct.

---

## Resolution paths

### Path A — Wait for ROCm 7.3 (recommended)

AMD ships ROCm quarterly. ROCm 7.3 is expected to add kernel 6.17 ABI support.

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

### Path B — Upgrade librealsense2-dkms for kernel 6.14 (~2–4 hours)

Fix the RealSense driver to compile on 6.14, then switch kernels. This unblocks GPU without waiting for AMD.

**Step 1 — Check if a newer version is available via apt:**
```bash
sudo apt-cache policy librealsense2-dkms
```
If the latest is still 1.3.28, proceed to Step 2. If newer, just `sudo apt-get upgrade librealsense2-dkms` and skip to Step 4.

**Step 2 — Build librealsense from Intel's GitHub:**
```bash
sudo apt-get remove librealsense2-dkms
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.1    # or latest tag: git tag | sort -V | tail -5
sudo ./scripts/setup_udev_rules.sh
sudo ./scripts/patch-realsense-ubuntu-lts-hwe.sh
```

**Step 3 — Verify the DKMS module builds for 6.14:**
```bash
sudo dkms build librealsense2-dkms/<version> -k 6.14.0-37-generic
```
If this succeeds, the RealSense driver will work on 6.14.

**Step 4 — Switch to kernel 6.14:**
Boot into 6.14 via physical GRUB menu at the machine (hold Shift during boot →
Advanced options → Ubuntu with Linux 6.14.0-37-generic). Then remove 6.17 from
the running 6.14 system:
```bash
sudo apt-get remove --purge linux-image-6.17.0-14-generic linux-image-generic-hwe-24.04
sudo update-grub
```

**Step 5 — Run MIGraphX benchmark:**
```bash
cd ~/AgrobotV2
./deployment/docker/run_rocm.sh bash
migraphx-driver perf --onnx models/dino_vitb14_patches.onnx --gpu
```

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

- [AMD Strix Halo ROCm optimization guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [ROCm 7.x release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [MIGraphX Python API](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/python.html)
- [librealsense2 GitHub releases](https://github.com/IntelRealSense/librealsense/releases)
- [ROCm/ROCm#5824 — gfx1151 tracking issue](https://github.com/ROCm/ROCm/issues/5824)
