# Sprint 3 — ROCm 7.x Upgrade Guide (NucBox / gfx1151)

## Why this document exists

MIGraphX 2.12 (ROCm 6.4) ships pre-compiled HIP kernels for discrete RDNA 3
targets (gfx1100 / RX 7900 series). The NucBox has a Ryzen AI Max 395 with
Radeon 8060S (gfx1151 / Strix Halo), which uses unified CPU+GPU memory — no
discrete VRAM. The ROCm 6.4 GPU allocator attempts to map pages using the
discrete-VRAM addressing model and faults on Strix Halo's unified memory
controller. This affects both PyTorch ROCm wheels and MIGraphX.

**Symptoms observed:**
- `torch` GPU tensors → `Memory access fault by GPU node-1 ... Page not present`
- `migraphx-driver perf --gpu` → same fault during kernel compilation
- `migraphx-driver perf --cpu` → `libmigraphx_cpu.so: No such file or directory`
  (CPU backend not shipped in ROCm 6.4 apt repo for Ubuntu 24.04)

ROCm 7.x adds native gfx1151 support (Strix Halo), proper unified memory GTT
management, and ships MIGraphX with gfx1151-compiled kernels.

---

## Current blocker — 3-way kernel conflict (as of March 2026)

The NucBox shipped with kernel **6.17.0-14-generic** (Ubuntu HWE). Attempting
to switch to 6.14 (ROCm 7.2's tested target) reveals a 3-way incompatibility:

| Requirement | Kernel 6.14 | Kernel 6.17 |
|-------------|------------|------------|
| ROCm 7.2 GPU (MIGraphX) | ✅ ABI match | ❌ `Page not present` fault |
| RealSense D456 (librealsense2-dkms 1.3.28) | ❌ DKMS compile fails | ✅ works |
| amdgpu DKMS | ✅ compiles | ✅ built-in |

There is no single kernel version available in Ubuntu 24.04's repos (as of March
2026) that satisfies all three simultaneously.

**What was attempted:**
1. `AGROBOT_FORCE_CPU=1` — fixed PyTorch GPU fault, CPU eval works ✅
2. `HIP_VISIBLE_DEVICES=""` — fixed SAM2 HIP init fault ✅
3. ROCm 6.4 → ROCm 7.2 Docker upgrade — correct direction, MIGraphX 2.15 installed ✅
4. Kernel 6.14 install — amdgpu DKMS compiled, but librealsense2-dkms 1.3.28 failed ❌
5. GRUB kernel switch — blocked because kernel 6.17 was active at time of removal attempt

**Resolution path (choose one):**

**Option A — Wait for ROCm 7.3 (recommended, ~4–8 weeks)**
AMD is shipping ROCm updates quarterly. ROCm 7.3 is expected to add kernel 6.17
ABI support. When released: update the apt repo in `Dockerfile.rocm` to `7.3`,
rebuild, and `migraphx-driver perf --gpu` works. Zero codebase changes required.

**Option B — Upgrade librealsense2-dkms to a version supporting kernel 6.14**
Check if Intel has released a newer librealsense2-dkms that compiles on 6.14:
```bash
sudo apt-cache policy librealsense2-dkms
# or check: https://github.com/IntelRealSense/librealsense/releases
```
If a compatible version exists, upgrade it, switch to 6.14, and GPU works.

**Impact on project:** GPU inference path is deferred. CPU baseline (1,032 ms/frame)
is the current production path. All other Sprint 3 tasks (fine-tuning, 3DGS,
failure modes) proceed without GPU. The GPU number fills in automatically once
the blocker resolves.

---

## What needs to happen on the NucBox host

### 1. Verify kernel version

```bash
uname -r
```

ROCm 7.x requires kernel **≥ 6.14** with gfx1151 AMDKFD patches.
Check: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html

If kernel < 6.14, upgrade first:
```bash
sudo apt-get install linux-generic-hwe-24.04
sudo reboot
```

### 2. Remove ROCm 6.4 apt repo

```bash
sudo rm /etc/apt/sources.list.d/rocm.list
sudo rm /etc/apt/preferences.d/rocm-pin-600
sudo rm /etc/apt/keyrings/rocm.gpg
sudo apt-get update
```

### 3. Add ROCm 7.x apt repo

```bash
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
    | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
    https://repo.radeon.com/rocm/apt/7.0 noble main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

echo "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600

sudo apt-get update
```

### 4. Rebuild the Docker image

After the host ROCm repo is updated, the container's `apt-get install rocm-hip-runtime migraphx` will pull ROCm 7.x packages automatically:

```bash
cd ~/AgrobotV2
docker build \
  --file deployment/docker/Dockerfile.rocm \
  --tag agrobot-tom-v2/rocm:latest \
  --no-cache \
  .
```

### 5. Verify inside container

```bash
./deployment/docker/run_rocm.sh bash
rocm-smi                          # should show Radeon 8060S with correct GFX version
migraphx-driver --version         # should show 2.14+ or higher
migraphx-driver perf \
  --onnx models/dino_vitb14_patches.onnx \
  --gpu                           # should compile and benchmark without fault
```

---

## Expected Sprint 3 results after upgrade

| Step | Command | Expected result |
|------|---------|----------------|
| MIGraphX compile | `migraphx-driver compile --onnx dino_vitb14_patches.onnx --gpu` | Compiles without fault, saves `.mxr` binary |
| MIGraphX benchmark | `migraphx-driver perf --onnx dino_vitb14_patches.onnx --gpu` | Summary: mean < 33 ms |
| End-to-end eval | `run_eval.py` with MIGraphX backend | Detection + mAP at >30 FPS |

Target: **<33 ms mean latency** (>30 FPS) on Radeon 8060S with MIGraphX.

---

## PyTorch ROCm 7.x wheels

Once the host is on ROCm 7.x, replace the PyTorch index URL in `Dockerfile.rocm`:

```dockerfile
# Change from:
--index-url https://download.pytorch.org/whl/rocm6.4

# To:
--index-url https://download.pytorch.org/whl/rocm7.0
```

This enables native GPU inference via PyTorch on gfx1151, removing the need
for `AGROBOT_FORCE_CPU=1` and `HIP_VISIBLE_DEVICES=""` workarounds.

---

## Reference

- [AMD Strix Halo ROCm optimization guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [ROCm 7.x release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [MIGraphX gfx1151 tracking issue](https://github.com/ROCm/ROCm/issues/5824)
