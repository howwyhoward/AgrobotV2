#!/usr/bin/env python3
"""
test_qwen_vl.py — Standalone Qwen2.5-VL smoke-test on saved spatial crops.

Purpose: Test VLM pick selection on real JPEG crops from eval_reports/spatial_crops/
         WITHOUT needing ROS, Docker, or the NucBox. Runs on Mac (MPS) or CPU.

Usage [MAC]:
  PYTHONPATH=perception python3 tools/test_qwen_vl.py
  PYTHONPATH=perception python3 tools/test_qwen_vl.py --crops eval_reports/spatial_crops
  PYTHONPATH=perception python3 tools/test_qwen_vl.py --policy closest_first
  PYTHONPATH=perception python3 tools/test_qwen_vl.py --dry-run  # show prompt, skip inference

Filename format expected (from tools/save_spatial_crops.py):
  {timestamp}_f{frame:04d}_t{tomato_id}_z{z}m_r{r}cm_s{score}.jpg
  e.g. 20260415_043538_f0001_t0_z0.54m_r4.7cm_s0.56.jpg

Target environment: [MAC] — uses MPS if available, CPU fallback.
Sprint: 4 — Qwen-VL prompt engineering and offline validation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

_POLICY_PROMPTS = {
    "ripe_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: red/ripe over green/unripe, then larger, then closer to camera.\n"
    ),
    "closest_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: the tomato closest to the camera (smallest z distance).\n"
    ),
    "largest_first": (
        "You are guiding an agricultural robot arm to pick tomatoes.\n"
        "Prefer: the largest tomato by apparent size, then ripeness.\n"
    ),
}

_SINGLE_TEMPLATE = (
    "You are guiding an agricultural robot arm.\n"
    "Here is the only visible tomato (distance={z:.2f}m, radius={r:.1f}cm):\n"
    "Should this tomato be picked? Reply YES or NO with one sentence of reasoning."
)

_MULTI_FOOTER = (
    "\nReply with ONLY the tomato number to pick (e.g. \"0\" or \"1\").\n"
    "Do not explain. Just the number."
)


# ─── Filename parsing ──────────────────────────────────────────────────────────

_FNAME_RE = re.compile(
    r"(?P<ts>\d{8}_\d{6})_f(?P<frame>\d+)_t(?P<tid>\d+)"
    r"_z(?P<z>[\d.]+)m_r(?P<r>[\d.]+)cm_s(?P<score>[\d.]+)\.jpg"
)


def parse_crop_filename(path: Path) -> dict | None:
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return {
        "path": path,
        "timestamp": m.group("ts"),
        "frame": int(m.group("frame")),
        "tomato_id": int(m.group("tid")),
        "z": float(m.group("z")),
        "r": float(m.group("r")),
        "score": float(m.group("score")),
    }


def load_frames(crops_dir: Path) -> dict[str, list[dict]]:
    """Group crop metadata by frame key (timestamp + frame number)."""
    frames: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(crops_dir.glob("*.jpg")):
        meta = parse_crop_filename(p)
        if meta:
            key = f"{meta['timestamp']}_f{meta['frame']:04d}"
            frames[key].append(meta)
        else:
            print(f"  [SKIP] Unrecognised filename: {p.name}")
    return dict(frames)


# ─── Inference ────────────────────────────────────────────────────────────────

def build_messages(crops: list[dict], policy: str) -> tuple[list[dict], str]:
    """Build Qwen-VL message list and a human-readable prompt summary."""
    from PIL import Image as PILImage

    policy_text = _POLICY_PROMPTS.get(policy, _POLICY_PROMPTS["ripe_first"])

    if len(crops) == 1:
        c = crops[0]
        img = PILImage.open(c["path"]).convert("RGB")
        prompt_text = _SINGLE_TEMPLATE.format(z=c["z"], r=c["r"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        summary = prompt_text
        return messages, summary

    content: list[dict] = [
        {
            "type": "text",
            "text": f"{policy_text}Here are the {len(crops)} visible tomato candidates:\n",
        }
    ]
    summary_lines = [f"Policy: {policy}", f"Candidates: {len(crops)}"]

    for c in sorted(crops, key=lambda x: x["tomato_id"]):
        img = PILImage.open(c["path"]).convert("RGB")
        label = (
            f"\nTomato {c['tomato_id']} "
            f"(distance={c['z']:.2f}m, radius={c['r']:.1f}cm, score={c['score']:.2f}):"
        )
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": img})
        summary_lines.append(
            f"  Tomato {c['tomato_id']}: z={c['z']:.2f}m r={c['r']:.1f}cm "
            f"score={c['score']:.2f} img={c['path'].name}"
        )

    content.append({"type": "text", "text": _MULTI_FOOTER})
    messages = [{"role": "user", "content": content}]
    return messages, "\n".join(summary_lines)


def run_inference(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int,
    device: str,
) -> str:
    """Single Qwen-VL forward pass, returns response text."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()


def parse_pick(response: str, crops: list[dict]) -> dict | None:
    """Extract tomato_id from VLM response."""
    nums = re.findall(r"\b(\d+)\b", response)
    if nums:
        tid = int(nums[0])
        for c in crops:
            if c["tomato_id"] == tid:
                return c
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test Qwen2.5-VL on saved spatial crop JPEGs."
    )
    parser.add_argument(
        "--crops",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "eval_reports" / "spatial_crops",
        help="Directory of JPEG crops from save_spatial_crops.py.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID or local path (default: Qwen/Qwen2.5-VL-3B-Instruct).",
    )
    parser.add_argument(
        "--policy",
        default="ripe_first",
        choices=list(_POLICY_PROMPTS.keys()),
        help="Pick policy prompt to use.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max new tokens for VLM generation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt and crop list without running inference.",
    )
    args = parser.parse_args()

    if not args.crops.exists():
        print(f"ERROR: crops directory not found: {args.crops}")
        print("Run tools/save_spatial_crops.py on NucBox, then tools/pull_crops.sh on Mac.")
        sys.exit(1)

    frames = load_frames(args.crops)
    if not frames:
        print(f"No valid crop files found in {args.crops}")
        sys.exit(1)

    print(f"\nFound {len(frames)} frame(s) in {args.crops}")
    for key, crops in frames.items():
        print(f"  {key}: {len(crops)} tomato(s) — " +
              ", ".join(f"t{c['tomato_id']}(z={c['z']:.2f}m)" for c in crops))

    if args.dry_run:
        print("\n── Dry run: prompt preview ──────────────────────────────────────")
        for key, crops in frames.items():
            _, summary = build_messages(crops, args.policy)
            print(f"\nFrame {key}:\n{summary}")
        print("\n(Pass without --dry-run to run inference.)")
        return

    # Load model
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16   # MPS prefers float16 over bfloat16
    else:
        device = "cpu"
        dtype = torch.bfloat16

    repo_root = Path(__file__).resolve().parent.parent
    local_path = repo_root / "models" / "qwen_vl"
    source = str(local_path) if local_path.exists() else args.model

    print(f"\nLoading Qwen2.5-VL from '{source}' on {device} ({dtype}) ...")
    processor = AutoProcessor.from_pretrained(
        source, min_pixels=224 * 224, max_pixels=448 * 448
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, torch_dtype=dtype
    ).to(device).eval()
    print("Model loaded.\n")

    # Run inference per frame
    for key, crops in frames.items():
        print(f"── Frame {key} ({'×'.join(str(len(crops)) + ' tomato' + ('' if len(crops)==1 else 's'))})")
        messages, summary = build_messages(crops, args.policy)
        print(summary)
        print("\nRunning inference ...")
        response = run_inference(model, processor, messages, args.max_tokens, device)
        print(f"VLM response: '{response}'")

        selected = parse_pick(response, crops)
        if selected:
            print(
                f"→ PICK: Tomato {selected['tomato_id']} "
                f"at z={selected['z']:.3f}m, r={selected['r']:.1f}cm"
            )
        else:
            fallback = min(crops, key=lambda c: c["z"])
            print(
                f"→ Parse failed. Heuristic fallback: "
                f"Tomato {fallback['tomato_id']} (closest, z={fallback['z']:.3f}m)"
            )
        print()


if __name__ == "__main__":
    main()
