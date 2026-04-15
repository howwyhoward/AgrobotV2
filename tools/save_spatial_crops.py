#!/usr/bin/env python3
"""
save_spatial_crops.py — Save tomato JPEG crops from /agrobot/tomato_spatial to disk.

Purpose: Subscribe to NODE 2 output, decode each base64 clipped_image, and write
         timestamped JPEG files to eval_reports/spatial_crops/. Run this alongside
         the spatial node to build a local image archive for inspection on Mac.

Target environment: [NUCBOX] — run inside the ROCm container.
Sprint: 4 — visual inspection and Qwen-VL prompt engineering.

Usage:
  docker exec -it $(docker ps -q) bash
  source /opt/ros/jazzy/setup.bash && source /workspace/install/setup.bash
  export ROS_DOMAIN_ID=42
  python3 tools/save_spatial_crops.py

Then on Mac (pull over Tailscale):
  bash tools/pull_crops.sh
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CropSaverNode(Node):
    """Minimal ROS 2 subscriber that decodes and saves JPEG crops to disk."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__("crop_saver")
        self._output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self._frame = 0

        self.create_subscription(
            String, "/agrobot/tomato_spatial", self._callback, 10
        )
        self.get_logger().info(
            f"CropSaverNode ready. Saving JPEGs to: {output_dir}"
        )

    def _callback(self, msg: String) -> None:
        try:
            tomatoes = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"JSON parse error: {exc}")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._frame += 1
        saved = 0

        for t in tomatoes:
            jpeg_b64 = t.get("clipped_image")
            if not jpeg_b64:
                continue

            tid = t["tomato_id"]
            z = t["centroid"]["z"]
            r = t["sphere"]["radius"] * 100
            score = t["confidence"]

            # Filename encodes all key spatial metadata so you can read it in Finder.
            fname = (
                self._output_dir
                / f"{ts}_f{self._frame:04d}_t{tid}_z{z:.2f}m_r{r:.1f}cm_s{score:.2f}.jpg"
            )

            try:
                fname.write_bytes(base64.b64decode(jpeg_b64))
                saved += 1
                self.get_logger().info(f"  [{saved}] {fname.name}")
            except Exception as exc:
                self.get_logger().error(f"  Failed to write {fname.name}: {exc}")

        self.get_logger().info(
            f"Frame {self._frame} ({ts}): saved {saved}/{len(tomatoes)} crops."
        )


def main() -> None:
    rclpy.init()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "eval_reports" / "spatial_crops"

    node = CropSaverNode(output_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
