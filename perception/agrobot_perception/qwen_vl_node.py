"""
qwen_vl_node.py — VLM-Guided Tomato Pick Selection (NODE 3)

Architecture
------------
NODE 3 in the Agrobot TOM v2 picking pipeline. Consumes persistent tomato tracks
from /agrobot/tomato_tracks (NODE 2b output), shows each tomato's JPEG crop to
Qwen2.5-VL-3B-Instruct, and selects which tomato the arm should pick based on
ripeness, size, and accessibility reasoning.

Why Qwen2.5-VL-3B over heuristics?
  The naive heuristic (pick closest / highest score) cannot distinguish a ripe
  red tomato from an unripe green one at the same depth. Qwen-VL gives a
  language-conditioned pick policy — the operator says "pick the reddest" and
  the system adapts without retraining. All inference runs locally on the NucBox
  (96GB unified RAM, ~2GB for 3B bfloat16) — no cloud dependency.

Why not GPT-4V?
  API latency (1-5s + network) + no offline capability. Greenhouse rows often
  lack reliable internet. Local Qwen-VL runs in ~10-30s on CPU, acceptable
  since the detector itself takes ~17s per frame.

Inference model:
  Qwen/Qwen2.5-VL-3B-Instruct loaded in bfloat16 on CPU.
  3B × 2 bytes ≈ 6GB RAM — well within the 96GB NucBox budget.
  First run downloads from HuggingFace (~6GB). Subsequent runs load from
  models/qwen_vl/ if pre-saved there.

Data Flow
---------
  /agrobot/tomato_tracks (String JSON, from NODE 2b)
      │  list of {persistent_id, centroid, sphere, confidence, clipped_image, smoothed}
      ▼
  QwenVLNode._tracks_callback()
      │  skip if ≤0 smoothed tracks or inference already running
      │  decode clipped_image JPEGs → PIL Images
      │  build multi-image prompt → model.generate()
      │  parse persistent_id from response
      ▼
  /agrobot/pick_target     geometry_msgs/PoseStamped  Selected centroid, camera frame
  /agrobot/vlm_reasoning   std_msgs/String            Full VLM text response (logging)
  /agrobot/vlm_selection   std_msgs/String            JSON: selected tomato details

Topics
------
  Subscribed:
    /agrobot/tomato_tracks   std_msgs/String   Tracked + smoothed tomato JSON array

  Published:
    /agrobot/pick_target     geometry_msgs/PoseStamped
                             frame_id=camera_color_optical_frame, identity orientation.
                             Dani's arm planner feeds this directly to MoveIt2.
    /agrobot/vlm_reasoning   std_msgs/String   Raw VLM response for HIL logging.
    /agrobot/vlm_selection   std_msgs/String   JSON with full selected tomato record.

Parameters
----------
  model_path        string  Local path to saved model dir, or HuggingFace model ID.
                            (default: "Qwen/Qwen2.5-VL-3B-Instruct")
  pick_policy       string  Prompt style: "ripe_first" | "closest_first" | "largest_first"
                            (default: "ripe_first")
  min_smoothed_age  int     Only consider tracks with age >= this value (default: 3)
  max_new_tokens    int     Max tokens for VLM response (default: 64)
  tracks_topic      string  Input topic (default: /agrobot/tomato_tracks)

Target environment: [NUCBOX] — requires ~6GB RAM for model + PyTorch CPU inference.
Sprint: 4 — VLM-guided pick policy (GRAND_PLAN §4.1).
"""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# Qwen-VL imports — node starts without them but logs a warning and falls back
# to heuristic (pick closest smoothed tomato).
try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    _VLM_OK = True
except ImportError:
    _VLM_OK = False


# ─── Prompt Templates ─────────────────────────────────────────────────────────

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

_SINGLE_PROMPT = (
    "You are guiding an agricultural robot arm.\n"
    "Here is the only visible tomato (distance={z:.2f}m, radius={r:.1f}cm):\n"
    "Should this tomato be picked? Reply YES or NO with one sentence of reasoning."
)

_MULTI_PROMPT_HEADER = (
    "{policy_text}"
    "Here are the {n} visible tomato candidates:\n"
)

_MULTI_PROMPT_FOOTER = (
    "\nReply with ONLY the tomato number to pick (e.g. \"0\" or \"1\").\n"
    "Do not explain. Just the number."
)


class QwenVLNode(Node):
    """ROS 2 node that uses Qwen2.5-VL to select which tomato the arm should pick."""

    def __init__(self) -> None:
        super().__init__("qwen_vl")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("model_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.declare_parameter("pick_policy", "ripe_first")
        self.declare_parameter("min_smoothed_age", 3)
        self.declare_parameter("max_new_tokens", 64)
        self.declare_parameter("tracks_topic", "/agrobot/tomato_tracks")

        self._model_path: str = self.get_parameter("model_path").value
        self._policy: str = self.get_parameter("pick_policy").value
        self._min_age: int = self.get_parameter("min_smoothed_age").value
        self._max_tokens: int = self.get_parameter("max_new_tokens").value
        tracks_topic: str = self.get_parameter("tracks_topic").value

        # ── Model state ───────────────────────────────────────────────────────
        self._model = None
        self._processor = None
        self._vlm_available = False
        self._inference_running = False   # prevent callback re-entry during slow inference

        # ── Publishers ────────────────────────────────────────────────────────
        self._pick_pub = self.create_publisher(
            PoseStamped, "/agrobot/pick_target", 10
        )
        self._reasoning_pub = self.create_publisher(
            String, "/agrobot/vlm_reasoning", 10
        )
        self._selection_pub = self.create_publisher(
            String, "/agrobot/vlm_selection", 10
        )

        # ── Subscription ──────────────────────────────────────────────────────
        self.create_subscription(String, tracks_topic, self._tracks_callback, 10)

        # ── Load model in background thread so node starts immediately ────────
        if _VLM_OK and _PIL_OK:
            t = threading.Thread(target=self._load_model, daemon=True)
            t.start()
        else:
            missing = []
            if not _VLM_OK:
                missing.append("transformers / qwen-vl-utils")
            if not _PIL_OK:
                missing.append("Pillow")
            self.get_logger().warn(
                f"Missing: {', '.join(missing)}. "
                "Running in heuristic mode (closest smoothed tomato). "
                "Install with: pip install transformers qwen-vl-utils Pillow"
            )

        self.get_logger().info(
            f"QwenVLNode initialized. "
            f"policy='{self._policy}'  min_age={self._min_age}  "
            f"model='{self._model_path}'"
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Qwen2.5-VL in a background thread — ~30s first run from hub."""
        self.get_logger().info(
            f"Loading Qwen2.5-VL from '{self._model_path}' "
            "(this takes ~30s on first run; downloading ~6GB if not cached)..."
        )
        try:
            # Check for local save first (avoids re-download after first run).
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            local_path = repo_root / "models" / "qwen_vl"
            source = str(local_path) if local_path.exists() else self._model_path

            self._processor = AutoProcessor.from_pretrained(
                source,
                # Limit image resolution — crops are small (~100-200px).
                # Default max_pixels is 12845056 (1280*28*28); we don't need that.
                min_pixels=224 * 224,
                max_pixels=448 * 448,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                # bfloat16 on CPU: 3B × 2 bytes ≈ 6GB RAM. Fits in 96GB NucBox.
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            self._model.eval()
            self._vlm_available = True
            self.get_logger().info(
                "Qwen2.5-VL loaded. VLM-guided pick selection active."
            )
        except Exception as exc:
            self.get_logger().error(
                f"Qwen2.5-VL load failed: {exc}. "
                "Falling back to heuristic (closest smoothed tomato)."
            )

    # ── Main callback ──────────────────────────────────────────────────────────

    def _tracks_callback(self, msg: String) -> None:
        """Called on each /agrobot/tomato_tracks message."""
        if self._inference_running:
            self.get_logger().debug(
                "Inference still running — skipping stale tracks message."
            )
            return

        try:
            tracks: list[dict] = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"JSON parse error: {exc}")
            return

        # Only act on tracks that have converged (EMA has enough observations).
        candidates = [
            t for t in tracks
            if t.get("age", 0) >= self._min_age
        ]

        if not candidates:
            self.get_logger().debug(
                f"No smoothed tracks yet (min_age={self._min_age}). "
                f"Received {len(tracks)} raw track(s)."
            )
            return

        self.get_logger().info(
            f"Received {len(candidates)} smoothed track(s) — running selection."
        )

        if self._vlm_available and self._model is not None:
            self._inference_running = True
            try:
                selected = self._run_vlm(candidates)
            finally:
                self._inference_running = False
        else:
            # Heuristic fallback: pick the closest (minimum z) smoothed tomato.
            selected = min(candidates, key=lambda t: t["centroid"]["z"])
            self.get_logger().info(
                f"Heuristic: selected persistent_id={selected['persistent_id']} "
                f"(z={selected['centroid']['z']:.3f}m, closest)."
            )

        self._publish_selection(selected)

    # ── VLM inference ─────────────────────────────────────────────────────────

    def _decode_jpeg(self, b64: str) -> "PILImage.Image | None":
        """Decode a base64 JPEG string to a PIL Image."""
        try:
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as exc:
            self.get_logger().error(f"JPEG decode failed: {exc}")
            return None

    def _run_vlm(self, candidates: list[dict]) -> dict:
        """Run Qwen2.5-VL inference and return the selected tomato dict."""
        policy_text = _POLICY_PROMPTS.get(
            self._policy, _POLICY_PROMPTS["ripe_first"]
        )

        if len(candidates) == 1:
            t = candidates[0]
            img = self._decode_jpeg(t.get("clipped_image", ""))
            if img is None:
                return t

            prompt_text = _SINGLE_PROMPT.format(
                z=t["centroid"]["z"],
                r=t["sphere"]["radius"] * 100,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            response = self._infer(messages)
            self.get_logger().info(
                f"VLM single-tomato response: '{response}'"
            )
            # For single candidate, publish regardless of YES/NO answer.
            return t

        # Multiple candidates — show all crops in one prompt.
        content: list[dict] = [
            {
                "type": "text",
                "text": _MULTI_PROMPT_HEADER.format(
                    policy_text=policy_text, n=len(candidates)
                ),
            }
        ]
        for t in candidates:
            img = self._decode_jpeg(t.get("clipped_image", ""))
            z = t["centroid"]["z"]
            r = t["sphere"]["radius"] * 100
            pid = t["persistent_id"]
            content.append({
                "type": "text",
                "text": f"\nTomato {pid} (distance={z:.2f}m, radius={r:.1f}cm):",
            })
            if img is not None:
                content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": _MULTI_PROMPT_FOOTER})
        messages = [{"role": "user", "content": content}]

        response = self._infer(messages)
        self.get_logger().info(f"VLM multi-tomato response: '{response}'")

        selected = self._parse_selection(response, candidates)
        self.get_logger().info(
            f"VLM selected persistent_id={selected['persistent_id']} "
            f"(z={selected['centroid']['z']:.3f}m)."
        )
        return selected

    def _infer(self, messages: list[dict]) -> str:
        """Run one forward pass through Qwen2.5-VL and return the response text."""
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,     # greedy decoding — deterministic, faster
            )

        # Strip the input prompt tokens; keep only generated response.
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

    def _parse_selection(
        self, response: str, candidates: list[dict]
    ) -> dict:
        """Extract the selected persistent_id from the VLM response text.

        Looks for the first integer in the response and maps it to a candidate.
        Falls back to closest (min z) if parsing fails or the id is unknown.
        """
        import re
        nums = re.findall(r"\b(\d+)\b", response)
        if nums:
            pid = int(nums[0])
            for t in candidates:
                if t["persistent_id"] == pid:
                    return t
            self.get_logger().warn(
                f"VLM returned persistent_id={pid} which is not in candidates "
                f"{[t['persistent_id'] for t in candidates]}. Using closest."
            )
        else:
            self.get_logger().warn(
                f"Could not parse a number from VLM response: '{response}'. "
                "Using closest."
            )

        return min(candidates, key=lambda t: t["centroid"]["z"])

    # ── Publishing ─────────────────────────────────────────────────────────────

    def _publish_selection(self, tomato: dict) -> None:
        """Publish the selected tomato as a pick target for the arm planner."""
        now = self.get_clock().now().to_msg()
        pid = tomato["persistent_id"]
        c = tomato["centroid"]

        # PoseStamped: centroid in camera_color_optical_frame.
        # Orientation is identity — arm planner determines approach angle from TF.
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "camera_color_optical_frame"
        pose_msg.pose.position.x = float(c["x"])
        pose_msg.pose.position.y = float(c["y"])
        pose_msg.pose.position.z = float(c["z"])
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self._pick_pub.publish(pose_msg)

        # Full selection record for Qwen-VL (Qwen-VL reasoning text).
        selection = {
            "persistent_id": pid,
            "centroid": tomato["centroid"],
            "sphere": tomato["sphere"],
            "confidence": tomato["confidence"],
            "age": tomato.get("age", 0),
        }
        sel_msg = String()
        sel_msg.data = json.dumps(selection)
        self._selection_pub.publish(sel_msg)

        self.get_logger().info(
            f"Published pick_target: persistent_id={pid} "
            f"x={c['x']:+.3f} y={c['y']:+.3f} z={c['z']:.3f} m"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenVLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
