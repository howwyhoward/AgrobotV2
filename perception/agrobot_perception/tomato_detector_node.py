"""
tomato_detector_node.py — Tomato Detection ROS 2 Node

Architecture
------------
This node is the central perception pipeline entry point for Agrobot TOM v2.
It follows the "thin node, thick library" pattern:
  - The ROS 2 node itself (this file) is kept minimal: subscribe, preprocess,
    call detector, publish. No business logic here.
  - All image processing lives in `utils/image_utils.py`.
  - The detector is a pluggable backend (Placeholder → DINOv2+SAM2 in Sprint 2).

Data Flow
---------
  /camera/image_raw (sensor_msgs/Image)
      │
      ▼
  [cv_bridge] → BGR np.ndarray
      │
      ▼
  [preprocess_for_dino] → float32 CHW tensor
      │
      ▼
  [DINOv2SAM2Detector.detect()]   ← Sprint 3: swap for MIGraphX ONNX on ROCm
      │
      ▼
  /agrobot/detections (vision_msgs/Detection2DArray)
  /agrobot/debug_image (sensor_msgs/Image)       ← for Foxglove visualization

Topics
------
  Subscribed:
    /camera/image_raw                         sensor_msgs/Image   Raw camera frames
    <depth_topic>                             sensor_msgs/Image   Aligned depth (optional)
    <depth_camera_info_topic>                 sensor_msgs/CameraInfo (optional)

  Published:
    /agrobot/detections                       vision_msgs/Detection2DArray
    /agrobot/detections_3d                    vision_msgs/Detection3DArray (when depth enabled)
    /agrobot/safe_to_pick                     std_msgs/Bool  False = no pick this cycle
    /agrobot/debug_image                      sensor_msgs/Image (debug only)

Parameters
----------
  confidence_threshold    float   Minimum detection confidence (default: 0.5)
  input_width             int     Model input width in pixels (default: 518)
  input_height            int     Model input height in pixels (default: 518)
  publish_debug_image     bool    Whether to publish annotated debug frames (default: True)
  depth_topic             string  Optional. Aligned depth for 3D (default: "" = disabled)
  depth_camera_info_topic string  Optional. CameraInfo for depth (default: "")
  watchdog_timeout_ms     int     Publish empty detections if no frame within this window.
                                  0 = disabled. (default: 2000)
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Header
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    Detection3DArray,
    Detection3D,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Point

import cv2
import numpy as np

try:
    from cv_bridge import CvBridge
except ImportError:
    raise ImportError(
        "cv_bridge not found. Are you running inside the ROS 2 Docker container? "
        "Run: docker compose -f deployment/compose/docker-compose.yml run --rm dev bash"
    )

from agrobot_perception.utils.image_utils import (
    preprocess_for_dino,
    draw_detection_overlay,
)
from agrobot_perception.detectors.dino_sam2_detector import (
    DINOv2SAM2Detector,
    _select_device,
)


# ─── QoS Profiles ─────────────────────────────────────────────────────────────
# Camera topics from real sensors use "Best Effort" reliability — they drop
# frames rather than queue them. Using Reliable QoS here would cause a
# QoS compatibility mismatch warning and the subscription would receive nothing.
SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class PlaceholderDetector:
    """Stub detector — returns zero detections.

    Retained as a documented fallback. To revert to the stub:
      self._detector = PlaceholderDetector()

    Returns:
        List of dicts: [{"box": [x1,y1,x2,y2], "score": float, "label": str}]
    """

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        return []


class TomatoDetectorNode(Node):
    """ROS 2 node that detects tomatoes in camera frames."""

    def __init__(self) -> None:
        super().__init__("tomato_detector")

        # ── Parameters ────────────────────────────────────────────────────────
        # Declare all parameters with defaults. Users can override via:
        #   ros2 run agrobot_perception tomato_detector --ros-args -p confidence_threshold:=0.7
        # Or in a launch file (see launch/perception.launch.py).
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("input_width", 518)
        self.declare_parameter("input_height", 518)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("depth_camera_info_topic", "")
        # Watchdog: if no camera frame arrives within this window, publish
        # empty detections and safe_to_pick=False (FM-1 in FAILURE_MODES.md).
        # Set to 0 to disable the watchdog entirely.
        self.declare_parameter("watchdog_timeout_ms", 2000)

        self._conf_threshold = self.get_parameter("confidence_threshold").value
        self._input_size = (
            self.get_parameter("input_width").value,
            self.get_parameter("input_height").value,
        )
        self._publish_debug = self.get_parameter("publish_debug_image").value
        self._depth_topic = self.get_parameter("depth_topic").value
        self._depth_info_topic = self.get_parameter("depth_camera_info_topic").value
        self._depth_image: np.ndarray | None = None
        self._depth_K: tuple[float, float, float, float] | None = None
        self._watchdog_timeout_ms: int = self.get_parameter("watchdog_timeout_ms").value
        self._last_frame_time: float = 0.0

        # ── Core Components ───────────────────────────────────────────────────
        self._bridge = CvBridge()
        # Sprint 2: DINOv2+SAM2 zero-shot detector.
        # Sprint 3: swap for MIGraphX-optimized ONNX export (ROCm edge).
        self._detector = DINOv2SAM2Detector(
            device=_select_device(),
            confidence_threshold=self._conf_threshold,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            SENSOR_QOS,
        )

        if self._depth_topic and self._depth_info_topic:
            self._depth_sub = self.create_subscription(
                Image, self._depth_topic, self._depth_callback, SENSOR_QOS
            )
            self._depth_info_sub = self.create_subscription(
                CameraInfo,
                self._depth_info_topic,
                self._depth_info_callback,
                10,
            )
            self._detections_3d_pub = self.create_publisher(
                Detection3DArray, "/agrobot/detections_3d", 10
            )

        # ── Publishers ────────────────────────────────────────────────────────
        self._detections_pub = self.create_publisher(
            Detection2DArray,
            "/agrobot/detections",
            10,
        )
        # safe_to_pick: False when no detections or watchdog triggered.
        # The arm planner subscribes here to gate pick attempts.
        self._safe_to_pick_pub = self.create_publisher(
            Bool,
            "/agrobot/safe_to_pick",
            10,
        )

        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                Image,
                "/agrobot/debug_image",
                1,
            )

        # ── Watchdog timer ────────────────────────────────────────────────────
        if self._watchdog_timeout_ms > 0:
            self._watchdog_timer = self.create_timer(
                self._watchdog_timeout_ms / 1000.0,
                self._watchdog_callback,
            )
            self.get_logger().info(
                f"Watchdog enabled: safe_to_pick=False if no frame "
                f"in {self._watchdog_timeout_ms} ms."
            )

        self.get_logger().info(
            f"TomatoDetectorNode initialized. "
            f"conf_threshold={self._conf_threshold}, "
            f"input_size={self._input_size}"
        )

    def _watchdog_callback(self) -> None:
        """Fires if no camera frame has arrived within watchdog_timeout_ms.

        Publishes empty detections + safe_to_pick=False so the planner gets
        an explicit signal rather than silence. See FM-1 in docs/FAILURE_MODES.md.
        """
        if self._last_frame_time == 0.0:
            # Node just started, no frame ever received yet — don't alarm.
            return

        elapsed_ms = (time.monotonic() - self._last_frame_time) * 1000.0
        if elapsed_ms >= self._watchdog_timeout_ms:
            self.get_logger().warn(
                f"No camera frame for {elapsed_ms:.0f} ms "
                f"(timeout={self._watchdog_timeout_ms} ms). "
                "Publishing empty detections — safe_to_pick=False. "
                "Check /camera/image_raw and the RealSense driver.",
                throttle_duration_sec=1.0,
            )
            self._publish_detections([], Header())
            self._publish_safe_to_pick(False)

    def _publish_safe_to_pick(self, safe: bool) -> None:
        msg = Bool()
        msg.data = safe
        self._safe_to_pick_pub.publish(msg)

    def _image_callback(self, msg: Image) -> None:
        """Called for every incoming camera frame."""
        self._last_frame_time = time.monotonic()

        try:
            bgr_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        preprocessed = preprocess_for_dino(bgr_frame, input_size=self._input_size)
        raw_detections = self._detector.detect(preprocessed)
        detections = [d for d in raw_detections if d["score"] >= self._conf_threshold]

        self._publish_detections(detections, msg.header)
        # Explicit safe_to_pick signal every frame — planner doesn't need to
        # infer from detection count; it reads this directly.
        self._publish_safe_to_pick(len(detections) > 0)

        if self._depth_image is not None and self._depth_K is not None and detections:
            self._publish_detections_3d(
                detections, msg.header, bgr_frame.shape, self._input_size
            )

        if self._publish_debug:
            self._publish_debug_image(bgr_frame, detections, msg.header)

    def _publish_detections(self, detections: list[dict], header: Header) -> None:
        """Build and publish a vision_msgs/Detection2DArray message."""
        array_msg = Detection2DArray()
        array_msg.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Bounding box center + size (vision_msgs convention).
            x1, y1, x2, y2 = det["box"]
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)

            # Hypothesis: class label + confidence.
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = det["label"]
            hypothesis.hypothesis.score = float(det["score"])
            detection.results.append(hypothesis)

            array_msg.detections.append(detection)

        self._detections_pub.publish(array_msg)

        if detections:
            self.get_logger().debug(f"Published {len(detections)} detection(s).")

    def _publish_debug_image(
        self,
        bgr_frame: np.ndarray,
        detections: list[dict],
        header: Header,
    ) -> None:
        """Annotate the frame with bounding boxes and publish for visualization."""
        annotated = bgr_frame.copy()
        if detections:
            draw_detection_overlay(
                annotated,
                boxes=[d["box"] for d in detections],
                labels=[d["label"] for d in detections],
                scores=[d["score"] for d in detections],
            )
        debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg.header = header
        self._debug_pub.publish(debug_msg)

    def _depth_callback(self, msg: Image) -> None:
        try:
            self._depth_image = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception:
            self._depth_image = None

    def _depth_info_callback(self, msg: CameraInfo) -> None:
        K = msg.k
        if len(K) >= 6:
            self._depth_K = (float(K[0]), float(K[4]), float(K[2]), float(K[5]))

    def _publish_detections_3d(
        self,
        detections: list[dict],
        header: Header,
        color_shape: tuple,
        input_size: tuple[int, int],
    ) -> None:
        if self._depth_image is None or self._depth_K is None:
            return
        fx, fy, cx, cy = self._depth_K
        h, w = color_shape[:2]
        iw, ih = input_size
        scale = min(iw / w, ih / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_x = (iw - new_w) // 2
        pad_y = (ih - new_h) // 2

        array_3d = Detection3DArray()
        array_3d.header = header

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx_518 = (x1 + x2) / 2.0
            cy_518 = (y1 + y2) / 2.0
            u_orig = (cx_518 - pad_x) / scale
            v_orig = (cy_518 - pad_y) / scale
            u_int = int(round(u_orig))
            v_int = int(round(v_orig))
            if u_int < 0 or u_int >= self._depth_image.shape[1] or v_int < 0 or v_int >= self._depth_image.shape[0]:
                continue
            z = float(self._depth_image[v_int, u_int])
            if z <= 0 or not np.isfinite(z):
                continue
            x_cam = (u_orig - cx) * z / fx
            y_cam = (v_orig - cy) * z / fy

            d3 = Detection3D()
            d3.header = header
            d3.bbox.center.position.x = x_cam
            d3.bbox.center.position.y = y_cam
            d3.bbox.center.position.z = z
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = det["label"]
            hyp.hypothesis.score = float(det["score"])
            d3.results.append(hyp)
            array_3d.detections.append(d3)

        if array_3d.detections:
            self._detections_3d_pub.publish(array_3d)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TomatoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
