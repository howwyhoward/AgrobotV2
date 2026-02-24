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
  [TomatoDetector.detect()]   ← Sprint 2: replace with DINOv2+SAM2
      │
      ▼
  /agrobot/detections (vision_msgs/Detection2DArray)
  /agrobot/debug_image (sensor_msgs/Image)       ← for Foxglove visualization

Topics
------
  Subscribed:
    /camera/image_raw       sensor_msgs/Image   Raw camera frames

  Published:
    /agrobot/detections     vision_msgs/Detection2DArray
    /agrobot/debug_image    sensor_msgs/Image   Annotated frame (debug only)

Parameters
----------
  confidence_threshold  float   Minimum detection confidence (default: 0.5)
  input_width           int     Model input width in pixels (default: 518)
  input_height          int     Model input height in pixels (default: 518)
  publish_debug_image   bool    Whether to publish annotated debug frames (default: True)
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header

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

    This class has the exact interface that Sprint 2's DINOv2+SAM2 detector
    will implement. Swapping the backend in Sprint 2 means:
      1. Replace this class with `DINOv2SAM2Detector` in a new file.
      2. Change the import in __init__ of TomatoDetectorNode.
      3. Nothing else in the node changes.

    Returns:
        List of dicts: [{"box": [x1,y1,x2,y2], "score": float, "label": str}]
    """

    def detect(self, preprocessed_chw: np.ndarray) -> list[dict]:
        # Sprint 2: replace with actual DINOv2 feature extraction + SAM2 prompting.
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

        self._conf_threshold = self.get_parameter("confidence_threshold").value
        self._input_size = (
            self.get_parameter("input_width").value,
            self.get_parameter("input_height").value,
        )
        self._publish_debug = self.get_parameter("publish_debug_image").value

        # ── Core Components ───────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._detector = PlaceholderDetector()

        # ── Subscribers ───────────────────────────────────────────────────────
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            SENSOR_QOS,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._detections_pub = self.create_publisher(
            Detection2DArray,
            "/agrobot/detections",
            10,  # QoS depth
        )

        if self._publish_debug:
            self._debug_pub = self.create_publisher(
                Image,
                "/agrobot/debug_image",
                1,
            )

        self.get_logger().info(
            f"TomatoDetectorNode initialized. "
            f"conf_threshold={self._conf_threshold}, "
            f"input_size={self._input_size}"
        )

    def _image_callback(self, msg: Image) -> None:
        """Called for every incoming camera frame."""
        try:
            # Convert ROS Image message → OpenCV BGR numpy array.
            bgr_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Preprocess: BGR → RGB → letterbox → normalize → CHW float32.
        preprocessed = preprocess_for_dino(bgr_frame, input_size=self._input_size)

        # Run detection (stub for now, DINOv2+SAM2 in Sprint 2).
        raw_detections = self._detector.detect(preprocessed)

        # Filter by confidence threshold.
        detections = [d for d in raw_detections if d["score"] >= self._conf_threshold]

        # Publish structured detections.
        self._publish_detections(detections, msg.header)

        # Publish annotated debug image if enabled.
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
