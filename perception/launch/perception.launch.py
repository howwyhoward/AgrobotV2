"""
perception.launch.py — Launch file for the Agrobot perception stack.

ROS 2 launch files are Python scripts. They describe which nodes to start,
with what parameters, remappings, and in what namespace.

Run inside the container:
    ros2 launch agrobot_perception perception.launch.py
    ros2 launch agrobot_perception perception.launch.py confidence_threshold:=0.3

Production config (S4.12 — matches REPRODUCE.md best result):
    AGROBOT_FORCE_CPU=1 HIP_VISIBLE_DEVICES="" \\
    ros2 launch agrobot_perception perception.launch.py \\
      depth_topic:=/camera/camera/depth/image_rect_raw \\
      depth_camera_info_topic:=/camera/camera/depth/camera_info

Why a launch file vs `ros2 run`?
- Starts multiple nodes in one command (detector + future spatial node + planner).
- Centralized parameter configuration (no long --ros-args chains).
- Handles node lifecycle, respawn policies, and namespace isolation cleanly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Camera topics ──────────────────────────────────────────────────────────
    camera_topic_arg = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/camera/color/image_raw",
        description="Color image topic from RealSense driver (sensor_msgs/Image).",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="",
        description=(
            "Aligned depth image topic (sensor_msgs/Image). "
            "Empty = 3D output disabled. "
            "Typical value: /camera/camera/depth/image_rect_raw"
        ),
    )
    depth_camera_info_arg = DeclareLaunchArgument(
        "depth_camera_info_topic",
        default_value="",
        description=(
            "CameraInfo for depth (sensor_msgs/CameraInfo). "
            "Typical value: /camera/camera/depth/camera_info"
        ),
    )

    # ── Detection thresholds ───────────────────────────────────────────────────
    conf_threshold_arg = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.35",
        description="Minimum detection score [0.0, 1.0]. S4.12 best: 0.35.",
    )
    nms_iou_arg = DeclareLaunchArgument(
        "nms_iou_threshold",
        default_value="0.5",
        description="IoU threshold for box NMS. S4.12 best: 0.5.",
    )

    # ── SAM2 AMG parameters ────────────────────────────────────────────────────
    amg_points_arg = DeclareLaunchArgument(
        "amg_points_per_side",
        default_value="28",
        description=(
            "SAM2 AMG grid density. 28 → 784 proposals. "
            "S4.12 best: 28. Higher = more recall, slower (32 = 22s/frame CPU)."
        ),
    )
    max_detections_arg = DeclareLaunchArgument(
        "max_detections",
        default_value="30",
        description="Max tomatoes reported per frame. S4.12 best: 30.",
    )

    # ── DINOv2 scoring parameters ──────────────────────────────────────────────
    dino_weight_arg = DeclareLaunchArgument(
        "dino_score_weight",
        default_value="0.7",
        description=(
            "α in: score = α·dino_sim + (1-α)·pred_iou. "
            "S4.12 best: 0.7. 1.0 = pure DINOv2, 0.0 = pure SAM2 quality."
        ),
    )
    neg_weight_arg = DeclareLaunchArgument(
        "negative_weight",
        default_value="1.0",
        description="λ for contrastive negative suppression. S4.12 best: 1.0.",
    )

    # ── Embedding paths ────────────────────────────────────────────────────────
    query_emb_arg = DeclareLaunchArgument(
        "query_embedding_path",
        default_value="models/query_embedding_k4.pt",
        description="Path to k=4 prototype query embedding (built by build_query_embedding.py).",
    )
    neg_emb_arg = DeclareLaunchArgument(
        "negative_embedding_path",
        default_value="models/negative_embedding.pt",
        description="Path to background-mean negative embedding.",
    )
    sam2_ckpt_arg = DeclareLaunchArgument(
        "sam2_checkpoint",
        default_value="",
        description=(
            "Path to SAM2 checkpoint. Empty = auto-detect models/sam2/. "
            "Use models/sam2/sam2_tomato_finetuned.pt for point-prompt fine-tuned version."
        ),
    )

    # ── Debug ──────────────────────────────────────────────────────────────────
    publish_debug_arg = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="true",
        description="Publish annotated debug image on /agrobot/debug_image (Foxglove).",
    )

    # ── Tomato Detector Node (Node 1) ──────────────────────────────────────────
    tomato_detector_node = Node(
        package="agrobot_perception",
        executable="tomato_detector",
        name="tomato_detector",
        namespace="agrobot",
        output="screen",
        # Force CPU and disable ROCm at the OS level for this node's process.
        # SAM2's C++ extensions probe the AMD GPU driver at import time regardless
        # of the Python device flag, triggering a gfx1151 kernel crash. Setting
        # these here guarantees they are set for the spawned process, not just
        # inherited from the shell (which ros2 launch does not reliably propagate).
        additional_env={
            "AGROBOT_FORCE_CPU": "1",
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
        },
        remappings=[
            ("/camera/image_raw", LaunchConfiguration("camera_topic")),
        ],
        parameters=[
            {
                "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
                "input_width": 518,
                "input_height": 518,
                "depth_topic": LaunchConfiguration("depth_topic"),
                "depth_camera_info_topic": LaunchConfiguration("depth_camera_info_topic"),
                "amg_points_per_side": LaunchConfiguration("amg_points_per_side"),
                "max_detections": LaunchConfiguration("max_detections"),
                "nms_iou_threshold": LaunchConfiguration("nms_iou_threshold"),
                "dino_score_weight": LaunchConfiguration("dino_score_weight"),
                "negative_weight": LaunchConfiguration("negative_weight"),
                "query_embedding_path": LaunchConfiguration("query_embedding_path"),
                "negative_embedding_path": LaunchConfiguration("negative_embedding_path"),
                "sam2_checkpoint": LaunchConfiguration("sam2_checkpoint"),
            }
        ],
    )

    return LaunchDescription(
        [
            camera_topic_arg,
            depth_topic_arg,
            depth_camera_info_arg,
            conf_threshold_arg,
            nms_iou_arg,
            amg_points_arg,
            max_detections_arg,
            dino_weight_arg,
            neg_weight_arg,
            query_emb_arg,
            neg_emb_arg,
            sam2_ckpt_arg,
            publish_debug_arg,
            tomato_detector_node,
        ]
    )
