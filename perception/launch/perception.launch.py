"""
perception.launch.py — Launch file for the Agrobot perception stack.

ROS 2 launch files are Python scripts. They describe which nodes to start,
with what parameters, remappings, and in what namespace.

Run inside the container:
    ros2 launch agrobot_perception perception.launch.py
    ros2 launch agrobot_perception perception.launch.py confidence_threshold:=0.7

Why a launch file vs `ros2 run`?
- Starts multiple nodes in one command (detector + future depth + future planner).
- Centralized parameter configuration (no long --ros-args chains).
- Handles node lifecycle, respawn policies, and namespace isolation cleanly.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # ── Launch Arguments (overridable from CLI) ────────────────────────────────
    # DeclareLaunchArgument makes a parameter overridable at launch time:
    #   ros2 launch agrobot_perception perception.launch.py confidence_threshold:=0.8
    conf_threshold_arg = DeclareLaunchArgument(
        "confidence_threshold",
        default_value="0.5",
        description="Minimum detection confidence score [0.0, 1.0]",
    )

    publish_debug_arg = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="true",
        description="Publish annotated debug image on /agrobot/debug_image",
    )

    camera_topic_arg = DeclareLaunchArgument(
        "camera_topic",
        default_value="/camera/camera/color/image_raw",
        description="Input camera topic (sensor_msgs/Image). Default matches the "
                    "realsense2_camera ROS 2 driver topic for the Intel D456.",
    )

    # ── Tomato Detector Node ───────────────────────────────────────────────────
    tomato_detector_node = Node(
        package="agrobot_perception",
        executable="tomato_detector",
        name="tomato_detector",
        namespace="agrobot",
        output="screen",
        # Remappings: decouple the node's hardcoded topic names from the
        # actual topic names in the system. The node always uses /camera/image_raw
        # internally, but we can remap it to whatever the real camera publishes.
        remappings=[
            ("/camera/image_raw", LaunchConfiguration("camera_topic")),
        ],
        parameters=[
            {
                "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
                "input_width": 518,
                "input_height": 518,
            }
        ],
    )

    return LaunchDescription(
        [
            conf_threshold_arg,
            publish_debug_arg,
            camera_topic_arg,
            tomato_detector_node,
        ]
    )
