"""
TU Delft Swarm Lab - Human Swarm Interaction Project
Description: Track and filter multiple keypoints linearly.
Author: Alexander James Becoy
Date: 2024-10-28
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Get the launch file directory
    keypoint_tracking_dir = get_package_share_directory('keypoints_tracking')

    # Define the path to the config file
    config_file = os.path.join(keypoint_tracking_dir, 'config', 'multiple_keypoints_tracking_params.yaml')

    return LaunchDescription([
        Node(
            package='keypoints_tracking',
            executable='multiple_keypoints_tracking_node',
            name='multiple_keypoints_tracking_node',
            output='screen',
            parameters=[config_file]
        )
    ])
