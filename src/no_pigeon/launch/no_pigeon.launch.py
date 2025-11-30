#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='no_pigeon',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen',
            parameters=[
                {'fps': 30},
                {'width': 1920},
                {'height': 1080}
            ]
        ),
        Node(
            package='no_pigeon',
            executable='pigeon_detector',
            name='pigeon_detector',
            output='screen',
            parameters=[
                {'model_path': 'yolov8n.pt'},
                {'confidence_threshold': 0.5},
                {'use_imagenet_classification': False}
            ]
        ),
        Node(
            package='no_pigeon',
            executable='sound_player',
            name='sound_player',
            output='screen',
            parameters=[
                {'sound_folder': '/home/david/Documents/no_pigeon/sounds'},
                {'detection_duration_seconds': 2.0},
                {'snippet_duration_seconds': 25.0}
            ]
        ),
    ])
