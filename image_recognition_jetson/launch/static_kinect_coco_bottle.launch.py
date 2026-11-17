from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('image_recognition_jetson'), 'launch', 'static_kinect.launch.py',
            ])),
            launch_arguments={
                'prototxt_path': 'DetectNet-COCO-Bottle/deploy.prototxt',
                'model_path': 'DetectNet-COCO-Bottle/snapshot_iter_59700.caffemodel',
            }.items(),
        ),
    ])
