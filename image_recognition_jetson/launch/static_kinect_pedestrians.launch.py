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
                'prototxt_path': 'ped-100/deploy.prototxt',
                'model_path': 'ped-100/snapshot_iter_70800.caffemodel',
            }.items(),
        ),
    ])
