from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    networks_path = LaunchConfiguration('networks_path')
    prototxt_path = PathJoinSubstitution([networks_path, LaunchConfiguration('prototxt_path')])
    model_path = PathJoinSubstitution([networks_path, LaunchConfiguration('model_path')])
    remappings = [('image', 'image_rect_color')]
    return LaunchDescription([
        DeclareLaunchArgument('networks_path', default_value='/home/ubuntu/jetson-inference/data/networks'),
        DeclareLaunchArgument('prototxt_path', default_value='ped-100/deploy.prototxt'),
        DeclareLaunchArgument('model_path', default_value='ped-100/snapshot_iter_70800.caffemodel'),
        Node(
            package='image_recognition_jetson', executable='detect_net_ros',
            namespace='camera/rgb', name='detect_net', output='screen',
            parameters=[{
                'prototxt_path': ParameterValue(prototxt_path, value_type=str),
                'model_path': ParameterValue(model_path, value_type=str),
            }],
            remappings=remappings,
        ),
        Node(
            package='image_recognition_util', executable='publish_image_with_recognitions_overlay',
            namespace='camera/rgb', name='overlay', remappings=remappings,
        ),
    ])
