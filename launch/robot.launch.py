from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='test_onnx',
            executable='test_onnx_node',
            name='test_onnx',
            output='screen',
            # parameters=[
            #     {'use_sim_time': False}
            # ]
        )
    ])