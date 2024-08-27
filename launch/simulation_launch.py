from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.substitutions.path_join_substitution import PathJoinSubstitution
import pathlib
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


PACKAGE_NAME = 'surround-view-segbev'
USE_SIM_TIME = True

package_dir = get_package_share_directory(PACKAGE_NAME)


def get_ros2_nodes():
    robot_urdf = os.path.join(package_dir, os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'resources/descriptions', 'EgoVehicle.urdf'))))

    with open(robot_urdf, 'r') as urdf:
        robot_description = urdf.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': USE_SIM_TIME, 'robot_description': robot_description}],
        arguments=[robot_urdf],
    )

    static_transforms = [
        ['map', 'odom'],
        ['odom', 'base_link'],
    ]
    
    static_transform_nodes = []

    for transform in static_transforms:
        static_transform_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0'] + transform,
            parameters=[{'use_sim_time': USE_SIM_TIME}]
        ))

    return [
        robot_state_publisher_node,
    ] + static_transform_nodes


def generate_launch_description():
    world = LaunchConfiguration('world')
    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resources/worlds', world]), ros2_supervisor=True, stream=True)
    robot_description_path = os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'resources/descriptions', 'EgoVehicle.urdf')))

    vehicle_driver = WebotsController(
        robot_name='ego_vehicle',
        parameters=[{'robot_description': robot_description_path}],
        respawn=True,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='main.wbt',
            description='Main simulation world',
        ),
        webots,
        webots._supervisor,
        vehicle_driver,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ] + get_ros2_nodes())
