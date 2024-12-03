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


PACKAGE_NAME = 'surround_view_segbev'
USE_SIM_TIME = True

package_dir = get_package_share_directory(PACKAGE_NAME)
ego_vehicle_urdf = os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'resource/descriptions', 'EgoVehicle.urdf')))


def get_ros2_nodes():
    with open(ego_vehicle_urdf, 'r') as urdf:
        ego_vehicle_description = urdf.read()

    ego_vehicle_state_publisher_node = Node(
        executable='robot_state_publisher', 
        package='robot_state_publisher', 
        name='robot_state_publisher', 
        parameters=[{
            'use_sim_time': USE_SIM_TIME, 
            'robot_description': ego_vehicle_description, 
        }], 
        arguments=[ego_vehicle_urdf], 
        output='screen', 
    )

    surround_view_node = Node(
        executable='surround_view_node', 
        package=PACKAGE_NAME, 
        name='surround_view_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    gps_path_planning_node = Node(
        executable='gps_path_planning_node', 
        package=PACKAGE_NAME, 
        name='gps_path_planning_node', 
        parameters=[{'use_sim_time': False}], 
        output='screen', 
    )

    static_transforms = [
        ['map', 'odom'], 
        ['odom', 'base_link'], 
    ]
    
    static_transform_nodes = []

    for transform in static_transforms:
        static_transform_nodes.append(Node(
            executable='static_transform_publisher', 
            package='tf2_ros', 
            name='static_transform_publisher', 
            parameters=[{'use_sim_time': USE_SIM_TIME}], 
            arguments=['0', '0', '0', '0', '0', '0'] + transform, 
            output='screen', 
        ))

    return [
        ego_vehicle_state_publisher_node, 
        surround_view_node, 
        gps_path_planning_node, 
    ] + static_transform_nodes


def generate_launch_description():
    world = LaunchConfiguration('world')
    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resource/worlds', world]), stream=True)

    ego_vehicle_controller = WebotsController(
        respawn=True, 
        parameters=[{'robot_description': ego_vehicle_urdf}], 
        robot_name='ego_vehicle', 
    )

    os.environ['USING_EXTERN_CONTROLLER'] = 'True'

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', 
            default_value='main.wbt', 
            description='Main simulation world', 
        ), 
        webots, 
        ego_vehicle_controller, 
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots, 
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())], 
            )
        ),
    ] + get_ros2_nodes())
