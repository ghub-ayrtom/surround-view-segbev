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


USE_SIM_TIME = True
PACKAGE_NAME = 'surround_view_segbev'

package_dir = get_package_share_directory(PACKAGE_NAME)

nav2_params_yaml = os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'configs/nav2_params.yaml')))
ego_vehicle_urdf = os.path.join(package_dir, pathlib.Path(os.path.join(package_dir, 'resource/descriptions/EgoVehicle.urdf')))


def get_ros2_nodes():
    with open(ego_vehicle_urdf, 'r') as urdf:
        ego_vehicle_description = urdf.read()

    ego_vehicle_state_publisher_node = Node(
        package='robot_state_publisher', 
        executable='robot_state_publisher', 
        name='robot_state_publisher', 
        parameters=[{
            'use_sim_time': USE_SIM_TIME, 
            'robot_description': ego_vehicle_description, 
        }], 
        arguments=[ego_vehicle_urdf], 
        output='screen', 
    )

    surround_view_node = Node(
        package=PACKAGE_NAME, 
        executable='surround_view_node', 
        name='surround_view_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    gps_path_planning_node = Node(
        package=PACKAGE_NAME, 
        executable='gps_path_planning_node', 
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
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='static_transform_publisher', 
            parameters=[{'use_sim_time': USE_SIM_TIME}], 
            arguments=[
                '--x', '0.00', 
                '--y', '0.00', 
                '--z', '0.00', 
                '--roll', '0.00', 
                '--pitch', '0.00', 
                '--yaw', '0.00', 
                '--frame-id', transform[0], 
                '--child-frame-id', transform[1], 
            ], 
            output='screen', 
        ))

    return [
        ego_vehicle_state_publisher_node, 
        surround_view_node, 
        gps_path_planning_node, 
    ] + static_transform_nodes


def get_nav2_nodes():
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', 
        executable='lifecycle_manager', 
        name='lifecycle_manager_navigation', 
        output='screen', 
        parameters=[{
            'use_sim_time': USE_SIM_TIME, 
            'autostart': True, 
            'node_names': [
                'controller_server', 
                'planner_server', 
                'behavior_server', 
                'bt_navigator', 
                'waypoint_follower', 
            ]
        }]
    )

    controller_server = Node(
        package='nav2_controller', 
        executable='controller_server', 
        name='controller_server', 
        output='screen', 
        parameters=[nav2_params_yaml], 
    )

    planner_server = Node(
        package='nav2_planner', 
        executable='planner_server', 
        name='planner_server', 
        output='screen', 
        parameters=[{'use_sim_time': USE_SIM_TIME}, nav2_params_yaml], 
    )

    behavior_server = Node(
        package='nav2_behaviors', 
        executable='behavior_server', 
        name='behavior_server', 
        output='screen', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
    )

    bt_navigator = Node(
        package='nav2_bt_navigator', 
        executable='bt_navigator', 
        name='bt_navigator', 
        output='screen', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
    )

    waypoint_follower = Node(
        package='nav2_waypoint_follower', 
        executable='waypoint_follower', 
        name='waypoint_follower', 
        output='screen', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
    )

    return [
        lifecycle_manager, 
        controller_server, 
        planner_server, 
        behavior_server, 
        bt_navigator, 
        waypoint_follower, 
    ]


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
    ] + get_ros2_nodes())  # + get_nav2_nodes()
