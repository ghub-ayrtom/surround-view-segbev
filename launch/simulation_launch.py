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

    ego_vehicle_odometry_node = Node(
        executable='ego_vehicle_odometry_node', 
        package=PACKAGE_NAME, 
        name='ego_vehicle_odometry_node', 
        parameters=[{'use_sim_time': False}], 
        output='screen', 
    )

    gps_path_planning_node = Node(
        executable='gps_path_planning_node', 
        package=PACKAGE_NAME, 
        name='gps_path_planning_node', 
        parameters=[{'use_sim_time': False}], 
        output='screen', 
    )

    surround_view_node = Node(
        executable='surround_view_node', 
        package=PACKAGE_NAME, 
        name='surround_view_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    return [
        ego_vehicle_state_publisher_node, 
        ego_vehicle_odometry_node, 
        # gps_path_planning_node, 
        # surround_view_node, 
    ]


def get_nav2_nodes():
    lifecycle_manager = Node(
        executable='lifecycle_manager', 
        package='nav2_lifecycle_manager', 
        name='lifecycle_manager_navigation', 
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
        }], 
        output='screen', 
    )

    controller_server = Node(
        executable='controller_server', 
        package='nav2_controller', 
        name='controller_server', 
        parameters=[nav2_params_yaml], 
        output='screen', 
    )

    planner_server = Node(
        executable='planner_server', 
        package='nav2_planner', 
        name='planner_server', 
        parameters=[{'use_sim_time': USE_SIM_TIME}, nav2_params_yaml], 
        output='screen', 
    )

    behavior_server = Node(
        executable='behavior_server', 
        package='nav2_behaviors', 
        name='behavior_server', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    bt_navigator = Node(
        executable='bt_navigator', 
        package='nav2_bt_navigator', 
        name='bt_navigator', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    waypoint_follower = Node(
        executable='waypoint_follower', 
        package='nav2_waypoint_follower', 
        name='waypoint_follower', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
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
