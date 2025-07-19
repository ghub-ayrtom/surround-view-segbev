from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
import os
from launch.substitutions.path_join_substitution import PathJoinSubstitution
import pathlib
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


USE_SIM_TIME = True
PACKAGE_NAME = 'surround_view_segbev'

package_dir = get_package_share_directory(PACKAGE_NAME)

ego_vehicle_urdf = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'resource/descriptions/EgoVehicle.urdf'))
)
pointcloud_to_laserscan_params_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, f'{PACKAGE_NAME}/configs/pointcloud_to_laserscan_params.yaml')), 
)
mapper_params_online_async_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, f'{PACKAGE_NAME}/configs/slam_toolbox/mapper_params_online_async.yaml')), 
)
config_rviz = os.path.join(
    pathlib.Path(os.path.join(package_dir, f'{PACKAGE_NAME}/configs/rviz/mapping.rviz')), 
)


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

    async_pointcloud_merge_node = Node(
        executable='async_pointcloud_merge_node', 
        package='pointcloud_preprocessing', 
        name='async_pointcloud_merge_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    ego_vehicle_odometry_node = Node(
        executable='ego_vehicle_odometry_node', 
        package=PACKAGE_NAME, 
        name='ego_vehicle_odometry_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    pointcloud_to_laserscan_node = Node(
        executable='pointcloud_to_laserscan_node', 
        package='pointcloud_to_laserscan', 
        name='pointcloud_to_laserscan', 
        parameters=[
            pointcloud_to_laserscan_params_yaml, 
            {'use_sim_time': USE_SIM_TIME}
        ], 
        output='screen', 
    )
    pointcloud_to_laserscan_bridge_node = Node(
        executable='pointcloud_to_laserscan_bridge_node', 
        package=PACKAGE_NAME, 
        name='pointcloud_to_laserscan_bridge_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    async_slam_toolbox_node = LifecycleNode(
        executable='async_slam_toolbox_node', 
        package='slam_toolbox', 
        name='slam_toolbox', 
        namespace='', 
        parameters=[
          mapper_params_online_async_yaml, 
          {
            'use_sim_time': USE_SIM_TIME, 
            'use_lifecycle_manager': False, 
          }
        ], 
        output='screen', 
    )

    rviz = Node(
        executable='rviz2', 
        package='rviz2', 
        name='rviz2', 
        namespace='', 
        arguments=['-d', config_rviz], 
        output='screen', 
    )

    return [
        ego_vehicle_state_publisher_node, 
        async_pointcloud_merge_node, 
        ego_vehicle_odometry_node, 
        pointcloud_to_laserscan_node, 
        pointcloud_to_laserscan_bridge_node, 
        async_slam_toolbox_node, 
        rviz, 
    ]


def generate_launch_description():
    world = LaunchConfiguration('world')
    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resource/worlds', world]))

    ego_vehicle_controller = WebotsController(
        respawn=True, 
        parameters=[{'robot_description': ego_vehicle_urdf}], 
        robot_name='ego_vehicle', 
    )

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
