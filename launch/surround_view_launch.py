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
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


USE_SIM_TIME = True
PACKAGE_NAME = 'surround_view_segbev'

package_dir = get_package_share_directory(PACKAGE_NAME)

ego_vehicle_urdf = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'resource/descriptions/EgoVehicle.urdf'))
)

# Симуляционное время не работает для gps_path_planning_node если задать ему только 
# 'use_sim_time': 'True' в parameters, поэтому была создана данная переменная
configured_params = ParameterFile(
    RewrittenYaml(
        # Можно указать путь абсолютно до любого файла просто, 
        # чтобы убрать ошибку "No such file or directory"
        source_file=ego_vehicle_urdf, 
        param_rewrites={'use_sim_time': str(USE_SIM_TIME)}, 
        root_key='', 
        convert_types=True, 
    ), 
    allow_substs=True, 
)


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
        parameters=[configured_params], 
        output='screen', 
    )

    static_transforms = [
        ['map', 'odom'], 
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
                '--qx', '0.00', 
                '--qy', '0.00', 
                '--qz', '0.00', 
                '--qw', '1.00', 
                # '--roll', '0.00', 
                # '--pitch', '0.00', 
                # '--yaw', '0.00', 
                '--frame-id', transform[0], 
                '--child-frame-id', transform[1], 
            ], 
            output='screen', 
        ))

    return [
        ego_vehicle_state_publisher_node, 
        # surround_view_node, 
        gps_path_planning_node, 
    ] + static_transform_nodes


def generate_launch_description():
    world = LaunchConfiguration('world')
    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resource/worlds', world]))

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
