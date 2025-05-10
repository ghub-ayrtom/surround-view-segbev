from launch.actions import GroupAction, DeclareLaunchArgument, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PythonExpression, LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode, LoadComposableNodes
import os
from launch.substitutions.path_join_substitution import PathJoinSubstitution
import pathlib
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher
from launch_ros.descriptions import ParameterFile, ComposableNode
from nav2_common.launch import RewrittenYaml
from launch.conditions import IfCondition
from launch import LaunchDescription
import launch


USE_SIM_TIME = True
PACKAGE_NAME = 'surround_view_segbev'

package_dir = get_package_share_directory(PACKAGE_NAME)

map_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'configs/slam_toolbox/maps/main_wbt/main_wbt.yaml'))
)
nav2_params_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'configs/nav2_params.yaml'))
)
ego_vehicle_urdf = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'resource/descriptions/EgoVehicle.urdf'))
)
pointcloud_to_laserscan_params_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'configs/pointcloud_to_laserscan_params.yaml')), 
)
mapper_params_online_async_yaml = os.path.join(
    package_dir, 
    pathlib.Path(os.path.join(package_dir, 'configs/slam_toolbox/mapper_params_online_async.yaml')), 
)
config_rviz = os.path.join(
    pathlib.Path(os.path.join(package_dir, 'configs/rviz/navigation.rviz')), 
)


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (namespace, '/', container_name)
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')
    world = LaunchConfiguration('world')

    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resource/worlds', world]))

    ego_vehicle_controller = WebotsController(
        respawn=True, 
        parameters=[{'robot_description': ego_vehicle_urdf}], 
        robot_name='ego_vehicle', 
    )

    lifecycle_nodes = [
        'map_server', 
        'amcl', 
        'controller_server', 
        'smoother_server', 
        'planner_server', 
        'behavior_server', 
        'bt_navigator', 
        'waypoint_follower', 
        'velocity_smoother', 
    ]

    remappings = [
        ('/tf', 'tf'), 
        ('/tf_static', 'tf_static'), 
    ]

    param_substitutions = {
        'use_sim_time': use_sim_time, 
        'yaml_filename': map_yaml_file, 
        'autostart': autostart, 
    }

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file, 
            root_key=namespace, 
            param_rewrites=param_substitutions, 
            convert_types=True, 
        ), 
        allow_substs=True, 
    )

    stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', 
        default_value='', 
        description='Top-level namespace', 
    )
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map', 
        default_value=map_yaml, 
        description='Full path to map yaml file to load', 
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='True', 
        description='Use simulation (Webots/Gazebo) clock if true', 
    )
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', 
        default_value='True', 
        description='Automatically startup the Nav2 stack', 
    )
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file', 
        default_value=nav2_params_yaml, 
        description='Full path to the ROS 2 parameters file to use for all launched nodes', 
    )
    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', 
        default_value='False', 
        description='Use composed bringup if True', 
    )
    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name', 
        default_value='nav2_container', 
        description='The name of conatiner that nodes will load in if use composition', 
    )
    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', 
        default_value='False', 
        description='Whether to respawn if a node crashes. Applied when composition is disabled', 
    )
    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', 
        default_value='info', 
        description='Logging level', 
    )
    declare_world_cmd = DeclareLaunchArgument(
        'world', 
        default_value='main.wbt', 
        description='Main simulation world', 
    )

    with open(ego_vehicle_urdf, 'r') as urdf:
        ego_vehicle_description = urdf.read()

    load_nodes = GroupAction(
        condition=IfCondition(PythonExpression(['not ', use_composition])), 
        actions=[
            Node(
                executable='map_server', 
                package='nav2_map_server', 
                name='map_server', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='amcl', 
                package='nav2_amcl', 
                name='amcl', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='lifecycle_manager', 
                package='nav2_lifecycle_manager', 
                name='lifecycle_manager_localization', 
                parameters=[{
                    'use_sim_time': use_sim_time, 
                    'autostart': autostart, 
                    'node_names': lifecycle_nodes, 
                }], 
                arguments=['--ros-args', '--log-level', log_level], 
                output='screen', 
            ), 
            Node(
                executable='controller_server', 
                package='nav2_controller', 
                parameters=[configured_params], 
                remappings=remappings + [('cmd_vel', 'cmd_vel_nav')], 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='smoother_server', 
                package='nav2_smoother', 
                name='smoother_server', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='planner_server', 
                package='nav2_planner', 
                name='planner_server', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='behavior_server', 
                package='nav2_behaviors', 
                name='behavior_server', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='bt_navigator', 
                package='nav2_bt_navigator', 
                name='bt_navigator', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='waypoint_follower', 
                package='nav2_waypoint_follower', 
                name='waypoint_follower', 
                parameters=[configured_params], 
                remappings=remappings, 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='velocity_smoother', 
                package='nav2_velocity_smoother', 
                name='velocity_smoother', 
                parameters=[configured_params], 
                remappings=remappings + [
                    ('cmd_vel', 'cmd_vel_nav'), 
                    ('cmd_vel_smoothed', 'cmd_vel'), 
                ], 
                arguments=['--ros-args', '--log-level', log_level], 
                respawn=use_respawn, 
                respawn_delay=2.0, 
                output='screen', 
            ), 
            Node(
                executable='lifecycle_manager', 
                package='nav2_lifecycle_manager', 
                name='lifecycle_manager_navigation', 
                parameters=[{
                    'use_sim_time': use_sim_time, 
                    'autostart': autostart, 
                    'node_names': lifecycle_nodes, 
                }], 
                arguments=['--ros-args', '--log-level', log_level], 
                output='screen', 
            ), 

            Node(
                executable='robot_state_publisher', 
                package='robot_state_publisher', 
                name='robot_state_publisher', 
                parameters=[{
                    'use_sim_time': USE_SIM_TIME, 
                    'robot_description': ego_vehicle_description, 
                }], 
                arguments=[ego_vehicle_urdf], 
                output='screen', 
            ), 
            Node(
                executable='nav2_path_planning_node', 
                package=PACKAGE_NAME, 
                name='nav2_path_planning_node', 
                parameters=[configured_params], 
                output='screen', 
            ), 
            Node(
                executable='pointcloud_to_laserscan_node', 
                package='pointcloud_to_laserscan', 
                name='pointcloud_to_laserscan', 
                parameters=[pointcloud_to_laserscan_params_yaml, {'use_sim_time': USE_SIM_TIME}], 
                output='screen', 
            ), 
            Node(
                executable='pointcloud_to_laserscan_bridge_node', 
                package=PACKAGE_NAME, 
                name='pointcloud_to_laserscan_bridge_node', 
                parameters=[{'use_sim_time': USE_SIM_TIME}], 
                output='screen', 
            ), 
            # LifecycleNode(
            #     executable='localization_slam_toolbox_node', 
            #     package='slam_toolbox', 
            #     name='slam_toolbox', 
            #     namespace='', 
            #     parameters=[
            #         mapper_params_online_async_yaml, 
            #         {
            #             'use_sim_time': USE_SIM_TIME, 
            #             'use_lifecycle_manager': False, 
            #         }, 
            #     ], 
            #     output='screen', 
            # ), 
            Node(
                executable='rviz2', 
                package='rviz2', 
                name='rviz2', 
                namespace='', 
                arguments=['-d', config_rviz], 
                output='screen', 
            ), 
        ]
    )

    load_composable_nodes = LoadComposableNodes(
        condition=IfCondition(use_composition), 
        target_container=container_name_full, 
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_map_server', 
                plugin='nav2_map_server::MapServer', 
                name='map_server', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_amcl', 
                plugin='nav2_amcl::AmclNode', 
                name='amcl', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_lifecycle_manager', 
                plugin='nav2_lifecycle_manager::LifecycleManager', 
                name='lifecycle_manager_localization', 
                parameters=[{
                    'use_sim_time': use_sim_time, 
                    'autostart': autostart, 
                    'node_names': lifecycle_nodes, 
                }], 
            ), 
            ComposableNode(
                package='nav2_controller', 
                plugin='nav2_controller::ControllerServer', 
                name='controller_server', 
                parameters=[configured_params], 
                remappings=remappings + [('cmd_vel', 'cmd_vel_nav')], 
            ), 
            ComposableNode(
                package='nav2_smoother', 
                plugin='nav2_smoother::SmootherServer', 
                name='smoother_server', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_planner', 
                plugin='nav2_planner::PlannerServer', 
                name='planner_server', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_behaviors', 
                plugin='behavior_server::BehaviorServer', 
                name='behavior_server', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_bt_navigator', 
                plugin='nav2_bt_navigator::BtNavigator', 
                name='bt_navigator', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_waypoint_follower', 
                plugin='nav2_waypoint_follower::WaypointFollower', 
                name='waypoint_follower', 
                parameters=[configured_params], 
                remappings=remappings, 
            ), 
            ComposableNode(
                package='nav2_velocity_smoother', 
                plugin='nav2_velocity_smoother::VelocitySmoother', 
                name='velocity_smoother', 
                parameters=[configured_params], 
                remappings=remappings + [
                    ('cmd_vel', 'cmd_vel_nav'), 
                    ('cmd_vel_smoothed', 'cmd_vel'), 
                ], 
            ), 
            ComposableNode(
                package='nav2_lifecycle_manager', 
                plugin='nav2_lifecycle_manager::LifecycleManager', 
                name='lifecycle_manager_navigation', 
                parameters=[{
                    'use_sim_time': use_sim_time, 
                    'autostart': autostart, 
                    'node_names': lifecycle_nodes, 
                }], 
            ), 
        ], 
    )

    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)

    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)
    ld.add_action(declare_use_respawn_cmd)
    ld.add_action(declare_log_level_cmd)
    ld.add_action(declare_world_cmd)

    ld.add_action(webots)
    ld.add_action(ego_vehicle_controller)

    ld.add_action(
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots, 
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())], 
            ), 
        ), 
    )

    ld.add_action(load_nodes)
    ld.add_action(load_composable_nodes)

    return ld
