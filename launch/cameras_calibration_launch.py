from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import launch
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from webots_ros2_driver.webots_launcher import WebotsLauncher


USE_SIM_TIME = True  # Использовать симуляционное время
PACKAGE_NAME = 'surround_view_segbev'

package_dir = get_package_share_directory(PACKAGE_NAME)


# Позволяет получить список запускаемых узлов
def get_ros2_nodes():
    chessboards_controller_node = Node(
        executable='chessboards_controller_node', 
        package=PACKAGE_NAME, 
        name='chessboards_controller_node', 
        parameters=[{'use_sim_time': USE_SIM_TIME}], 
        output='screen', 
    )

    # Список статических преобразований координатных систем в утилите RViz2
    static_transforms = [
        ['map', 'odom'], 
        ['odom', 'base_link'], 
    ]
    
    static_transform_nodes = []

    # Создание узла для каждого из этих преобразований
    for transform in static_transforms:
        static_transform_nodes.append(Node(
            executable='static_transform_publisher', 
            package='tf2_ros', 
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
        chessboards_controller_node, 
    ] + static_transform_nodes


# Генерирует описание (параметры) процесса запуска
def generate_launch_description():
    world = LaunchConfiguration('world')
    webots = WebotsLauncher(world=PathJoinSubstitution([package_dir, 'resource/worlds', world]))  # Инициализация симулятора Webots

    return LaunchDescription([
        # Аргумент для указания мира симуляции
        DeclareLaunchArgument(
            'world', 
            default_value='waltz.wbt',  # Название файла, содержащего в себе описание сцены
            description='Cameras chessboard calibration world', 
        ), 
        webots, 
        # Регистрация обработчика событий, который будет срабатывать при завершении работы программы
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots, 
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())], 
            )
        ),
    ] + get_ros2_nodes())
