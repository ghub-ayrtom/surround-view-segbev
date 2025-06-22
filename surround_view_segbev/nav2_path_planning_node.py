from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Imu, MagneticField
import traceback
import math
from ackermann_msgs.msg import AckermannDrive
from configs import global_settings, qos_profiles
from .scripts.utils import *
from geometry_msgs.msg import Quaternion, PoseStamped, PointStamped, TransformStamped
from std_msgs.msg import Int8, Bool, Header
from nav2_msgs.srv import GetCostmap
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class Nav2PathPlanningNode(Node):
    def __init__(self):
        try:
            super().__init__('gps_path_planning_node')

            self.max_speed = global_settings.EGO_VEHICLE_MAX_SPEED
            self.max_steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE

            self.ego_vehicle_vector = []

            self.yaw = 0.0
            self.create_subscription(MagneticField, '/compass', self.__compass_callback, qos_profiles.compass_qos)

            self.current_route_point_index = 0
            self.target_route_point_distance_tolerance = 10.0  # Метры

            # active, latitude (Y), longitude (X), distance
            self.route = [
                # Прямой маршрут
                [False, -11.963690028682464, 5.2088432228038036e-05, 0.0], 
                [False, -25.67394982981072, -0.49400071880558144, 0.0], 
                [False, -46.41808641024842, -2.4318363345312566, 0.0], 
                [False, -68.61867245073576, -5.094688322299945, 0.0], 
                [False, -86.63189742590187, -9.060645314363162, 0.0], 
                [False, -107.24139435613779, -14.148174564711638, 0.0], 
                [False, -123.9810719748813, -18.29085928374735, 0.0], 
                [False, -148.88037876917127, -26.212097694766854, 0.0], 
                [False, -167.96186154599184, -32.98349023383825, 0.0], 
                [False, -187.21385833870892, -41.10184125272146, 0.0], 
                [False, -205.29747051023415, -49.262066866216664, 0.0], 
                [False, -224.51754976658805, -59.25226950479315, 0.0], 

                # Манёвр разворота
                [False, -237.6226944137152, -72.29597944739069, 0.0],   # Конечная точка прямого маршрута
                [False, -239.0611297960799, -62.53582496176923, 0.0],   # Разворот с использованием задней передачи
                [False, -231.06157489530074, -64.46095083773474, 0.0],  # Выравнивание эго-автомобиля в зоне разворота

                # Обратный маршрут
                [False, -206.06446109548503, -48.98740324555575, 0.0], 
                [False, -184.30030105310024, -39.803618149329864, 0.0], 
                [False, -163.81630319575325, -31.16002068464577, 0.0], 
                [False, -141.61071257981888, -23.38744257937712, 0.0], 
                [False, -120.16468769450869, -17.09848324259087, 0.0], 
                [False, -93.41271420518974, -11.084197931988644, 0.0], 
                [False, -72.57038964547466, -7.506952012684565, 0.0], 
                [False, -49.262673975374994, -4.050975531279891, 0.0], 
                [False, -19.85974918467336, -1.8194805849739484, 0.0], 
                [False, -6.61277317789151, -1.350083839530962, 0.0], 
                [False, 7.953267541770455, -0.8339659896667804, 0.0], 
                [False, 15.789484322454202, -0.552692217589275, 0.0], 

                [False, 15.694846690561448, 0.04872481437466736, 0.0],  # Конечная точка обратного маршрута
            ]

            self.imu_data = Quaternion()
            self.create_subscription(Imu, '/ego_vehicle/imu', self.__imu_callback, qos_profiles.imu_qos)

            self.ego_vehicle_position = []
            self.create_subscription(PointStamped, '/ego_vehicle/gps', self.__gps_callback, qos_profiles.gps_qos)

            self.tfs = TransformStamped()
            self.tfs.header.frame_id = 'odom'
            self.tfs.child_frame_id = 'base_link'

            self.tf_broadcaster = TransformBroadcaster(self)

            self.odom = Odometry()
            self.odom.header.frame_id = 'odom'
            self.odom.child_frame_id = 'base_link'

            # В течение нескольких секунд после запуска решения, эго-автомобиль будет находиться в центре глобального фрейма, 
            # однако, из-за его смещения, данные одометрии в этот момент не будут соответствовать фактическому месторасположению робота на карте
            self.skip_odom_publishes_count = 100

            self.odom_publisher = self.create_publisher(Odometry, '/odom', qos_profiles.odometry_qos)

            self.drive_command = AckermannDrive()
            self.cmd_ackermann_publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', qos_profiles.cmd_qos)

            self.speed_factor = Int8()
            self.speed_factor_publisher = self.create_publisher(Int8, '/speed_factor', qos_profiles.cmd_qos)

            self.nav2_switch_command = Bool()
            self.nav2_switch_publisher = self.create_publisher(Bool, '/nav2_switch', qos_profiles.default_qos)

            self.turn_ego_vehicle_around = False
            self.move_ego_vehicle_forward = True
            self.stop_ego_vehicle_event_time = 0.0

            self.current_goal = PoseStamped()
            self.current_goal.header = Header()
            self.current_goal.header.frame_id = 'map'  # 'odom'

            self.global_costmap = None
            self.global_costmap_width = 0.0
            self.global_costmap_height = 0.0
            self.global_costmap_origin = [-83.2, -246, 0]  # main_wbt.yaml
            self.global_costmap_resolution = None

            self.get_global_costmap_client = self.create_client(GetCostmap, '/global_costmap/get_costmap')

            self.goal_handle = None
            self.current_goal_pose_requested = False
            self.navigate_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

            self.create_timer(0.2, self.__get_global_costmap_request)
            self.create_timer(0.2, self.__navigate)

            self.callbacks_status = {
                'compass': False, 
                'gps': False, 
            }

            self.get_logger().info('Successfully launched!')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __compass_callback(self, message):
        x, y, z = message.magnetic_field.x, message.magnetic_field.y, message.magnetic_field.z
        self.yaw = math.atan2(y, x)

        if self.yaw < 0.0:
            self.yaw += 2.0 * math.pi  # [0, 2π]
        self.ego_vehicle_vector = [math.cos(self.yaw), math.sin(self.yaw)]

        if not math.isnan(self.yaw) and not self.callbacks_status['compass']:
            self.callbacks_status['compass'] = True

    def __imu_callback(self, message):
        self.imu_data = message.orientation

    def __gps_callback(self, message):
        self.ego_vehicle_position = [message.point.y, message.point.x]  # y, x

        # self.get_logger().info(f'[False, {self.ego_vehicle_position[0]}, {self.ego_vehicle_position[1]}, 0.0], ')

        self.tfs.header.stamp = self.get_clock().now().to_msg()

        self.tfs.transform.translation.x = message.point.x
        self.tfs.transform.translation.y = message.point.y
        self.tfs.transform.rotation = self.imu_data

        self.tf_broadcaster.sendTransform(self.tfs)

        self.odom.header.stamp = self.get_clock().now().to_msg()

        self.odom.pose.pose.position.x = message.point.x
        self.odom.pose.pose.position.y = message.point.y
        self.odom.pose.pose.orientation = self.imu_data

        if self.skip_odom_publishes_count == 0:
            self.odom_publisher.publish(self.odom)
        else:
            self.skip_odom_publishes_count -= 1

        if not self.callbacks_status['gps']:
            self.callbacks_status['gps'] = True

    def __stop_ego_vehicle(self):
        self.drive_command.speed = 0.0
        self.drive_command.steering_angle = 0.0
        self.cmd_ackermann_publisher.publish(self.drive_command)

    def __get_global_costmap_request(self):
        if not self.get_global_costmap_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("The service GetMap not available!")
            return False

        request = GetCostmap.Request()
        future = self.get_global_costmap_client.call_async(request)
        future.add_done_callback(self.__get_global_costmap_response)

    def __get_global_costmap_response(self, future):
        try:
            if not future.result():
                self.get_logger().error('GetMap request error!')
            else:
                self.global_costmap = future.result().map

                self.global_costmap_width = self.global_costmap.metadata.size_x
                self.global_costmap_height = self.global_costmap.metadata.size_y

                if all(value is None for value in [self.global_costmap_resolution]):  # self.global_costmap_origin
                    # self.global_costmap_origin = self.global_costmap.metadata.origin.position
                    self.global_costmap_resolution = self.global_costmap.metadata.resolution
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __check_goal_pose_obstacle_collision(self, goal_pose):
        '''
            status_codes:
                0 - текущая точка маршрута находится в области проезда
                1 - текущая точка маршрута находится в области препятствия
                2 - текущая точка маршрута находится в области за границами карты
                3 - глобальная карта стоимости не была получена или пуста
        '''

        if self.global_costmap is not None and self.global_costmap.data and len(self.global_costmap.data) != 0:
            # Преобразуем абсолютные GPS-координаты Goal Pose в относительные координаты Global Costmap
            mx = round((goal_pose.pose.position.x - self.global_costmap_origin[0]) / self.global_costmap_resolution)  # self.global_costmap_origin.x
            my = round((goal_pose.pose.position.y - self.global_costmap_origin[1]) / self.global_costmap_resolution)  # self.global_costmap_origin.y

            if mx < 0 or my < 0 or mx >= self.global_costmap_width or my >= self.global_costmap_height:
                self.get_logger().warning('The current Goal Pose is outside the global costmap')
                return 2

            index = my * self.global_costmap_width + mx
            cost = self.global_costmap.data[index]

            # self.get_logger().info(f'[{goal_pose.pose.position.x}, {goal_pose.pose.position.y}] -> {cost}')

            if cost >= 100:
                return 1
            else:
                return 0
        else:
            return 3

    def __send_goal_pose_request(self):
        request = NavigateToPose.Goal()

        request.pose.header.frame_id = self.current_goal.header.frame_id
        request.pose.header.stamp = self.get_clock().now().to_msg()

        request.pose.pose.position = self.current_goal.pose.position
        request.pose.pose.orientation = self.current_goal.pose.orientation

        send_goal_pose_future = self.navigate_to_pose_client.send_goal_async(request)
        send_goal_pose_future.add_done_callback(self.__send_goal_pose_response)

    def __send_goal_pose_response(self, future):
        try:
            if not future.result():
                self.get_logger().error('NavigateToPose request error!')
            else:
                self.goal_handle = future.result()

                if not self.goal_handle.accepted:
                    self.current_goal_pose_requested = False
                    self.get_logger().warning('The current Goal Pose was rejected by the server')
                else:
                    status_code = self.__check_goal_pose_obstacle_collision(self.current_goal)  # 0

                    if status_code == 0:
                        self.current_goal_pose_requested = True
                        self.get_logger().info('The current Goal Pose was accepted by the server')
                    elif self.goal_handle:
                        self.current_goal_pose_requested = False

                        cancel_goal_pose_future = self.navigate_to_pose_client._cancel_goal_async(self.goal_handle)
                        cancel_goal_pose_future.add_done_callback(self.__cancel_goal_pose_response)

                        if status_code == 1:
                            # Если движение осуществляется по прямому маршруту
                            if self.current_route_point_index < 14:
                                self.route[self.current_route_point_index][1] += 1.0  # Смещаем точку на себя
                            # Иначе, если движение осуществляется по обратному маршруту
                            elif self.current_route_point_index > 14:
                                self.route[self.current_route_point_index][1] -= 1.0  # Также смещаем её на себя
                        elif status_code == 2:
                            if self.current_route_point_index < 14:
                                self.route[self.current_route_point_index][1] += 1.0  # Смещаем точку на себя 
                                self.route[self.current_route_point_index][2] -= 1.0  # и вправо
                            elif status_code == 2 and self.current_route_point_index > 14:
                                self.route[self.current_route_point_index][1] -= 1.0  # Также смещаем её на себя 
                                self.route[self.current_route_point_index][2] -= 1.0  # и вправо
                    else:
                        self.get_logger().warning('No active Goal Pose to cancel')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __cancel_goal_pose_response(self, future):
        try:
            if not future.result():
                self.get_logger().error('ActionClient request error!')
            else:
                response = future.result()

                if response:
                    self.get_logger().info('The current Goal Pose has been successfully canceled!')
                else:
                    self.get_logger().warning('Failed to cancel current Goal Pose')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def set_distance_tolerance(self, distance):
        self.target_route_point_distance_tolerance = distance

    def __navigate(self):
        command_time = self.get_clock().now().to_msg().sec

        if self.callbacks_status['compass'] and self.callbacks_status['gps']:
            # Если достигли конца маршрута движения
            if self.current_route_point_index >= len(self.route):
                self.__stop_ego_vehicle()
                return
            # Иначе, если достигли конечной точки прямого маршрута и 
            # собираемся идти на разворот с использованием задней передачи
            elif self.current_route_point_index == 13 and not self.turn_ego_vehicle_around:
                self.__stop_ego_vehicle()

                if self.stop_ego_vehicle_event_time == 0.0:
                    self.stop_ego_vehicle_event_time = self.get_clock().now().to_msg().sec
                    command_time = self.get_clock().now().to_msg().sec

                # Стоим на месте 3 секунды для стабилизации локализации
                if command_time - self.stop_ego_vehicle_event_time < 3.0:
                    return
                else:
                    self.turn_ego_vehicle_around = True
                    self.move_ego_vehicle_forward = False
                    self.stop_ego_vehicle_event_time = 0.0
            # Иначе, если закончили манёвр разворота и 
            # собираемся двигаться по обратному маршруту
            elif self.current_route_point_index == 14 and not self.move_ego_vehicle_forward:
                self.__stop_ego_vehicle()

                if self.stop_ego_vehicle_event_time == 0.0:
                    self.stop_ego_vehicle_event_time = self.get_clock().now().to_msg().sec
                    command_time = self.get_clock().now().to_msg().sec

                if command_time - self.stop_ego_vehicle_event_time < 3.0:
                    return
                else:
                    self.turn_ego_vehicle_around = False
                    self.move_ego_vehicle_forward = True
                    self.stop_ego_vehicle_event_time = 0.0
            elif self.current_route_point_index == 15:
                # Переключаемся на движение по Nav2
                self.nav2_switch_command.data = False
                self.nav2_switch_publisher.publish(self.nav2_switch_command)
            
            self.route[self.current_route_point_index][0] = True

            current_latitude, current_longitude = self.ego_vehicle_position
            _, target_latitude, target_longitude, _ = self.route[self.current_route_point_index]

            dx = target_longitude - current_longitude
            dy = target_latitude - current_latitude

            distance_to_target_route_point = math.sqrt(dx**2 + dy**2)
            self.route[self.current_route_point_index][3] = distance_to_target_route_point

            # Если текущая точка маршрута движения была успешно достигнута по координатам Nav 2 (4 - STATUS_SUCCEEDED)
            if self.goal_handle and self.goal_handle.status == 4 and self.current_goal_pose_requested:
                self.route[self.current_route_point_index][3] = 0.0
                self.route[self.current_route_point_index][0] = False

                self.current_goal_pose_requested = False
                self.current_route_point_index += 1  # Движемся к следующей

                return

            self.speed_factor.data = 6  # 14
            self.speed_factor_publisher.publish(self.speed_factor)

            # Манёвр разворота и завершение маршрута жёстко заданы без использования Nav2
            if self.current_route_point_index in { 12, 13, 14, 27 }:
                self.set_distance_tolerance(3.0)

                # Если текущая точка маршрута движения была успешно достигнута по координатам GPS
                if distance_to_target_route_point < self.target_route_point_distance_tolerance:
                    self.route[self.current_route_point_index][3] = 0.0
                    self.route[self.current_route_point_index][0] = False
                    self.current_route_point_index += 1
                    return
                else:
                    if not self.nav2_switch_command.data:
                        cancel_goal_pose_future = self.navigate_to_pose_client._cancel_goal_async(self.goal_handle)
                        cancel_goal_pose_future.add_done_callback(self.__cancel_goal_pose_response)

                        # Переключаемся на движение по GPS
                        self.nav2_switch_command.data = True
                        self.nav2_switch_publisher.publish(self.nav2_switch_command)

                    angle_to_target_route_point = get_vectors_angle(self.ego_vehicle_vector, [dy, dx])

                    self.drive_command.speed = min(self.max_speed, distance_to_target_route_point * 2.5)  # Pk = 2.5
                    self.drive_command.steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, math.degrees(angle_to_target_route_point)))

                    if not self.move_ego_vehicle_forward:
                        self.drive_command.speed *= -1

                    self.cmd_ackermann_publisher.publish(self.drive_command)
            elif not self.current_goal_pose_requested:
                self.set_distance_tolerance(10.0)

                self.current_goal.header.stamp = self.get_clock().now().to_msg()
                self.current_goal.pose.position.x = target_longitude
                self.current_goal.pose.position.y = target_latitude

                self.__send_goal_pose_request()


def main(args=None):
    try:
        rclpy.init(args=args)

        node = Nav2PathPlanningNode()
        rclpy.spin(node)
        node.destroy_node()

        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(''.join(traceback.TracebackException.from_exception(e).format()))
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
