from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image, MagneticField
import traceback
import math
from ackermann_msgs.msg import AckermannDrive
from surround_view_segbev.configs import global_settings, qos_profiles
from surround_view_segbev.scripts.utils import *
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped


class GPSPathPlanningNode(Node):
    def __init__(self):
        try:
            super().__init__('gps_path_planning_node')

            self.max_speed = global_settings.EGO_VEHICLE_MAX_SPEED
            self.max_steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE

            self.ego_vehicle_vector = []
            self.ego_vehicle_vector_relative = []

            self.yaw = 0.0
            self.create_subscription(MagneticField, '/compass', self.__compass_callback, qos_profiles.compass_qos)

            self.current_route_point_index = 0
            self.current_route_point_relative = None

            # active, latitude (Y), longitude (X), distance
            self.route = [
                # Прямой маршрут
                [False, -1.4129509338032613, -3.592190037567409, 0.0], 
                [False, -20.765200935279992, -5.843628462165249, 0.0], 
                [False, -44.744094027607225, -5.868065170391731, 0.0], 
                [False, -68.68892490861772, -9.127819865044577, 0.0], 
                [False, -101.06958371560515, -9.720227202100812, 0.0], 
                [False, -126.41332568745254, -22.344930829123705, 0.0], 
                [False, -148.8355197865538, -32.0232763017157, 0.0], 
                [False, -158.24739199611858, -36.37899373766482, 0.0], 
                [False, -184.3583541074493, -40.36716564136646, 0.0], 
                [False, -216.63725565640067, -49.80040946616456, 0.0], 
                [False, -226.08321407565063, -54.06991582316019, 0.0], 

                # Манёвр разворота
                [False, -238.76825539211907, -72.09706520141663, 0.0],  # Конечная точка прямого маршрута
                [False, -240.63281246089198, -60.08929785278786, 0.0],  # Разворот с использованием задней передачи
                [False, -228.72681284713371, -64.64053948786818, 0.0],  # Выравнивание эго-автомобиля в зоне разворота

                # Обратный маршрут
                [False, -205.93356014409082, -56.31834460171626, 0.0], 
                [False, -180.2142127138494, -43.30353481229337, 0.0], 
                [False, -147.53688422829327, -18.803022026884896, 0.0], 
                [False, -123.25208075159956, -11.556227609467937, 0.0], 
                [False, -98.99325199612764, -4.712499629779494, 0.0], 
                [False, -67.3851441049213, -10.301625475824743, 0.0], 
                [False, -39.87935651635924, -6.1180585996615955, 0.0], 
                [False, -15.5295532678878, 4.139480743096476, 0.0], 
                [False, 0.5200656120483349, 3.4487924524527473, 0.0], 

                [False, 16.482491991750244, -1.2168780265368024, 0.0],  # Конечная точка обратного маршрута
            ]

            self.__split_route()

            self.ego_vehicle_position = []
            self.create_subscription(PointStamped, '/ego_vehicle/gps', self.__gps_callback, qos_profiles.gps_qos)

            self.drive_command = AckermannDrive()
            self.cmd_ackermann_publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', qos_profiles.cmd_qos)

            self.turn_ego_vehicle_around = False
            self.move_ego_vehicle_forward = True
            self.stop_ego_vehicle_event_time = 0.0

            self.create_timer(0.2, self.__navigate)
            self.create_subscription(Image, '/surround_view', self.__surround_view_callback, qos_profiles.image_qos)

            self.callbacks_status = {
                'compass': False, 
                'gps': False, 
                'surround_view': False, 
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

        self.ego_vehicle_vector_relative = [
            self.ego_vehicle_vector[0] * math.cos(-self.yaw) - self.ego_vehicle_vector[1] * math.sin(-self.yaw), 
            self.ego_vehicle_vector[0] * math.sin(-self.yaw) + self.ego_vehicle_vector[1] * math.cos(-self.yaw), 
        ]

        if not math.isnan(self.yaw) and not self.callbacks_status['compass']:
            self.callbacks_status['compass'] = True

    def __split_route(self):
        route_splitted = [self.route[0]]

        for i in range(1, len(self.route)):
            start_point = self.route[i - 1]
            start_latitude, start_longitude = start_point[1], start_point[2]

            end_point = self.route[i]
            end_latitude, end_longitude = end_point[1], end_point[2]

            dx = end_longitude - start_longitude
            dy = end_latitude - start_latitude

            points_distance = math.sqrt(dx**2 + dy**2)

            if points_distance > 15.0:
                split_points_count = 2

                while points_distance / split_points_count > 10.0:
                    split_points_count += 1
                
                intermediate_latitudes = np.linspace(start_latitude, end_latitude, split_points_count)
                intermediate_longitudes = np.linspace(start_longitude, end_longitude, split_points_count)

                coordinates = list(zip(intermediate_latitudes, intermediate_longitudes))

                for j in range(1, len(coordinates)):
                    route_splitted.append([False, coordinates[j][0], coordinates[j][1], 0.0])
            else:
                route_splitted.append([False, end_latitude, end_longitude, 0.0])

        self.route = route_splitted

    def __gps_callback(self, message):
        self.ego_vehicle_position = [message.point.y, message.point.x]

        if not self.callbacks_status['gps']:
            self.callbacks_status['gps'] = True

    def __stop_ego_vehicle(self):
        self.drive_command.speed = 0.0
        self.drive_command.steering_angle = 0.0
        self.cmd_ackermann_publisher.publish(self.drive_command)

    def __navigate(self):
        command_time = self.get_clock().now().to_msg().sec

        if self.callbacks_status['compass'] and self.callbacks_status['gps']:
            # Если достигли конца маршрута движения
            if self.current_route_point_index >= len(self.route):
                self.__stop_ego_vehicle()
                return
            # Иначе, если достигли конечной точки прямого маршрута и 
            # собираемся идти на разворот с использованием задней передачи
            elif self.current_route_point_index == 22 and not self.turn_ego_vehicle_around:   # 12 без вызова __split_route
                self.__stop_ego_vehicle()

                if self.stop_ego_vehicle_event_time == 0.0:
                    self.stop_ego_vehicle_event_time = self.get_clock().now().to_msg().sec
                    command_time = self.get_clock().now().to_msg().sec

                # Стоим на месте 3 секунды
                if command_time - self.stop_ego_vehicle_event_time < 3.0:
                    return
                else:
                    self.turn_ego_vehicle_around = True
                    self.move_ego_vehicle_forward = False
                    self.stop_ego_vehicle_event_time = 0.0
            # Иначе, если закончили манёвр разворота и 
            # собираемся двигаться по обратному маршруту
            elif self.current_route_point_index == 23 and not self.move_ego_vehicle_forward:  # 13 без вызова __split_route
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
            
            self.route[self.current_route_point_index][0] = True

            current_latitude, current_longitude = self.ego_vehicle_position
            _, target_latitude, target_longitude, _ = self.route[self.current_route_point_index]

            dx = target_longitude - current_longitude
            dy = target_latitude - current_latitude

            distance_to_target_route_point = math.sqrt(dx**2 + dy**2)
            self.route[self.current_route_point_index][3] = distance_to_target_route_point

            # Если текущая точка маршрута движения была успешно достигнута по координатам GPS
            if distance_to_target_route_point < 5.0:
                self.route[self.current_route_point_index][3] = 0.0
                self.route[self.current_route_point_index][0] = False
                self.current_route_point_index += 1
                return

            angle_to_target_route_point = get_vectors_angle(self.ego_vehicle_vector, [dy, dx])

            self.drive_command.speed = min(self.max_speed, distance_to_target_route_point * 2.5)  # Pk = 2.5
            self.drive_command.steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, math.degrees(angle_to_target_route_point)))

            if not self.move_ego_vehicle_forward:
                self.drive_command.speed *= -1

            self.cmd_ackermann_publisher.publish(self.drive_command)

    def __surround_view_callback(self, message):
        if self.callbacks_status['compass'] and self.callbacks_status['gps']:
            surround_view_image = CvBridge().imgmsg_to_cv2(message, 'rgb8')

            surround_view_frame, self.current_route_point_relative = draw_path_on_surround_view(
                surround_view_image, 
                self.ego_vehicle_vector_relative, 
                self.yaw, 
                self.ego_vehicle_position, 
                self.current_route_point_index, 
                self.route, 
            )

            cv2.imshow('surround_view', surround_view_frame)
            cv2.waitKey(1)

            if not self.callbacks_status['surround_view']:
                self.callbacks_status['surround_view'] = True


def main(args=None):
    try:
        rclpy.init(args=args)

        node = GPSPathPlanningNode()
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
