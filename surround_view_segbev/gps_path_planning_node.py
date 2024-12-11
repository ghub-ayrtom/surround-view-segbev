from rclpy.node import Node
import rclpy
from sensor_msgs.msg import NavSatFix, Image, MagneticField
import traceback
import math
from ackermann_msgs.msg import AckermannDrive
from configs import global_settings
from .scripts.utils import *
from cv_bridge import CvBridge


class GPSPathPlanningNode(Node):

    def __init__(self):
        try:
            super().__init__('gps_path_planning_node')

            self.p = 2.5
            self.max_speed = global_settings.EGO_VEHICLE_MAX_SPEED

            self.ego_vehicle_turn_angle = 0.0
            self.ego_vehicle_turn_angle_previous = 0.0
            self.max_steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE

            self.ego_vehicle_vector = []
            self.ego_vehicle_vector_relative = []

            self.yaw = 0.0
            self.create_subscription(MagneticField, '/compass', self.__compass_callback, 1)

            self.current_route_point_index = 0
            # active, latitude (Y), longitude (X), distance
            self.route = [
                [False, 16.991067331610402, -0.00010735764714900826, 0.0], 
                [False, 9.968392320728903, -0.00013375661671748124, 0.0], 
                [False, 1.3373338213758672, 5.0329741047400685, 0.0], 
            ]

            self.ego_vehicle_position = []
            self.create_subscription(NavSatFix, '/gps', self.__gps_callback, 1)

            self.drive_command = AckermannDrive()
            self.cmd_ackermann_publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', 1)

            self.create_timer(0.1, self.__navigate)
            self.create_subscription(Image, '/surround_view', self.__surround_view_callback, 10)

            self.callbacks_status = {
                'compass': False, 
                'gps': False, 
                'surround_view': False, 
            }

            self._logger.info('Successfully launched!')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __surround_view_callback(self, message):
        if self.callbacks_status['compass'] and self.callbacks_status['gps']:
            surround_view_image = CvBridge().imgmsg_to_cv2(message, 'bgr8')

            surround_view_frame = draw_path_on_surround_view(
                surround_view_image, 
                self.ego_vehicle_vector_relative, 
                self.yaw, 
                self.ego_vehicle_position, 
                self.route, 
            )

            cv2.imshow('surround_view', surround_view_frame)
            cv2.waitKey(1)

            if not self.callbacks_status['surround_view']:
                self.callbacks_status['surround_view'] = True


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

        if not self.callbacks_status['compass']:
            self.callbacks_status['compass'] = True


    def __gps_callback(self, message):
        self.ego_vehicle_position = [message.latitude, message.longitude]  # y, x

        # self._logger.info(f'{self.ego_vehicle_position}')

        if not self.callbacks_status['gps']:
            self.callbacks_status['gps'] = True


    def __navigate(self):
        if self.callbacks_status['gps']:
            if self.current_route_point_index >= len(self.route):
                self.drive_command.speed = 0.0
                self.drive_command.steering_angle = 0.0
                self.cmd_ackermann_publisher.publish(self.drive_command)
                return
            
            self.route[self.current_route_point_index][0] = True

            current_latitude, current_longitude = self.ego_vehicle_position
            _, target_latitude, target_longitude, _ = self.route[self.current_route_point_index]

            dy = target_latitude - current_latitude
            dx = target_longitude - current_longitude

            # if self.current_route_point_index < len(self.route) - 1:
            #     current_route_point_vector = self.route[self.current_route_point_index]
            #     next_route_point_vector = self.route[self.current_route_point_index + 1]

            #     median_route_point_vector = get_median_vector(current_route_point_vector, next_route_point_vector, 0.75)

            #     dy = median_route_point_vector[0] - current_latitude
            #     dx = median_route_point_vector[1] - current_longitude

            distance_to_target_route_point = math.sqrt(dy**2 + dx**2)
            self.route[self.current_route_point_index][3] = distance_to_target_route_point

            if distance_to_target_route_point < 5.0:
                self.route[self.current_route_point_index][0] = False
                self.route[self.current_route_point_index][3] = 0.0
                self.current_route_point_index += 1
                return

            Pk = 125
            Ik = 0.965

            angle_to_target_route_point = get_vectors_angle(self.ego_vehicle_vector, [dy, dx])
            self.ego_vehicle_turn_angle = float(min(1.0, max(-1.0, angle_to_target_route_point / Pk)))

            angle_to_target_route_point = (self.ego_vehicle_turn_angle_previous - self.ego_vehicle_turn_angle) * Ik
            self.ego_vehicle_turn_angle += angle_to_target_route_point

            self.ego_vehicle_turn_angle_previous = self.ego_vehicle_turn_angle

            self.drive_command.speed = min(self.max_speed, distance_to_target_route_point * self.p)
            self.drive_command.steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, math.degrees(angle_to_target_route_point)))

            # self.cmd_ackermann_publisher.publish(self.drive_command)


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
