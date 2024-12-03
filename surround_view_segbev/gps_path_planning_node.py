from rclpy.node import Node
import rclpy
from sensor_msgs.msg import NavSatFix, Image, Imu
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
            self.ego_vehicle_position = []

            self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
            self.create_subscription(Imu, '/imu', self.__imu_callback, 1)

            # active, latitude (Y), longitude (X), distance
            self.route = [
                [False, -3.0033403515877835, 1.9076212569577976e-05, 0.0], 
                # [False, -32.06279467008776, -7.789157866052413, 0.0], 
                # [False, -51.335662524526406, -2.525439545257973, 0.0], 
                # [False, -148.01160921471228, -28.23647957745029, 0.0], 
                # [False, -173.03383070766156, -28.133704336866714, 0.0], 
            ]

            self.current_route_point_index = 0
            self.create_subscription(NavSatFix, '/gps', self.__gps_callback, 1)

            self.drive_command = AckermannDrive()
            self.cmd_ackermann_publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', 1)

            self.create_timer(0.1, self.__navigate)
            self.create_subscription(Image, '/surround_view', self.__surround_view_callback, 10)

            self.callbacks_status = {
                'imu': False, 
                'gps': False, 
                'surround_view': False, 
            }

            self._logger.info('Successfully launched!')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __surround_view_callback(self, message):
        if self.callbacks_status['imu'] and self.callbacks_status['gps']:
            surround_view_image = CvBridge().imgmsg_to_cv2(message, 'bgr8')

            surround_view_frame = draw_path_on_surround_view(
                surround_view_image, 
                self.ego_vehicle_vector, 
                self.ego_vehicle_position, 
                self.route, 
            )

            cv2.imshow('surround_view', surround_view_frame)
            cv2.waitKey(1)

            if not self.callbacks_status['surround_view']:
                self.callbacks_status['surround_view'] = True


    def __imu_callback(self, message):
        x, y, z, w = message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w
        self.roll, self.pitch, self.yaw = euler_from_quaternion(x, y, z, w)

        self.ego_vehicle_vector = [math.cos(self.yaw), math.sin(self.yaw)]

        if not self.callbacks_status['imu']:
            self.callbacks_status['imu'] = True


    def __gps_callback(self, message):
        self.ego_vehicle_position = [message.latitude, message.longitude]

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

            if self.current_route_point_index < len(self.route) - 1:
                current_route_point_vector = self.route[self.current_route_point_index]
                next_route_point_vector = self.route[self.current_route_point_index + 1]

                median_route_point_vector = get_median_vector(current_route_point_vector, next_route_point_vector, 0.75)

                dy = median_route_point_vector[0] - current_latitude
                dx = median_route_point_vector[1] - current_longitude

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
