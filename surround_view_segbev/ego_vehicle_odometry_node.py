from rclpy.node import Node
import rclpy
import traceback
import math
from configs import qos_profiles
from .scripts.utils import *
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3
from std_msgs.msg import Header, Float32MultiArray, Float64
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import NavSatFix
import tf_transformations as tf


class EgoVehicleOdometryNode(Node):
    def __init__(self):
        try:
            super().__init__('ego_vehicle_odometry_node')

            self.odometry_transform = TransformStamped()
            self.odometry_transform.header = Header()
            self.odometry_transform.header.frame_id = 'odom'
            self.odometry_transform.child_frame_id = 'base_link'

            self.odometry_message = Odometry()
            self.odometry_message.header = Header()
            self.odometry_message.header.frame_id = 'odom'
            self.odometry_message.child_frame_id = 'base_link'

            self.ego_vehicle_orientation_euler = Vector3()
            self.ego_vehicle_orientation_quaternion = Quaternion()

            self.create_subscription(Vector3, '/imu_euler', self.__imu_euler_callback, qos_profiles.imu_qos)
            self.create_subscription(Quaternion, '/imu_quaternion', self.__imu_quaternion_callback, qos_profiles.imu_qos)

            self.wheel_radius = 0.4  # Радиус колеса в метрах (EgoVehicle.urdf)
            self.wheel_base = 1.7    # Примерное значение ширины колеи в метрах для Mercedes-Benz Sprinter

            self.rear_wheels_angular_velocities = Float32MultiArray()
            self.create_subscription(Float32MultiArray, '/wheel_encoders', self.__wheel_encoders_callback, qos_profiles.encoders_qos)

            self.vx = 0.0
            self.vy = 0.0
            self.vth = 0.0

            # Начальные GPS-координаты эго-автомобиля в метрах
            self.x = 0.00102224
            self.y = -1.14854e-09
            self.z = 0.0  # 0.369613

            self.create_subscription(Float64, '/speed', self.__speed_callback, qos_profiles.cmd_qos)

            self.ego_vehicle_position = [0.0, 0.0]
            self.ego_vehicle_position_previous = [0.0, 0.0]

            self.create_subscription(NavSatFix, '/gps', self.__gps_callback, qos_profiles.gps_qos)

            self.transform_broadcaster = TransformBroadcaster(self)
            self.odometry_publisher = self.create_publisher(Odometry, '/odom', qos_profiles.odometry_qos)

            self.last_odometry_publisher_callback_time = 0.0
            self.create_timer(0.1, self.__odometry_publisher_callback)

            self._logger.info('Successfully launched!')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __imu_euler_callback(self, message):
        self.ego_vehicle_orientation_euler = message  # roll, pitch, yaw

    def __imu_quaternion_callback(self, message):
        self.ego_vehicle_orientation_quaternion = message  # qx, qy, qz, qw

    def __wheel_encoders_callback(self, message):
        self.rear_wheels_angular_velocities = message  # left, right
        self.vth = (self.rear_wheels_angular_velocities.data[1] - self.rear_wheels_angular_velocities.data[0]) / self.wheel_base

    def __speed_callback(self, message):
        self.vx = message.data

    def __gps_callback(self, message):
        self.ego_vehicle_position = [message.longitude, message.latitude]
        self.ego_vehicle_position_previous = self.ego_vehicle_position

    def __odometry_publisher_callback(self):
        current_callback_time = self.get_clock().now().to_msg()
        # dt = current_callback_time.sec - self.last_odometry_publisher_callback_time

        # Δx = (self.vx * math.cos(self.vth) - self.vy * math.sin(self.vth)) * dt
        # Δy = (self.vx * math.sin(self.vth) + self.vy * math.cos(self.vth)) * dt
        # Δθ = self.vth * dt

        # self.x += Δx
        # self.y += Δy
        # self.z += Δθ

        # orientation_quaternion = tf.quaternion_from_euler(0.0, 0.0, self.z)

        self.odometry_transform.header.stamp = current_callback_time
        self.odometry_transform.transform.translation.x = self.ego_vehicle_position[0]  # self.x
        self.odometry_transform.transform.translation.y = self.ego_vehicle_position[1]  # self.y
        self.odometry_transform.transform.translation.z = 0.0
        self.odometry_transform.transform.rotation = self.ego_vehicle_orientation_quaternion

        # self.odometry_transform.transform.rotation.x = orientation_quaternion[0]
        # self.odometry_transform.transform.rotation.y = orientation_quaternion[1]
        # self.odometry_transform.transform.rotation.z = orientation_quaternion[2]
        # self.odometry_transform.transform.rotation.w = orientation_quaternion[3]

        self.transform_broadcaster.sendTransform(self.odometry_transform)

        self.odometry_message.header.stamp = current_callback_time
        self.odometry_message.pose.pose.position.x = self.ego_vehicle_position[0]  # self.x
        self.odometry_message.pose.pose.position.y = self.ego_vehicle_position[1]  # self.y
        self.odometry_message.pose.pose.position.z = 0.0
        self.odometry_message.pose.pose.orientation = self.ego_vehicle_orientation_quaternion

        # self.odometry_message.pose.pose.orientation.x = orientation_quaternion[0]
        # self.odometry_message.pose.pose.orientation.y = orientation_quaternion[1]
        # self.odometry_message.pose.pose.orientation.z = orientation_quaternion[2]
        # self.odometry_message.pose.pose.orientation.w = orientation_quaternion[3]

        self.odometry_message.twist.twist.linear.x = self.vx
        self.odometry_message.twist.twist.linear.y = self.vy
        self.odometry_message.twist.twist.linear.z = 0.0

        self.odometry_message.twist.twist.angular.x = 0.0
        self.odometry_message.twist.twist.angular.y = 0.0
        self.odometry_message.twist.twist.angular.z = self.vth

        self.odometry_publisher.publish(self.odometry_message)
        self.last_odometry_publisher_callback_time = current_callback_time.sec


def main(args=None):
    try:
        rclpy.init(args=args)

        node = EgoVehicleOdometryNode()
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
