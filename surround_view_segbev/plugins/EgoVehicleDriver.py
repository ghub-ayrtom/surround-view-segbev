import rclpy
from ackermann_msgs.msg import AckermannDrive
import traceback
import time
from configs import global_settings
from sensor_msgs.msg import NavSatFix, MagneticField
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class EgoVehicleDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__drive_command = AckermannDrive()

        self.__compass = self.__robot.getDevice('compass')
        self.__compass_message = MagneticField()
        self.__compass.enable(200)  # Обновление каждые 200 мс (5 Гц)

        self.__imu = self.__robot.getDevice('imu')
        self.__imu.enable(100)

        self.__odometry_message = Odometry()
        self.__odometry_message.header = Header()
        self.__odometry_message.header.frame_id = 'odom'
        self.__odometry_message.child_frame_id = 'base_link'

        self.__transform = TransformStamped()
        self.__transform.header = Header()
        self.__transform.header.frame_id = 'odom'
        self.__transform.child_frame_id = 'base_link'

        self.__gps = self.__robot.getDevice('gps')
        self.__gps_mesage = NavSatFix()
        self.__gps.enable(1000)

        # longitude, latitude
        self.__previous_position = [0.0, 0.0]

        rclpy.init(args=None)
        self.__node = rclpy.create_node('ego_vehicle_driver_node')

        self.__node_executor = MultiThreadedExecutor()
        self.__node_executor.add_node(self.__node)

        compass_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=3, 
        )

        self.__compass_publisher = self.__node.create_publisher(MagneticField, '/compass', compass_qos)
        self.__node.create_timer(0.1, self.__compass_publisher_callback)

        odometry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=5, 
        )

        self.__odometry_publisher = self.__node.create_publisher(Odometry, '/odom', odometry_qos)
        self.__transform_broadcaster = TransformBroadcaster(self.__node)

        self.__last_odometry_publisher_callback_time = 0.0
        self.__node.create_timer(0.2, self.__odometry_publisher_callback)

        gps_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=1, 
        )

        self.__gps_publisher = self.__node.create_publisher(NavSatFix, '/gps', gps_qos)
        self.__node.create_timer(1.0, self.__gps_publisher_callback)

        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=1, 
        )

        self.__last_cmd_ackermann_callback_time = 0.0
        self.__node.create_subscription(AckermannDrive, '/cmd_ackermann', self.__cmd_ackermann_callback, cmd_qos)

        self.__node.create_subscription(Twist, '/cmd_vel', self.__cmd_vel_callback, cmd_qos)

        self.__node._logger.info('Successfully launched!')

    def __compass_publisher_callback(self):
        coordinates = self.__compass.getValues()

        self.__compass_message.magnetic_field.x = coordinates[0]
        self.__compass_message.magnetic_field.y = coordinates[1]
        self.__compass_message.magnetic_field.z = coordinates[2]

        self.__compass_publisher.publish(self.__compass_message)

    def __odometry_publisher_callback(self):
        current_callback_time = time.time()
        dt = current_callback_time - self.__last_odometry_publisher_callback_time
        self.__last_odometry_publisher_callback_time = current_callback_time

        orientation = self.__imu.getQuaternion()

        if any(np.isnan(orientation)):
            orientation = [0.0, 0.0, 0.0, 1.0]

        # Рассчитываем линейную скорость
        dx = self.__gps_mesage.longitude - self.__previous_position[0]
        dy = self.__gps_mesage.latitude - self.__previous_position[1]
        vx = dx / dt
        vy = dy / dt

        self.__transform.header.stamp = self.__odometry_message.header.stamp = self.__node.get_clock().now().to_msg()

        self.__transform.transform.translation.x = self.__odometry_message.pose.pose.position.x = self.__gps_mesage.longitude
        self.__transform.transform.translation.y = self.__odometry_message.pose.pose.position.y = self.__gps_mesage.latitude
        self.__transform.transform.translation.z = self.__odometry_message.pose.pose.position.z = 0.0

        self.__transform.transform.rotation.x = self.__odometry_message.pose.pose.orientation.x = orientation[0]
        self.__transform.transform.rotation.y = self.__odometry_message.pose.pose.orientation.y = orientation[1]
        self.__transform.transform.rotation.z = self.__odometry_message.pose.pose.orientation.z = orientation[2]
        self.__transform.transform.rotation.w = self.__odometry_message.pose.pose.orientation.w = orientation[3]

        self.__odometry_message.twist.twist.linear.x = self.__drive_command.speed  # vx
        self.__odometry_message.twist.twist.linear.y = vy
        self.__odometry_message.twist.twist.angular.z = self.__drive_command.steering_angle

        self.__odometry_publisher.publish(self.__odometry_message)
        self.__transform_broadcaster.sendTransform(self.__transform)

    def __gps_publisher_callback(self):
        coordinates = self.__gps.getValues()

        if not any(np.isnan(coordinate) for coordinate in coordinates):
            self.__gps_mesage.latitude = coordinates[1]
            self.__gps_mesage.longitude = coordinates[0]

            self.__previous_position = self.__gps_mesage.longitude, self.__gps_mesage.latitude

            self.__gps_publisher.publish(self.__gps_mesage)

    def __cmd_ackermann_callback(self, message):
        self.__last_cmd_ackermann_callback_time = time.time()

        if isinstance(message.speed, float) and isinstance(message.steering_angle, float):
            if message.speed > 0:
                if message.speed > global_settings.EGO_VEHICLE_MAX_SPEED:
                    message.speed = global_settings.EGO_VEHICLE_MAX_SPEED
            elif message.speed < 0:
                if message.speed < -global_settings.EGO_VEHICLE_MAX_SPEED:
                    message.speed = -global_settings.EGO_VEHICLE_MAX_SPEED

            if message.steering_angle > 0:
                if message.steering_angle > global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE:
                    message.steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE
            elif message.steering_angle < 0:
                if message.steering_angle < -global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE:
                    message.steering_angle = -global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE
            message.steering_angle *= 3.14 / 180  # Радианы

            self.__drive_command = message
        else:
            self.__stop()

    def __stop(self):
        self.__drive_command.speed = 0.0
        self.__drive_command.steering_angle = 0.0

    def __cmd_vel_callback(self, message):
        drive_command = AckermannDrive()

        drive_command.speed = message.linear.x
        drive_command.steering_angle = message.angular.z

        self.__cmd_ackermann_callback(drive_command)

    def step(self):
        try:
            self.__node_executor.spin_once(timeout_sec=0)
            current_step_time = time.time()

            if current_step_time - self.__last_cmd_ackermann_callback_time > 1:
                self.__stop()

            self.__robot.setCruisingSpeed(self.__drive_command.speed)
            self.__robot.setSteeringAngle(self.__drive_command.steering_angle)
        except Exception as e:
            print(''.join(traceback.TracebackException.from_exception(e).format()))
