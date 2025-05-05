import rclpy
from ackermann_msgs.msg import AckermannDrive
import traceback
import time
from configs import global_settings, qos_profiles
from sensor_msgs.msg import NavSatFix, MagneticField, PointCloud2
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from geometry_msgs.msg import Twist, Quaternion, Vector3, TransformStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32MultiArray, Float64
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import rclpy.wait_for_message


class EgoVehicleDriver:
    def init(self, webots_node, properties):
        rclpy.init(args=None)
        self.__node = rclpy.create_node('ego_vehicle_driver_node')

        self.__node_executor = MultiThreadedExecutor()
        self.__node_executor.add_node(self.__node)
        
        self.__robot = webots_node.robot
        self.__drive_command = AckermannDrive()

        self.__compass = self.__robot.getDevice('compass')
        self.__compass_message = MagneticField()
        self.__compass.enable(200)  # Обновление каждые 200 мс (5 Гц)

        self.__imu = self.__robot.getDevice('imu')
        self.__quaternion_message = Quaternion()
        self.__imu.enable(32)

        # Датчики угловой скорости (энкодеры) задних колёс
        self.__left_rear_sensor = self.__robot.getDevice('left_rear_sensor')
        self.__encoders_message = Float32MultiArray()
        self.__right_rear_sensor = self.__robot.getDevice('right_rear_sensor')

        self.__left_rear_sensor.enable(32)
        self.__right_rear_sensor.enable(32)

        self.__gps = self.__robot.getDevice('gps')
        self.__gps_mesage = NavSatFix()
        self.__gps.enable(256)

        self.__lidar = self.__robot.getDevice('lidar')
        self.__lidar.enable(100)

        self.__node.create_subscription(PointCloud2, '/ego_vehicle/lidar/point_cloud', self.__point_cloud_callback, qos_profiles.lidar_qos)
        self.__point_cloud_publisher = self.__node.create_publisher(PointCloud2, '/cloud_in', qos_profiles.lidar_qos)

        self.__compass_publisher = self.__node.create_publisher(MagneticField, '/compass', qos_profiles.compass_qos)
        self.__node.create_timer(0.2, self.__compass_publisher_callback)

        self.__tfs = TransformStamped()
        self.__tfs.header.frame_id = 'odom'
        self.__tfs.child_frame_id = 'base_link'

        self.__tf_broadcaster = TransformBroadcaster(self.__node)

        self.__odom = Odometry()
        self.__odom.header.frame_id = 'odom'
        self.__odom.child_frame_id = 'base_link'

        self.__odom_publisher = self.__node.create_publisher(Odometry, '/odom', qos_profiles.odometry_qos)
        # self.__odom_subscriber = self.__node.create_subscription(Odometry, '/odom', self.__odom_callback, qos_profiles.odometry_qos)

        self.__initial_pose_publisher = self.__node.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos_profiles.pose_qos)

        self.__imu_euler_publisher = self.__node.create_publisher(Vector3, '/imu_euler', qos_profiles.imu_qos)
        self.__imu_quaternion_publisher = self.__node.create_publisher(Quaternion, '/imu_quaternion', qos_profiles.imu_qos)
        
        self.__node.create_timer(0.032, self.__imu_odom_publisher_callback)

        # self.__wheel_encoders_publisher = self.__node.create_publisher(Float32MultiArray, '/wheel_encoders', qos_profiles.encoders_qos)
        # self.__node.create_timer(0.032, self.__wheel_encoders_publisher_callback)

        self.__gps_publisher = self.__node.create_publisher(NavSatFix, '/gps', qos_profiles.gps_qos)
        self.__node.create_timer(0.256, self.__gps_publisher_callback)

        self.__last_cmd_ackermann_callback_time = 0.0
        self.__node.create_subscription(AckermannDrive, '/cmd_ackermann', self.__cmd_ackermann_callback, qos_profiles.cmd_qos)

        self.__speed_message = Float64()
        self.__speed_publisher = self.__node.create_publisher(Float64, '/speed', qos_profiles.cmd_qos)

        self.__node.create_subscription(Twist, '/cmd_vel', self.__cmd_vel_callback, qos_profiles.cmd_qos)

        self.__node._logger.info('Successfully launched!')

    def __point_cloud_callback(self, message):
        message.header.stamp = self.__node.get_clock().now().to_msg()
        message.header.frame_id = 'lidar'
        self.__point_cloud_publisher.publish(message)

    def __compass_publisher_callback(self):
        coordinates = self.__compass.getValues()

        self.__compass_message.magnetic_field.x = coordinates[0]
        self.__compass_message.magnetic_field.y = coordinates[1]
        self.__compass_message.magnetic_field.z = coordinates[2]

        self.__compass_publisher.publish(self.__compass_message)

    def __odom_callback(self, message):
        initial_pose = PoseWithCovarianceStamped()

        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.__node.get_clock().now().to_msg()

        initial_pose.pose.pose = message.pose.pose
        initial_pose.pose.covariance = message.pose.covariance

        if rclpy.wait_for_message.wait_for_message(PoseWithCovarianceStamped, self.__node, '/amcl_pose', time_to_wait=5.0):
            self.__initial_pose_publisher.publish(initial_pose)
            self.__node.destroy_subscription(self.__odom_subscriber)

    def __imu_odom_publisher_callback(self):
        # orientation_euler = self.__imu.getRollPitchYaw()
        orientation_quaternion = self.__imu.getQuaternion()

        # if any(np.isnan(orientation_euler)):
        #     orientation_euler = [0.0, 0.0, 0.0]
        if any(np.isnan(orientation_quaternion)):
            orientation_quaternion = [0.0, 0.0, 0.0, 1.0]

        self.__quaternion_message.x = orientation_quaternion[0]
        self.__quaternion_message.y = orientation_quaternion[1]
        self.__quaternion_message.z = orientation_quaternion[2]
        self.__quaternion_message.w = orientation_quaternion[3]

        self.__tfs.header.stamp = self.__node.get_clock().now().to_msg()

        self.__tfs.transform.translation.x = self.__gps_mesage.longitude
        self.__tfs.transform.translation.y = self.__gps_mesage.latitude
        self.__tfs.transform.translation.z = 0.0

        self.__tfs.transform.rotation.x = self.__quaternion_message.x
        self.__tfs.transform.rotation.y = self.__quaternion_message.y
        self.__tfs.transform.rotation.z = self.__quaternion_message.z
        self.__tfs.transform.rotation.w = self.__quaternion_message.w

        self.__tf_broadcaster.sendTransform(self.__tfs)

        self.__odom.header.stamp = self.__node.get_clock().now().to_msg()

        self.__odom.pose.pose.position.x = self.__gps_mesage.longitude
        self.__odom.pose.pose.position.y = self.__gps_mesage.latitude
        self.__odom.pose.pose.position.z = 0.0

        self.__odom.pose.pose.orientation.x = self.__quaternion_message.x
        self.__odom.pose.pose.orientation.y = self.__quaternion_message.y
        self.__odom.pose.pose.orientation.z = self.__quaternion_message.z
        self.__odom.pose.pose.orientation.w = self.__quaternion_message.w

        self.__odom_publisher.publish(self.__odom)

        # self.__imu_euler_publisher.publish(orientation_euler)
        self.__imu_quaternion_publisher.publish(self.__quaternion_message)

    def __wheel_encoders_publisher_callback(self):
        ω_left = self.__left_rear_sensor.getValue()    # Получаем угловые скорости (рад/с) задних колёс
        ω_right = self.__right_rear_sensor.getValue()  #

        if np.isnan(ω_left) or self.__drive_command.speed == 0.0:
            ω_left = 0.0
        if np.isnan(ω_right) or self.__drive_command.speed == 0.0:
            ω_right = 0.0

        self.__encoders_message.data = [ω_left, ω_right]
        self.__wheel_encoders_publisher.publish(self.__encoders_message)

    def __gps_publisher_callback(self):
        coordinates = self.__gps.getValues()

        if not any(np.isnan(coordinate) for coordinate in coordinates):
            self.__gps_mesage.latitude = coordinates[1]
            self.__gps_mesage.longitude = coordinates[0]

            self.__gps_publisher.publish(self.__gps_mesage)

    def __cmd_ackermann_callback(self, message):
        self.__last_cmd_ackermann_callback_time = time.time()

        if isinstance(message.speed, float) and isinstance(message.steering_angle, float):
            # if message.speed > 0:
            #     if message.speed > global_settings.EGO_VEHICLE_MAX_SPEED:
            #         message.speed = global_settings.EGO_VEHICLE_MAX_SPEED
            # elif message.speed < 0:
            #     if message.speed < -global_settings.EGO_VEHICLE_MAX_SPEED:
            #         message.speed = -global_settings.EGO_VEHICLE_MAX_SPEED

            # if message.steering_angle > 0:
            #     if message.steering_angle > global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE:
            #         message.steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE
            # elif message.steering_angle < 0:
            #     if message.steering_angle < -global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE:
            #         message.steering_angle = -global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE
            # message.steering_angle *= 3.14 / 180  # Радианы

            self.__drive_command = message
        else:
            self.__stop()

    def __stop(self):
        self.__drive_command.speed = 0.0
        self.__drive_command.steering_angle = 0.0

    def __cmd_vel_callback(self, message):
        drive_command = AckermannDrive()

        drive_command.speed = message.linear.x
        drive_command.steering_angle = -message.angular.z

        self.__cmd_ackermann_callback(drive_command)

    def step(self):
        try:
            self.__node_executor.spin_once(timeout_sec=0.0)
            current_step_time = time.time()

            if current_step_time - self.__last_cmd_ackermann_callback_time > 1:
                self.__stop()

            self.__robot.setCruisingSpeed(self.__drive_command.speed)
            self.__robot.setSteeringAngle(self.__drive_command.steering_angle)

            self.__speed_message.data = self.__drive_command.speed
            self.__speed_publisher.publish(self.__speed_message)
        except Exception as e:
            print(''.join(traceback.TracebackException.from_exception(e).format()))
