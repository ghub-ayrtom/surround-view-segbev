import rclpy
from ackermann_msgs.msg import AckermannDrive
import traceback
import time
from surround_view_segbev.configs import global_settings, qos_profiles
from sensor_msgs.msg import MagneticField
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy.wait_for_message
from rosgraph_msgs.msg import Clock


class EgoVehicleDriver:
    def init(self, webots_node, properties):
        rclpy.init(args=None)
        self.__node = rclpy.create_node('ego_vehicle_driver_node')

        self.__node_executor = MultiThreadedExecutor()
        self.__node_executor.add_node(self.__node)
        
        self.__clock = Clock()

        self.__clock_publisher = self.__node.create_publisher(Clock, '/clock', 10)
        self.__node.create_timer(0.01, self.__publish_clock)  # basicTimeStep в Webots равен 10 мс (100 Гц)

        self.__robot = webots_node.robot
        self.__drive_command = AckermannDrive()

        self.__compass = self.__robot.getDevice('compass')
        self.__compass_message = MagneticField()
        self.__compass.enable(200)  # Обновление каждые 200 мс (5 Гц)

        self.__compass_publisher = self.__node.create_publisher(MagneticField, '/compass', qos_profiles.compass_qos)
        self.__node.create_timer(0.2, self.__compass_publisher_callback)

        self.__initial_pose_publisher = self.__node.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos_profiles.pose_qos)
        self.__odom_subscriber = self.__node.create_subscription(Odometry, '/odom', self.__odom_callback, qos_profiles.odometry_qos)

        self.__speed_factor = 14  # 6
        self.__last_cmd_ackermann_callback_time = 0.0

        self.__node.create_subscription(Twist, '/cmd_vel', self.__cmd_vel_callback, qos_profiles.cmd_qos)
        self.__node.create_subscription(AckermannDrive, '/cmd_ackermann', self.__cmd_ackermann_callback, qos_profiles.cmd_qos)

        self.__node.create_timer(1.0, self.__dipped_beams_callback)

        self.__node.get_logger().info('Successfully launched!')

    def __publish_clock(self):
        now = time.time()

        sec = int(now)
        nanosec = int((now - sec) * 1e9)

        self.__clock.clock.sec = sec
        self.__clock.clock.nanosec = nanosec

        self.__clock_publisher.publish(self.__clock)

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

        message.pose.pose.position.y -= 0.85

        initial_pose.pose.pose = message.pose.pose
        initial_pose.pose.covariance = message.pose.covariance

        if rclpy.wait_for_message.wait_for_message(PoseWithCovarianceStamped, self.__node, '/amcl_pose', time_to_wait=5.0):
            self.__initial_pose_publisher.publish(initial_pose)
            self.__node.destroy_subscription(self.__odom_subscriber)

    def __cmd_ackermann_callback(self, message):
        self.__last_cmd_ackermann_callback_time = time.time()

        if isinstance(message.speed, float) and isinstance(message.steering_angle, float):
            if message.speed > 0:
                if message.speed > global_settings.EGO_VEHICLE_MAX_SPEED:
                    message.speed = global_settings.EGO_VEHICLE_MAX_SPEED
            elif message.speed < 0:
                if message.speed < -global_settings.EGO_VEHICLE_MAX_SPEED:
                    message.speed = -global_settings.EGO_VEHICLE_MAX_SPEED
            else:
                pass

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
        self.__drive_command.speed = message.linear.x * self.__speed_factor  # message.linear.x = (-)0.5
        '''
            * 2  - Model Predictive Path Integral
            * 10 - Regulated Pure Pursuit
        '''
        self.__drive_command.steering_angle = (-message.angular.z * 180 / 3.14) * 10  # Градусы

        if self.__drive_command.speed < 0.0:
            self.__drive_command.steering_angle *= -1

        self.__cmd_ackermann_callback(self.__drive_command)

    def __dipped_beams_callback(self):
        if self.__drive_command.speed != 0:
            self.__robot.setDippedBeams(not self.__robot.getDippedBeams())
        else:
            self.__robot.setDippedBeams(False)

    def step(self):
        try:
            rclpy.spin_once(self.__node, timeout_sec=0.0)
            current_step_time = time.time()

            if current_step_time - self.__last_cmd_ackermann_callback_time > 1:
                self.__stop()

            self.__robot.setCruisingSpeed(self.__drive_command.speed)
            self.__robot.setSteeringAngle(self.__drive_command.steering_angle)
        except Exception as e:
            print(''.join(traceback.TracebackException.from_exception(e).format()))
