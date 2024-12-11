import rclpy
from ackermann_msgs.msg import AckermannDrive
import traceback
import time
from configs import global_settings
from sensor_msgs.msg import NavSatFix, MagneticField
from rclpy.executors import MultiThreadedExecutor
import numpy as np


class EgoVehicleDriver:

    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__drive_command = AckermannDrive()

        self.__compass = self.__robot.getDevice('compass')
        self.__compass_message = MagneticField()
        self.__compass.enable(10)  # Обновление каждые 10 мс

        self.__gps = self.__robot.getDevice('gps')
        self.__gps_mesage = NavSatFix()
        self.__gps.enable(10)

        rclpy.init(args=None)
        self.__node = rclpy.create_node('ego_vehicle_driver_node')

        self.__node_executor = MultiThreadedExecutor()
        self.__node_executor.add_node(self.__node)

        self.__compass_publisher = self.__node.create_publisher(MagneticField, '/compass', 1)
        self.__node.create_timer(0.1, self.__compass_publisher_callback)

        self.__gps_publisher = self.__node.create_publisher(NavSatFix, '/gps', 1)
        self.__node.create_timer(0.1, self.__gps_publisher_callback)

        self.__last_cmd_ackermann_callback_time = 0.0
        self.__node.create_subscription(AckermannDrive, '/cmd_ackermann', self.__cmd_ackermann_callback, 1)

        self.__node._logger.info('Successfully launched!')


    def __compass_publisher_callback(self):
        coordinates = self.__compass.getValues()

        self.__compass_message.magnetic_field.x = coordinates[0]
        self.__compass_message.magnetic_field.y = coordinates[1]
        self.__compass_message.magnetic_field.z = coordinates[2]

        self.__compass_publisher.publish(self.__compass_message)


    def __gps_publisher_callback(self):
        coordinates = self.__gps.getValues()

        if not any(np.isnan(coordinate) for coordinate in coordinates):
            self.__gps_mesage.latitude = coordinates[1]
            self.__gps_mesage.longitude = coordinates[0]

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
