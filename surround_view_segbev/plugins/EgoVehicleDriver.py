import rclpy
from ackermann_msgs.msg import AckermannDrive
import traceback
import time
from configs import global_settings


class EgoVehicleDriver:

    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__drive_command = AckermannDrive()

        rclpy.init(args=None)
        self.__node = rclpy.create_node('ego_vehicle_driver_node')
        self.__node._logger.info('Successfully launched!')

        self.__last_callback_time = 0.0
        self.__node.create_subscription(AckermannDrive, '/cmd_ackermann', self.__cmd_ackermann_callback, 1)


    def __cmd_ackermann_callback(self, message):
        self.__last_callback_time = time.time()

        if isinstance(message.speed, float) and isinstance(message.steering_angle, float):
            if message.speed > global_settings.EGO_VEHICLE_MAX_SPEED:
                message.speed = global_settings.EGO_VEHICLE_MAX_SPEED
            if message.steering_angle > global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE:
                message.steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE
            message.steering_angle *= 3.14 / 180  # Радианы

            self.__drive_command = message
        else:
            self.__stop()


    def __stop(self):
        self.__drive_command.speed = 0.0
        self.__drive_command.steering_angle = 0.0


    def step(self):
        try:
            rclpy.spin_once(self.__node, timeout_sec=0)
            current_step_time = time.time()

            if current_step_time - self.__last_callback_time > 1:
                self.__stop()

            self.__robot.setCruisingSpeed(self.__drive_command.speed)
            self.__robot.setSteeringAngle(self.__drive_command.steering_angle)
        except Exception as e:
            print(''.join(traceback.TracebackException.from_exception(e).format()))
