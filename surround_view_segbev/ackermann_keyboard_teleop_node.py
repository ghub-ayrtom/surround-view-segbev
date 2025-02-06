import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
import sys, select, termios, tty
from numpy import clip
from configs import global_settings, qos_profiles
import traceback


control_keys_bindings = {
    '\x77': ['w', (1.0, 0.0)], 
    '\x61': ['a', (0.0, -1.0)], 
    '\x73': ['s', (-1.0, 0.0)], 
    '\x64': ['d', (0.0, 1.0)], 
    '\x20': ['space', (0.0, 0.0)], 
    '\x09': ['tab', (0.0, 0.0)], 
}


class AckermannKeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('ackermann_keyboard_teleop_node')

        self.current_speed = 0.0
        self.current_steering_angle = 0.0

        max_speed = global_settings.EGO_VEHICLE_MAX_SPEED
        max_steering_angle = global_settings.EGO_VEHICLE_MAX_STEERING_ANGLE

        self.speed_range = [-float(max_speed), float(max_speed)]
        self.steering_angle_range = [-float(max_steering_angle), float(max_steering_angle)]

        for key in control_keys_bindings:
            control_keys_bindings[key][1] = (
                control_keys_bindings[key][1][0] * float(max_speed) / 5, 
                control_keys_bindings[key][1][1] * float(max_steering_angle) / 5
            )

        self.message = AckermannDrive()
        self.cmd_ackermann_publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', qos_profiles.cmd_qos)

        self._logger.info(
            f'\n\n* * * * * * * * * * * * * * * * * * * * * * * * *\n'
            f'*                                               *\n'
            f'*   <WASD> to change speed and steering angle   *\n'
            f'*   <Space> to stop, <Tab> to align wheels      *\n'
            f'*   <Ctrl-C> or <Q> to exit                     *\n'
            f'*                                               *'
            f'\n* * * * * * * * * * * * * * * * * * * * * * * * *\n'
        )

        self._logger.info('Successfully launched!')

        # if global_settings.CONTROL_MODE == 'Manual':
        #     self.print_current_values()
        #     self.key_loop()
        # elif global_settings.CONTROL_MODE == 'Auto':
        #     self._logger.warning('Switch control mode from "Auto" to "Manual" first')
        # else:
        #     self._logger.error('Undefined control mode!')

        self.print_current_values()
        self.key_loop()

    def publisher_callback(self):
        self.message.speed = self.current_speed
        self.message.steering_angle = self.current_steering_angle
        self.cmd_ackermann_publisher.publish(self.message)

    def print_current_values(self):
        sys.stderr.write('\x1b[2J\x1b[H')

        self._logger.info(
            f'\n\033[34;1mSpeed: \033[32;1m{self.current_speed:.1f} Km/h '
            f'\033[34;1mAngle: \033[32;1m{self.current_steering_angle:.1f} Degrees\033[0m'
        )

        self.publisher_callback()

    def get_pressed_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)

        pressed_key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        return pressed_key

    def key_loop(self):
        self.settings = termios.tcgetattr(sys.stdin)

        try:
            while True:
                pressed_key = self.get_pressed_key()

                if pressed_key in control_keys_bindings:
                    if pressed_key == '\x20':  # Space
                        self.current_speed = 0.0
                    elif pressed_key == '\x09':  # Tab
                        self.current_steering_angle = 0.0
                    else:
                        self.current_speed += control_keys_bindings[pressed_key][1][0]
                        self.current_speed = clip(self.current_speed, *self.speed_range)

                        self.current_steering_angle += control_keys_bindings[pressed_key][1][1]
                        self.current_steering_angle = clip(self.current_steering_angle, *self.steering_angle_range)

                    self.print_current_values()
                elif pressed_key in ('\x03', '\x71'):  # Ctrl-C, Q
                    break
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))
        finally:
            sys.exit(0)


def main(args=None):
    try:
        rclpy.init(args=args)

        node = AckermannKeyboardTeleopNode()
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
