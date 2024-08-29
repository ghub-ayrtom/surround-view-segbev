import cv2
from ament_index_python.packages import get_package_share_directory
from PIL import Image
from rclpy.node import Node
import numpy as np
import os
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
import rclpy
import sensor_msgs.msg
import time
import traceback


PACKAGE_NAME = 'surround_view_segbev'
package_dir = get_package_share_directory(PACKAGE_NAME)


class SurroundViewNode(Node):
    def __init__(self):
        try:
            super().__init__('surround_view_node')
            self._logger.info('Successfully launched!')

            qos = qos_profile_sensor_data
            qos.reliability = QoSReliabilityPolicy.RELIABLE

            self.create_subscription(sensor_msgs.msg.Image, '/ego_vehicle/camera_front/image_color', self.__on_color_image_message, qos)
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __on_color_image_message(self, message):
        image_bytes = message.data
        image_bytes = np.frombuffer(image_bytes, dtype=np.uint8).reshape((message.height, message.width, 4))

        image = Image.fromarray(cv2.cvtColor(image_bytes, cv2.COLOR_RGBA2RGB))
        image = np.asarray(image)

        cv2.imwrite(os.path.join(package_dir, f'resource/images/{time.strftime("%Y%m%d-%H%M%S")}.png'), image)


def main(args=None):
    try:
        rclpy.init(args=args)
        surround_view_node = SurroundViewNode()
        rclpy.spin(surround_view_node)
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(''.join(traceback.TracebackException.from_exception(e).format()))
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
