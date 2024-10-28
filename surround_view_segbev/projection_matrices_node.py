from cv_bridge import CvBridge
from rclpy.node import Node
import rclpy
import sensor_msgs.msg
from message_filters import Subscriber, TimeSynchronizer
import traceback
import cv2
import numpy as np
from .scripts.simple_gui import PointSelector, display_image


camera_topics = {
    'camera_front_left': '/ego_vehicle/camera_front_left/image_color', 
    'camera_front': '/ego_vehicle/camera_front/image_color', 
    'camera_front_right': '/ego_vehicle/camera_front_right/image_color', 
    # 'camera_rear_left': '/ego_vehicle/camera_rear_left/image_color', 
    'camera_rear': '/ego_vehicle/camera_rear/image_color', 
    # 'camera_rear_right': '/ego_vehicle/camera_rear_right/image_color', 
}


class ProjectionMatricesNode(Node):

    def __init__(self):
        try:
            super().__init__('projection_matrices_node')
            self._logger.info('Successfully launched!')

            camera_topics_subscribers = []

            for camera_name in camera_topics.keys():
                subscriber = Subscriber(self, sensor_msgs.msg.Image, camera_topics[camera_name])
                subscriber.registerCallback(self.__on_color_image_message, camera_name)

                camera_topics_subscribers.append(subscriber)

            TimeSynchronizer(camera_topics_subscribers, 1).registerCallback(self.__on_color_image_message)
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __get_projection_matrix(self, image, camera_name):
        gui = PointSelector(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), title=camera_name)
        choice = gui.loop()

        dst_points = {
            'camera_front_left': [(300, 200), (680, 200), (300, 360), (680, 360)], 
            'camera_front': [(300, 200), (680, 200), (300, 360), (680, 360)], 
            'camera_front_right': [(300, 200), (680, 200), (300, 360), (680, 360)], 
            # 'camera_rear_left': [(300, 200), (680, 200), (300, 360), (680, 360)], 
            'camera_rear': [(300, 200), (680, 200), (300, 360), (680, 360)], 
            # 'camera_rear_right': [(300, 200), (680, 200), (300, 360), (680, 360)], 
        }

        if choice > 0:
            src = np.float32(gui.keypoints)
            dst = np.float32(dst_points[camera_name])

            projection_matrix = cv2.getPerspectiveTransform(src, dst)
            image_projected = cv2.warpPerspective(image, projection_matrix, (1000, 405))

            display_image("Bird's Eye View", image_projected)
            cv2.destroyAllWindows()


    def __on_color_image_message(self, message, camera_name):
        image_color = CvBridge().imgmsg_to_cv2(message, 'passthrough')

        match camera_name:
            case 'camera_front_left':
                self.__get_projection_matrix(image_color, camera_name)

            case 'camera_front':
                self.__get_projection_matrix(image_color, camera_name)

            case 'camera_front_right':
                self.__get_projection_matrix(image_color, camera_name)

            # case 'camera_rear_left':
            #     self.__get_projection_matrix(image_color, camera_name)

            case 'camera_rear':
                self.__get_projection_matrix(image_color, camera_name)

            # case 'camera_rear_right':
            #     self.__get_projection_matrix(image_color, camera_name)


def main(args=None):
    try:
        rclpy.init(args=args)

        node = ProjectionMatricesNode()
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
