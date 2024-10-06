from cv_bridge import CvBridge
import cv2
from configs import global_settings
from rclpy.node import Node
import numpy as np
import os
import rclpy
import sensor_msgs.msg
from message_filters import Subscriber, TimeSynchronizer
import time
import traceback


camera_topics = {
    # 'camera_front_left': '/ego_vehicle/camera_front_left/image_color',
    # 'camera_front_left_depth': '/ego_vehicle/camera_front_left_depth/image',

    'camera_front': '/ego_vehicle/camera_front/image_color',
    'camera_front_depth': '/ego_vehicle/camera_front_depth/image',

    # 'camera_front_right': '/ego_vehicle/camera_front_right/image_color',
    # 'camera_front_right_depth': '/ego_vehicle/camera_front_right_depth/image',

    'camera_rear_left': '/ego_vehicle/camera_rear_left/image_color',
    'camera_rear_left_depth': '/ego_vehicle/camera_rear_left_depth/image',

    'camera_rear': '/ego_vehicle/camera_rear/image_color',
    'camera_rear_depth': '/ego_vehicle/camera_rear_depth/image',

    'camera_rear_right': '/ego_vehicle/camera_rear_right/image_color',
    'camera_rear_right_depth': '/ego_vehicle/camera_rear_right_depth/image',
}


class SurroundViewNode(Node):

    def __init__(self):
        try:
            super().__init__('surround_view_node')
            self._logger.info('Successfully launched!')

            camera_topics_subscribers = []

            for camera_name in camera_topics.keys():
                subscriber = Subscriber(self, sensor_msgs.msg.Image, camera_topics[camera_name])
                subscriber.registerCallback(self.__on_color_image_message, camera_name)

                camera_topics_subscribers.append(subscriber)

            TimeSynchronizer(camera_topics_subscribers, 1).registerCallback(self.__on_color_image_message)
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __on_color_image_message(self, message, camera_name):
        if 'depth' not in camera_name:
            image_color = CvBridge().imgmsg_to_cv2(message, 'passthrough')
        else:
            # 32FC1 - один из типов кодировки "глубинных" изображений (32-битное число с плавающей запятой и одним каналом)
            image_distance = CvBridge().imgmsg_to_cv2(message, '32FC1')  # image_distance[height][width] для получения расстояния в метрах до конкретного пикселя изображения

            # Преобразуем неккоректные значения расстояния в чёрные пиксели (бесконечное расстояние)
            image_distance[np.isinf(image_distance)] = 0
            image_distance[np.isnan(image_distance)] = 0

            image_depth_normalized = cv2.normalize(image_distance, None, 0, 1, cv2.NORM_MINMAX)  # Преобразуем значение каждого пикселя изображения в диапазон от 0 до 1
            image_depth = (image_depth_normalized * 255).astype(np.uint8)  # Домножаем их на 255 для получения яркости и задаём тип - 8-битное изображение

        match camera_name:
            # case 'camera_front_left':
            #     cv2.imwrite(os.path.join(
            #         os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            #         f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front_left/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
            #     ), image_color)
            # case 'camera_front_left_depth':
            #     cv2.imwrite(os.path.join(
            #         os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            #         f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front_left/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
            #     ), image_depth)

            case 'camera_front':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_color)
            case 'camera_front_depth':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_depth)

            # case 'camera_front_right':
            #     cv2.imwrite(os.path.join(
            #         os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            #         f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front_right/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
            #     ), image_color)
            # case 'camera_front_right_depth':
            #     cv2.imwrite(os.path.join(
            #         os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            #         f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_front_right/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
            #     ), image_depth)

            case 'camera_rear_left':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear_left/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_color)
            case 'camera_rear_left_depth':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear_left/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_depth)

            case 'camera_rear':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_color)
            case 'camera_rear_depth':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_depth)

            case 'camera_rear_right':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear_right/color/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_color)
            case 'camera_rear_right_depth':
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/camera_rear_right/depth/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), image_depth)


def main(args=None):
    try:
        rclpy.init(args=args)

        surround_view_node = SurroundViewNode()
        rclpy.spin(surround_view_node)
        surround_view_node.destroy_node()

        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(''.join(traceback.TracebackException.from_exception(e).format()))
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
