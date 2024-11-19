from cv_bridge import CvBridge
import cv2
from rclpy.node import Node
import numpy as np
import rclpy
import sensor_msgs.msg
from message_filters import Subscriber, TimeSynchronizer
import traceback
from .scripts.CameraModel import CameraModel
from .scripts.BirdsEyeView import BirdsEyeView


camera_topics = {
    'camera_front_left': '/ego_vehicle/camera_front_left/image_color', 
    'camera_front_left_depth': '/ego_vehicle/camera_front_left_depth/image', 

    'camera_front': '/ego_vehicle/camera_front/image_color', 
    'camera_front_depth': '/ego_vehicle/camera_front_depth/image', 

    'camera_front_blind': '/ego_vehicle/camera_front_blind/image_color', 
    'camera_front_blind_depth': '/ego_vehicle/camera_front_blind_depth/image', 

    'camera_front_right': '/ego_vehicle/camera_front_right/image_color', 
    'camera_front_right_depth': '/ego_vehicle/camera_front_right_depth/image', 

    # 'camera_rear_left': '/ego_vehicle/camera_rear_left/image_color', 
    # 'camera_rear_left_depth': '/ego_vehicle/camera_rear_left_depth/image', 

    'camera_rear': '/ego_vehicle/camera_rear/image_color', 
    'camera_rear_depth': '/ego_vehicle/camera_rear_depth/image', 

    # 'camera_rear_right': '/ego_vehicle/camera_rear_right/image_color', 
    # 'camera_rear_right_depth': '/ego_vehicle/camera_rear_right_depth/image', 
}


class SurroundViewNode(Node):

    def __init__(self):
        try:
            super().__init__('surround_view_node')
            self._logger.info('Successfully launched!')

            self.camera_front_left = CameraModel('camera_front_left', self._logger)
            self.camera_front = CameraModel('camera_front', self._logger)
            self.camera_front_blind = CameraModel('camera_front_blind', self._logger)
            self.camera_front_right = CameraModel('camera_front_right', self._logger)

            # self.camera_rear_left = CameraModel('camera_rear_left', self._logger)
            self.camera_rear = CameraModel('camera_rear', self._logger)
            # self.camera_rear_right = CameraModel('camera_rear_right', self._logger)

            self.bev = BirdsEyeView(self._logger, load_weights_and_masks=True)
            self.surround_view_publisher = self.create_publisher(sensor_msgs.msg.Image, '/surround_view', 10)

            self.images_projected = {}
            camera_topics_subscribers = []

            for camera_name in camera_topics.keys():
                subscriber = Subscriber(self, sensor_msgs.msg.Image, camera_topics[camera_name])
                subscriber.registerCallback(self.__on_color_image_message, camera_name)

                camera_topics_subscribers.append(subscriber)

            TimeSynchronizer(camera_topics_subscribers, 1).registerCallback(self.__on_color_image_message)
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


    def __on_color_image_message(self, message, camera_name):
        if len(self.images_projected) < 5:  # 5 - количество используемых видеокамер
            if 'depth' not in camera_name:
                image_color = CvBridge().imgmsg_to_cv2(message, 'bgra8')
            else:
                # 32FC1 - один из типов кодировки "глубинных" изображений (32-битное число с плавающей запятой и одним каналом)
                image_distance = CvBridge().imgmsg_to_cv2(message, '32FC1')  # image_distance[height][width] для получения расстояния в метрах до конкретного пикселя изображения

                # Преобразуем неккоректные значения расстояния в чёрные пиксели (бесконечное расстояние)
                image_distance[np.isinf(image_distance)] = 0
                image_distance[np.isnan(image_distance)] = 0

                image_depth_normalized = cv2.normalize(image_distance, None, 0, 1, cv2.NORM_MINMAX)  # Преобразуем значение каждого пикселя изображения в диапазон от 0 до 1
                image_depth = (image_depth_normalized * 255).astype(np.uint8)  # Домножаем их на 255 для получения яркости и задаём тип - 8-битное изображение

            match camera_name:
                case 'camera_front_left':
                    if not self.camera_front_left.projection_matrix_received:
                        self.images_projected[camera_name] = self.camera_front_left.get_projection_matrix(image_color)
                        self.camera_front_left.projection_matrix_received = True
                case 'camera_front_left_depth':
                    pass

                case 'camera_front':
                    if not self.camera_front.projection_matrix_received:
                        self.images_projected[camera_name] = self.camera_front.get_projection_matrix(image_color)
                        self.camera_front.projection_matrix_received = True
                case 'camera_front_depth':
                    pass

                case 'camera_front_blind':
                    if not self.camera_front_blind.projection_matrix_received:
                        self.images_projected[camera_name] = self.camera_front_blind.get_projection_matrix(image_color)
                        self.camera_front_blind.projection_matrix_received = True
                case 'camera_front_blind_depth':
                    pass

                case 'camera_front_right':
                    if not self.camera_front_right.projection_matrix_received:
                        self.images_projected[camera_name] = self.camera_front_right.get_projection_matrix(image_color)
                        self.camera_front_right.projection_matrix_received = True
                case 'camera_front_right_depth':
                    pass

                # case 'camera_rear_left':
                #     if not self.camera_rear_left.projection_matrix_received:
                #         self.images_projected[camera_name] = self.camera_rear_left.get_projection_matrix(image_color)
                #         self.camera_rear_left.projection_matrix_received = True
                # case 'camera_rear_left_depth':
                #     pass

                case 'camera_rear':
                    if not self.camera_rear.projection_matrix_received:
                        self.images_projected[camera_name] = self.camera_rear.get_projection_matrix(image_color)
                        self.camera_rear.projection_matrix_received = True
                case 'camera_rear_depth':
                    pass

                # case 'camera_rear_right':
                #     if not self.camera_rear_right.projection_matrix_received:
                #         self.images_projected[camera_name] = self.camera_rear_right.get_projection_matrix(image_color)
                #         self.camera_rear_right.projection_matrix_received = True
                # case 'camera_rear_right_depth':
                #     pass
        else:
            self.bev.frames = self.images_projected

            # self.bev.luminance_balance()
            self.bev.stitch()
            self.bev.white_balance()
            self.bev.add_ego_vehicle()

            self.images_projected.clear()
            self.surround_view_publisher.publish(CvBridge().cv2_to_imgmsg(self.bev.image, 'rgb8'))

            self.camera_front_left.projection_matrix_received = False
            self.camera_front.projection_matrix_received = False
            self.camera_front_blind.projection_matrix_received = False
            self.camera_front_right.projection_matrix_received = False

            # self.camera_rear_left.projection_matrix_received = False
            self.camera_rear.projection_matrix_received = False
            # self.camera_rear_right.projection_matrix_received = False


def main(args=None):
    try:
        rclpy.init(args=args)

        node = SurroundViewNode()
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
