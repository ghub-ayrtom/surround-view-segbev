#!/usr/bin/env python3

from cv_bridge import CvBridge
import cv2
from rclpy.node import Node
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
import traceback
from surround_view_segbev.scripts.CameraModel import CameraModel
from surround_view_segbev.scripts.BirdsEyeView import BirdsEyeView
import os
import time
from surround_view_segbev.configs import global_settings, qos_profiles
import torch
from ultralytics import YOLO
from fastseg import MobileV3Large
from fastseg.image import colorize


camera_topics = {
    'camera_front_left': '/ego_vehicle/camera_front_left/image_color', 
    'camera_front_left_depth': '/ego_vehicle/camera_front_left_depth/image', 

    'camera_front': '/ego_vehicle/camera_front/image_color', 
    'camera_front_depth': '/ego_vehicle/camera_front_depth/image', 

    'camera_front_blind': '/ego_vehicle/camera_front_blind/image_color', 
    'camera_front_blind_depth': '/ego_vehicle/camera_front_blind_depth/image', 

    'camera_front_right': '/ego_vehicle/camera_front_right/image_color', 
    'camera_front_right_depth': '/ego_vehicle/camera_front_right_depth/image', 

    'camera_rear': '/ego_vehicle/camera_rear/image_color', 
    'camera_rear_depth': '/ego_vehicle/camera_rear_depth/image', 
}

YOLO_WEIGHTS_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
    f'models/{global_settings.USED_DETECTOR_FOLDER_NAME}/model.pt'
)

FASTSEG_WEIGHTS_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
    f'models/{global_settings.USED_SEGMENTOR_FOLDER_NAME}/model.pth'
)


class SurroundViewNode(Node):
    def __init__(self):
        try:
            super().__init__('surround_view_node')

            self.collect_models_training_data = False

            if global_settings.EGO_VEHICLE_CONTROL_MODE == 'Auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.yolo_model = YOLO(YOLO_WEIGHTS_PATH).to(device)

                self.fastseg_model = MobileV3Large(num_classes=19).to(device)
                self.fastseg_model.load_state_dict(torch.load(FASTSEG_WEIGHTS_PATH, map_location=device))
                self.fastseg_model.eval()

            self.camera_front_left = CameraModel('camera_front_left', self.get_logger())
            self.camera_front = CameraModel('camera_front', self.get_logger())
            self.camera_front_blind = CameraModel('camera_front_blind', self.get_logger())
            self.camera_front_right = CameraModel('camera_front_right', self.get_logger())
            self.camera_rear = CameraModel('camera_rear', self.get_logger())

            self.bev = BirdsEyeView(self.get_logger(), load_weights_and_masks=True)
            self.surround_view_publisher = self.create_publisher(Image, '/surround_view', qos_profiles.image_qos)

            self.images_projected_with_obstacles_info = {}
            camera_topics_subscribers = []

            for camera_name in camera_topics.keys():
                subscriber = Subscriber(self, Image, camera_topics[camera_name])
                subscriber.registerCallback(self.__on_color_image_message, camera_name)
                camera_topics_subscribers.append(subscriber)

            TimeSynchronizer(camera_topics_subscribers, queue_size=1).registerCallback(self.__on_color_image_message)
            # ApproximateTimeSynchronizer(camera_topics_subscribers, queue_size=1, slop=0.05).registerCallback(self.__on_color_image_message)

            self.get_logger().info('Successfully launched!')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __on_color_image_message(self, message, camera_name):   # (self, message_1, message_2, ..., message_10) для ApproximateTimeSynchronizer
        if len(self.images_projected_with_obstacles_info) < 5:  # 5 - количество используемых видеокамер
            predicted_bboxes = None

            if 'depth' not in camera_name:
                image_color = CvBridge().imgmsg_to_cv2(message, 'rgb8')

                # if global_settings.USED_DETECTOR_FOLDER_NAME == 'YOLO11' and global_settings.CONTROL_MODE == 'Auto':
                #     predicted_bboxes = self.yolo_model.predict(
                #         source=cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB), 
                #         conf=0.9, 
                #         verbose=False, 
                #         # show=True, 
                #     )
                if global_settings.USED_SEGMENTOR_FOLDER_NAME == 'FastSeg' and global_settings.EGO_VEHICLE_CONTROL_MODE == 'Auto':
                    predicted_labels = self.fastseg_model.predict_one(image_color)
                    image_color = np.array(colorize(predicted_labels, palette='surround_view_segbev'))
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
                    if self.collect_models_training_data:
                        cv2.imwrite(os.path.join(
                            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                        ), image_color)
                    if self.camera_front_left.parameters_loaded and not self.camera_front_left.projected_image_with_obstacles_info_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_left.get_projected_image_with_obstacles_info(
                            image_color, 
                            obstacle_bboxes=predicted_bboxes, 
                        )
                        self.camera_front_left.projected_image_with_obstacles_info_received = True
                case 'camera_front_left_depth':
                    if camera_name[:-6] in self.images_projected_with_obstacles_info.keys():
                        image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m = self.images_projected_with_obstacles_info[camera_name[:-6]]

                        for obstacle_center in obstacle_centers:
                            x, y = obstacle_center[0], obstacle_center[1]
                            obstacle_distances_m.append(image_distance[y][x])
                        
                        self.images_projected_with_obstacles_info[camera_name[:-6]] = image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m

                case 'camera_front':
                    if self.collect_models_training_data:
                        cv2.imwrite(os.path.join(
                            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                        ), image_color)
                    if self.camera_front.parameters_loaded and not self.camera_front.projected_image_with_obstacles_info_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front.get_projected_image_with_obstacles_info(
                            image_color, 
                            obstacle_bboxes=predicted_bboxes, 
                        )
                        self.camera_front.projected_image_with_obstacles_info_received = True
                case 'camera_front_depth':
                    if camera_name[:-6] in self.images_projected_with_obstacles_info.keys():
                        image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m = self.images_projected_with_obstacles_info[camera_name[:-6]]

                        for obstacle_center in obstacle_centers:
                            x, y = obstacle_center[0], obstacle_center[1]
                            obstacle_distances_m.append(image_distance[y][x])
                        
                        self.images_projected_with_obstacles_info[camera_name[:-6]] = image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m

                case 'camera_front_blind':
                    if self.collect_models_training_data:
                        cv2.imwrite(os.path.join(
                            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                        ), image_color)
                    if self.camera_front_blind.parameters_loaded and not self.camera_front_blind.projected_image_with_obstacles_info_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_blind.get_projected_image_with_obstacles_info(
                            image_color, 
                            obstacle_bboxes=predicted_bboxes, 
                        )
                        self.camera_front_blind.projected_image_with_obstacles_info_received = True
                case 'camera_front_blind_depth':
                    if camera_name[:-6] in self.images_projected_with_obstacles_info.keys():
                        image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m = self.images_projected_with_obstacles_info[camera_name[:-6]]

                        for obstacle_center in obstacle_centers:
                            x, y = obstacle_center[0], obstacle_center[1]
                            obstacle_distances_m.append(image_distance[y][x])
                        
                        self.images_projected_with_obstacles_info[camera_name[:-6]] = image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m

                case 'camera_front_right':
                    if self.collect_models_training_data:
                        cv2.imwrite(os.path.join(
                            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                        ), image_color)
                    if self.camera_front_right.parameters_loaded and not self.camera_front_right.projected_image_with_obstacles_info_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_right.get_projected_image_with_obstacles_info(
                            image_color, 
                            obstacle_bboxes=predicted_bboxes, 
                        )
                        self.camera_front_right.projected_image_with_obstacles_info_received = True
                case 'camera_front_right_depth':
                    if camera_name[:-6] in self.images_projected_with_obstacles_info.keys():
                        image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m = self.images_projected_with_obstacles_info[camera_name[:-6]]

                        for obstacle_center in obstacle_centers:
                            x, y = obstacle_center[0], obstacle_center[1]
                            obstacle_distances_m.append(image_distance[y][x])
                        
                        self.images_projected_with_obstacles_info[camera_name[:-6]] = image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m

                case 'camera_rear':
                    if self.collect_models_training_data:
                        cv2.imwrite(os.path.join(
                            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                        ), image_color)
                    if self.camera_rear.parameters_loaded and not self.camera_rear.projected_image_with_obstacles_info_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_rear.get_projected_image_with_obstacles_info(
                            image_color, 
                            obstacle_bboxes=predicted_bboxes, 
                        )
                        self.camera_rear.projected_image_with_obstacles_info_received = True
                case 'camera_rear_depth':
                    if camera_name[:-6] in self.images_projected_with_obstacles_info.keys():
                        image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m = self.images_projected_with_obstacles_info[camera_name[:-6]]

                        for obstacle_center in obstacle_centers:
                            x, y = obstacle_center[0], obstacle_center[1]
                            obstacle_distances_m.append(image_distance[y][x])
                        
                        self.images_projected_with_obstacles_info[camera_name[:-6]] = image_color_projected, obstacle_corners, obstacle_centers, obstacle_distances_m
        else:
            self.bev.frames = self.images_projected_with_obstacles_info

            # self.bev.luminance_balance()
            self.bev.stitch()
            self.bev.white_balance()

            # if global_settings.USED_SEGMENTOR_FOLDER_NAME == 'FastSegBEV' and global_settings.CONTROL_MODE == 'Auto':
            #     self.bev.image = cv2.cvtColor(self.bev.image, cv2.COLOR_BGR2RGB)
            #     predicted_labels = self.fastseg_model.predict_one(self.bev.image)
            #     self.bev.image = np.array(colorize(predicted_labels, palette='surround_view_segbev'))

            self.bev.add_ego_vehicle_and_track_obstacles()

            self.images_projected_with_obstacles_info.clear()
            self.surround_view_publisher.publish(CvBridge().cv2_to_imgmsg(self.bev.image, 'rgb8'))

            if self.collect_models_training_data:
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/surround_view/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), cv2.cvtColor(self.bev.image, cv2.COLOR_BGR2RGB))

            self.camera_front_left.projected_image_with_obstacles_info_received = False
            self.camera_front.projected_image_with_obstacles_info_received = False
            self.camera_front_blind.projected_image_with_obstacles_info_received = False
            self.camera_front_right.projected_image_with_obstacles_info_received = False
            self.camera_rear.projected_image_with_obstacles_info_received = False


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
