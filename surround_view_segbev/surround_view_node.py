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
import os
import time
from configs import global_settings, qos_profiles
import torch
from ultralytics import YOLO
from fastseg import MobileV3Large
from fastseg.image import colorize
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import yaml
from sensor_msgs.msg import LaserScan


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

            if global_settings.CONTROL_MODE == 'Auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.yolo_model = YOLO(YOLO_WEIGHTS_PATH).to(device)

                self.fastseg_model = MobileV3Large(num_classes=19).to(device)
                self.fastseg_model.load_state_dict(torch.load(FASTSEG_WEIGHTS_PATH, map_location=device))
                self.fastseg_model.eval()

            self.camera_front_left = CameraModel('camera_front_left', self._logger)
            self.camera_front = CameraModel('camera_front', self._logger)
            self.camera_front_blind = CameraModel('camera_front_blind', self._logger)
            self.camera_front_right = CameraModel('camera_front_right', self._logger)
            self.camera_rear = CameraModel('camera_rear', self._logger)

            self.bev = BirdsEyeView(self._logger, load_weights_and_masks=True)

            self.surround_view_publisher = self.create_publisher(sensor_msgs.msg.Image, '/surround_view', qos_profiles.image_qos)
            self.local_costmap_publisher = self.create_publisher(OccupancyGrid, '/local_costmap', qos_profiles.costmap_qos)
            self.laserscan_publisher = self.create_publisher(LaserScan, '/scan', qos_profiles.scan_qos)

            self.grid = OccupancyGrid()
            self.grid.header = Header()

            self.laserscan = LaserScan()
            self.laserscan.header = Header()

            self.__load_nav2_parameters()

            self.laserscan.angle_min = -np.pi  # Полный круговой обзор (360°)
            self.laserscan.angle_max = np.pi   #
            self.laserscan.angle_increment = np.pi / 360  # 0.5° на шаг

            self.laserscan.range_min = 0.2   # Метры
            self.laserscan.range_max = 20.0  #

            self.images_projected_with_obstacles_info = {}
            camera_topics_subscribers = []

            for camera_name in camera_topics.keys():
                subscriber = Subscriber(self, sensor_msgs.msg.Image, camera_topics[camera_name])
                subscriber.registerCallback(self.__on_color_image_message, camera_name)

                camera_topics_subscribers.append(subscriber)

            TimeSynchronizer(camera_topics_subscribers, 2).registerCallback(self.__on_color_image_message)

            self._logger.info('Successfully launched!')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __load_nav2_parameters(self):
        with open(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            'configs/nav2_params.yaml'
        )) as nav2_params_yaml_file:
            try:
                nav2_parameters = yaml.safe_load(nav2_params_yaml_file)
                self.grid.header.frame_id = nav2_parameters['local_costmap']['local_costmap']['ros__parameters']['global_frame']
                self.laserscan.header.frame_id = nav2_parameters['local_costmap']['local_costmap']['ros__parameters']['robot_base_frame']

                self.grid.info.width = nav2_parameters['local_costmap']['local_costmap']['ros__parameters']['width']
                self.grid.info.height = nav2_parameters['local_costmap']['local_costmap']['ros__parameters']['height']
                self.grid.info.resolution = nav2_parameters['local_costmap']['local_costmap']['ros__parameters']['resolution']

                self.grid.info.origin.position.x = -self.grid.info.height * self.grid.info.resolution / 2.0
                self.grid.info.origin.position.y = self.grid.info.width * self.grid.info.resolution / 2.0

                self.grid.info.origin.orientation.z = -1.0
                self.grid.info.origin.orientation.w = 1.0

                nav2_params_yaml_file.close()
            except yaml.YAMLError as e:
                self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __on_color_image_message(self, message, camera_name):
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
                if global_settings.USED_SEGMENTOR_FOLDER_NAME == 'FastSeg' and global_settings.CONTROL_MODE == 'Auto':
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
                    if not self.camera_front_left.projection_matrix_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_left.get_projection_matrix(image_color, obstacle_bboxes=predicted_bboxes)
                        self.camera_front_left.projection_matrix_received = True
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
                    if not self.camera_front.projection_matrix_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front.get_projection_matrix(image_color, obstacle_bboxes=predicted_bboxes)
                        self.camera_front.projection_matrix_received = True
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
                    if not self.camera_front_blind.projection_matrix_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_blind.get_projection_matrix(image_color, obstacle_bboxes=predicted_bboxes)
                        self.camera_front_blind.projection_matrix_received = True
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
                    if not self.camera_front_right.projection_matrix_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_front_right.get_projection_matrix(image_color, obstacle_bboxes=predicted_bboxes)
                        self.camera_front_right.projection_matrix_received = True
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
                    if not self.camera_rear.projection_matrix_received:
                        self.images_projected_with_obstacles_info[camera_name] = self.camera_rear.get_projection_matrix(image_color, obstacle_bboxes=predicted_bboxes)
                        self.camera_rear.projection_matrix_received = True
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

            local_costmap = self.bev.add_ego_vehicle_and_track_obstacles()
            local_costmap = cv2.resize(local_costmap, (self.grid.info.width, self.grid.info.height), interpolation=cv2.INTER_NEAREST)

            self.grid.header.stamp = self.get_clock().now().to_msg()
            self.grid.data = np.flipud(local_costmap).flatten().tolist()

            self.laserscan.header.stamp = self.get_clock().now().to_msg()
            self.laserscan.ranges = [self.laserscan.range_max] * int(
                (self.laserscan.angle_max - self.laserscan.angle_min) / self.laserscan.angle_increment)

            for x in range(self.grid.info.height):
                for y in range(self.grid.info.width):
                    i = x * self.grid.info.width + y

                    # Если ячейка локальной карты стоимости содержит препятствие
                    if self.grid.data[i] < 0:
                        cx = self.grid.info.origin.position.x + x * self.grid.info.resolution  # Находим её координаты
                        cy = self.grid.info.origin.position.y - y * self.grid.info.resolution  #

                        angle = np.arctan2(cy, cx)
                        distance = np.hypot(cx, cy)

                        j = int((angle - self.laserscan.angle_min) / self.laserscan.angle_increment)

                        if 0 <= j < len(self.laserscan.ranges):
                            self.laserscan.ranges[j] = min(self.laserscan.ranges[j], distance)

            self.local_costmap_publisher.publish(self.grid)
            # self.laserscan_publisher.publish(self.laserscan)

            self.images_projected_with_obstacles_info.clear()
            self.surround_view_publisher.publish(CvBridge().cv2_to_imgmsg(self.bev.image, 'rgb8'))

            if self.collect_models_training_data:
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/surround_view/data/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), cv2.cvtColor(self.bev.image, cv2.COLOR_BGR2RGB))

            self.camera_front_left.projection_matrix_received = False
            self.camera_front.projection_matrix_received = False
            self.camera_front_blind.projection_matrix_received = False
            self.camera_front_right.projection_matrix_received = False
            self.camera_rear.projection_matrix_received = False


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
