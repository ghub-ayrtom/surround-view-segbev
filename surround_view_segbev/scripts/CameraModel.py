import cv2
from surround_view_segbev.configs import global_settings
import numpy as np
import os
import time
from controller import Supervisor
import traceback
import yaml


if os.getenv('USING_EXTERN_CONTROLLER') is None:
    supervisor = Supervisor()

CALIBRATION_IMAGES_COUNT = 50
CHESSBOARD_PATTERN_SIZE = (7, 7)  # Размерность внутренних пересечений (паттерна) шахматной доски


class CameraModel:
    calibration_flags = (
        # cv2.CALIB_USE_INTRINSIC_GUESS +  # Использовать заданные внутренние параметры (fx, fy, cx, cy), а также коэффициенты радиальной и тангенциальной дисторсий в качестве начальных и оптимизировать их в процессе калибровки
        # cv2.CALIB_USE_EXTRINSIC_GUESS +  # Использовать заданные внешние параметры (rotation_vectors и translation_vectors) в качестве начальных и оптимизировать их в процессе калибровки
        cv2.CALIB_FIX_PRINCIPAL_POINT +  # Не изменять оптический центр линзы во время глобальной оптимизации
        cv2.CALIB_FIX_ASPECT_RATIO +  # Зафиксировать соотношение сторон фокусных расстояний по осям X и Y (пиксели видеокамеры квадратные)
        # cv2.CALIB_ZERO_TANGENT_DIST +  # Обнулить коэффициенты p1 и p2 тангенциальной дисторсии и не пытаться подобрать их во время оптимизации
        # cv2.CALIB_FIX_FOCAL_LENGTH +  # Зафиксировать заранее известное значение фокусного расстояния (focal length). Требуется установка флага cv2.CALIB_USE_INTRINSIC_GUESS
        cv2.CALIB_FIX_K3  # Зафиксировать соответствующий коэффициент радиальной дисторсии и не пытаться подобрать его во время оптимизации
        # cv2.CALIB_RATIONAL_MODEL +  # Использовать рациональную модель калибровки, которая учитывает коэффициенты k4, k5 и k6, а также возвращает 8 или более коэффициентов радильной дисторсии
        # cv2.CALIB_THIN_PRISM_MODEL +  # Использовать модель калибровки тонкой призмы, которая учитывает коэффициенты s1, s2, s3 и s4, а также возвращает 12 или более коэффициентов призменной дисторсии
        # cv2.CALIB_FIX_S1_S2_S3_S4 +  # Зафиксировать коэффициенты призменной дисторсии и не пытаться подобрать их во время оптимизации
        # cv2.CALIB_TILTED_MODEL +  # Использовать модель калибровки наклонного датчика, которая учитывает коэффициенты tauX и tauY, а также возвращает 14 коэффициентов наклонной дисторсии
        # cv2.CALIB_FIX_TAUX_TAUY +  # Зафиксировать коэффициенты наклонной дисторсии и не пытаться подобрать их во время оптимизации
        # cv2.CALIB_USE_QR +  # Использовать QR-декомпозицию вместо SVD-декомпозиции. Быстрее, но потенциально менее точно
        # cv2.CALIB_FIX_TANGENT_DIST +  # Зафиксировать коэффициенты тангенциальной дисторсии и не пытаться подобрать их во время оптимизации
        # cv2.CALIB_FIX_INTRINSIC +  # Зафиксировать внутренние параметры видеокамеры и не пытаться подобрать их во время оптимизации
        # cv2.CALIB_SAME_FOCAL_LENGTH +  # Обе камеры имеют одинаковое значение фокусного расстояния по осям X и Y
        # cv2.CALIB_ZERO_DISPARITY +  # Оптические центры линз обеих видеокамер имеют одинаковые координаты пикселей в выпрямленных видах
        # cv2.CALIB_USE_LU  # Использовать LU-декомпозицию вместо SVD-декомпозиции. Гораздо быстрее, но потенциально менее точно
    )

    def __init__(self, webots_camera_name, node_logger, related_chessboard=None, load_parameters=True):
        if os.getenv('USING_EXTERN_CONTROLLER') is None:
            self.device = supervisor.getDevice(webots_camera_name)

            if self.device is None:
                node_logger.error('The Webots camera device with the specified name was not found!')
            else:
                self.device.enable(100)  # Отправлять изображения каждые 100 мс (10 Гц)

        self.device_name = webots_camera_name

        self.load_parameters = load_parameters
        self.calibrating = False

        self.optical_characteristics = None
        self.calibration_images_count = CALIBRATION_IMAGES_COUNT

        self.node_logger = node_logger
        self.chessboard = related_chessboard

        self.parameters_loaded = False
        self.projected_image_with_obstacles_info_received = False

        ### Параметры камеры

        self.image_shape = [0, 0, 0]

        # Матрица внутренних (intrinsics) параметров камеры, таких как: fx и fy - фокусное расстояние в пикселях по соответствующим осям, 
        # cx и cy - координаты главной точки (principal point), которая обычно находится в центре изображения
        self.K = np.zeros((3, 3))
        # Вектор коэффициентов дисторсии, которая зачастую вызвана радиальными и тангенциальными искажениями из-за линзы объектива видеокамеры
        self.D = np.zeros((5, 1))

        self.projection_matrix = np.zeros((3, 3))

        # Векторы (ось) вращения для каждого из калибровочных изображений, длина которых - это значение угла поворота камеры по одной из трёх осей относительно мировых координат
        self.rotation_vectors = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(CALIBRATION_IMAGES_COUNT)]  # Ориентация в пространстве относительно сцены
        # Векторы перемещения для каждого из калибровочных изображений, длина которых - это значение смещения камеры по одной из трёх осей относительно центра мировых координат
        self.translation_vectors = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(CALIBRATION_IMAGES_COUNT)]  # Положение в пространстве относительно сцены

        self.projection_shape = ()  # Разрешение BEV-изображения для текущей видеокамеры

        self.load_camera_parameters()

    def load_camera_parameters(self):
        camera_parameters_file_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))), os.pardir), 
            f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml', 
        )

        if self.load_parameters and os.path.isfile(camera_parameters_file_path):
            try:
                fs = cv2.FileStorage(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__))), os.pardir), 
                    f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml', 
                ), cv2.FILE_STORAGE_READ)

                if fs.isOpened():
                    self.image_shape = np.array(fs.getNode('image_resolution').mat(), dtype=int).flatten()
                    self.K = np.array(fs.getNode('camera_matrix').mat())
                    self.D = np.array(fs.getNode('distortion_coefficients').mat())
                    self.projection_matrix = np.array(fs.getNode('projection_matrix').mat())

                    fs.release()
            except Exception as e:
                self.node_logger.error(''.join(traceback.TracebackException.from_exception(e).format()))
        else:
            with open(os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/webots_settings.yaml'
            )) as webots_settings_yaml_file:
                try:
                    self.optical_characteristics = yaml.safe_load(webots_settings_yaml_file)
                    self.image_shape = (self.optical_characteristics['image_height'], self.optical_characteristics['image_width'], 4)
                    webots_settings_yaml_file.close()
                except yaml.YAMLError as e:
                    self.node_logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

        with open(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            'scripts/BEVFormer/bev_parameters.yaml', 
        )) as bev_parameters_yaml:
            try:
                bev_parameters = yaml.safe_load(bev_parameters_yaml)

                near_shift_width = bev_parameters['near_shift_width']
                near_shift_height = bev_parameters['near_shift_height']

                far_shift_width = bev_parameters['far_shift_width']
                far_shift_height = bev_parameters['far_shift_height']

                total_width = bev_parameters['total_width_base'] + 2 * far_shift_width
                total_height = bev_parameters['total_height_base'] + 2 * far_shift_height

                vehicle_leftside_edges_x = far_shift_width + bev_parameters['vehicle_leftside_edges_x_inc'] + near_shift_width
                vehicle_topside_edges_y = far_shift_height + bev_parameters['vehicle_topside_edges_y_inc'] + near_shift_height

                match self.device_name:
                    case 'camera_front_left':
                        self.projection_shape = (total_height, vehicle_leftside_edges_x)
                    case 'camera_front':
                        self.projection_shape = (total_width, vehicle_topside_edges_y)
                    case 'camera_front_blind':
                        self.projection_shape = (total_width, vehicle_topside_edges_y)
                    case 'camera_front_right':
                        self.projection_shape = (total_height, vehicle_leftside_edges_x)
                    case 'camera_rear':
                        self.projection_shape = (total_width, vehicle_topside_edges_y)

                bev_parameters_yaml.close()
            except yaml.YAMLError as e:
                print(e)
        
        self.parameters_loaded = True

    def calibrate_camera(self, calibration_image, debug=False):
        calibration_image_gray = cv2.cvtColor(calibration_image, cv2.COLOR_RGBA2GRAY)

        if not self.calibrating:
            self.calibrating = True

            if self.optical_characteristics is None:
                with open(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/webots_settings.yaml', 
                )) as webots_settings_yaml_file:
                    try:
                        self.optical_characteristics = yaml.safe_load(webots_settings_yaml_file)
                        webots_settings_yaml_file.close()
                    except yaml.YAMLError as e:
                        self.node_logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

            self.object_points_3D = []  # Список обнаруженных углов шахматной доски в 3D-пространстве (реальный мир)
            self.image_points_2D = []  # Список обнаруженных углов шахматной доски в 2D-пространстве (плоскость изображения)

            self.K[0][0] = self.optical_characteristics['focal_length']  # fx
            self.K[0][1] = 0.0
            self.K[0][2] = self.optical_characteristics['image_width'] * self.optical_characteristics['distortion_center'][0]  # cx
            self.K[1][0] = 0.0
            self.K[1][1] = self.optical_characteristics['focal_length']  # fy
            self.K[1][2] = self.optical_characteristics['image_height'] * self.optical_characteristics['distortion_center'][1]  # cy
            self.K[2][0] = 0.0
            self.K[2][1] = 0.0
            self.K[2][2] = 1.0

            self.D[0] = self.optical_characteristics['radial_distortion'][0]  # k1
            self.D[1] = self.optical_characteristics['radial_distortion'][1]  # k2
            self.D[2] = self.optical_characteristics['tangential_distortion'][0]  # p1
            self.D[3] = self.optical_characteristics['tangential_distortion'][1]  # p2
            self.D[4] = 0.0  # k3

        if self.calibration_images_count > 0:
            self.node_logger.info(f'[{self.device_name}] Searching for a chessboard pattern corners...')

            found, corners = cv2.findChessboardCorners(calibration_image_gray, CHESSBOARD_PATTERN_SIZE)

            if found:
                self.calibration_images_count -= 1
                self.node_logger.info(f'[{self.device_name}] The corners was found, {self.calibration_images_count} calibration images remained\n')

                # 0.1 - порог точности в пикселях +/ИЛИ 30 - максимальное количество итераций
                term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                # 5x5 пикселей - размер окна внутри которого будут уточняться координаты найденных углов шахматной доски с субпиксельной точностью
                cv2.cornerSubPix(calibration_image_gray, corners, (5, 5), (-1, -1), term_criteria)  # (-1, -1) - уточнение будет производиться по всему окну поиска (без так называемой "мёртвой зоны")

                self.object_points_3D.append(self.chessboard.pattern_points_3D)
                self.image_points_2D.append(corners.reshape(-1, 2))  # Убираем дополнительное измерение (массив), возникшее от того, что библиотека OpenCV использует трёхмерные структуры данных

                if debug:
                    calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_RGBA2RGB)
                    cv2.drawChessboardCorners(calibration_image, CHESSBOARD_PATTERN_SIZE, corners, found)
                    cv2.imwrite(os.path.join(
                        os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                        f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{self.device_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
                    ), calibration_image)
        elif self.calibration_images_count == 0:
            self.calibration_images_count -= 1
            self.node_logger.info(f'[{self.device_name}] Calculation of camera parameters...')

            image_points_2D_array = np.asarray(self.image_points_2D)
            object_points_3D_array = np.asarray(self.object_points_3D)

            # Добавляем дополнительное измерение (оборачиваем массив массивов в очередной массив) 
            # для соответствия формату принимаемого функцией calibrateCamera аргумента
            image_points_2D_expanded = np.expand_dims(image_points_2D_array, -2)
            object_points_3D_expanded = np.expand_dims(object_points_3D_array, -2)

            _, self.K, self.D, self.rotation_vectors, self.translation_vectors = cv2.calibrateCamera(
                object_points_3D_expanded, 
                image_points_2D_expanded, 
                calibration_image_gray.shape[::-1], 
                self.K, 
                self.D, 
                self.rotation_vectors, 
                self.translation_vectors, 
                CameraModel.calibration_flags
            )

            if debug:
                calibration_image_undistorted = self.undistort(calibration_image)
                
                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{self.device_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), calibration_image_undistorted)

            camera_parameters_file = cv2.FileStorage(os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml'
            ), cv2.FILE_STORAGE_WRITE)

            if camera_parameters_file.isOpened():
                camera_parameters_file.write('image_resolution', np.int32([calibration_image.shape[0], calibration_image.shape[1], 4]))
                camera_parameters_file.write('camera_matrix', self.K)
                camera_parameters_file.write('distortion_coefficients', self.D)

                camera_parameters_file.release()

            mean_error = 0

            for i in range(len(self.object_points_3D)):
                # Преобразуем каждую точку объекта в соответствующую ей точку на изображении
                image_points, _ = cv2.projectPoints(self.object_points_3D[i], self.rotation_vectors[i], self.translation_vectors[i], self.K, self.D)
                image_points = image_points.reshape(-1, 2)  # Приводим к формату N x 2 для соответствия self.image_points_2D[i]

                error = cv2.norm(self.image_points_2D[i], image_points, cv2.NORM_L2) / len(image_points)  # Расчитываем абсолютную норму
                mean_error += error

            self.node_logger.info(f'[{self.device_name}] Successfully saved on the path "../configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml"!\n')
            self.node_logger.info(f'[{self.device_name}] Re-projection Error: {mean_error / len(self.object_points_3D)}\n')

            self.calibrating = False

    def undistort(self, image):
        image_height, image_width = image.shape[:2]

        self_K_new, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (image_width, image_height), 1, (image_width, image_height))
        map_x, map_y = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), self_K_new, (image_width, image_height), cv2.CV_32FC1)

        # Применяем карты выравнивания для устранения искажений и создания ровного изображения
        image_undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        x, y, w, h = roi  # Обрезаем искажённые края исправленного изображения
        image_undistorted = image_undistorted[y:y+h, x:x+w]  #

        return image_undistorted

    def flip(self, image):
        match self.device_name:
            case 'camera_front_left':
                return cv2.transpose(image)[::-1]
            case 'camera_front' | 'camera_front_blind':
                return image.copy()
            case 'camera_front_right':
                return np.flip(cv2.transpose(image), 1)
            case 'camera_rear':
                return image.copy()[::-1, ::-1, :]

    def get_projected_image_with_obstacles_info(self, image, obstacle_bboxes=None, gotten=True):
        image_undistorted = image  # self.undistort(image)
        image_projected = cv2.warpPerspective(image_undistorted, self.projection_matrix, self.projection_shape)

        obstacle_corners = []
        obstacle_centers = []
        obstacle_distances_m = []

        if obstacle_bboxes is not None:
            for obstacle in obstacle_bboxes:
                for bbox in obstacle.boxes.data:
                    x1, y1, x2, y2, _, cls = bbox.cpu().numpy()

                    if cls == 0:  # plastic_barrel
                        # Однородные координаты углов ограничивающей рамки
                        bbox_corners_homogeneous = np.array([
                            [x1, y1, 1],  # Верхний левый
                            [x2, y1, 1],  # Верхний правый
                            [x2, y2, 1],  # Нижний правый
                            [x1, y2, 1],  # Нижний левый
                        ])

                        # Однородные координаты центра ограничивающей рамки
                        bbox_center_homogeneous = np.array([int((x1 + x2) / 2), int((y1 + y2) / 2), 1])

                        bbox_corners_warped = np.dot(self.projection_matrix, bbox_corners_homogeneous.T).T
                        bbox_corners_warped = bbox_corners_warped[:, :2] / bbox_corners_warped[:, 2:3]  # Нормализация

                        bbox_center_warped = np.dot(self.projection_matrix, bbox_center_homogeneous)
                        bbox_center_warped /= bbox_center_warped[2]  #

                        bbox_corners = [x1, y1, x2, y2, np.round(bbox_corners_warped[:, :2]).astype(int).tolist()]
                        bbox_center = [int((x1 + x2) / 2), int((y1 + y2) / 2), int(bbox_center_warped[0]), int(bbox_center_warped[1])]

                        obstacle_corners.append(bbox_corners)
                        obstacle_centers.append(bbox_center)

        return self.flip(cv2.cvtColor(image_projected, cv2.COLOR_RGB2BGR)), obstacle_corners, obstacle_centers, obstacle_distances_m
