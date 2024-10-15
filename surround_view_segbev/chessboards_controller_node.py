import cv2
from configs import global_settings
from rclpy.node import Node
import numpy as np
import os
import rclpy
from controller import Supervisor
import time
import traceback
import yaml


TIME_STEP = 32
CALIBRATION_IMAGES_COUNT = 50

CHESSBOARD_SQUARE_SIZE = 2.5  # Размеры квадрата шахматной доски в сантиметрах
CHESSBOARD_PATTERN_SIZE = (7, 7)  # Размерность внутренних пересечений (паттерна) шахматной доски
CHESSBOARDS_MOVEMENT_SENSITIVITY = 0.1  # Минимальное значение изменения координат траектории движения шахматных досок в Webots (одинаковое для всех)

supervisor = None
node_logger = None

cameras_webots_settings = None
cameras_image_width, cameras_image_height = 0, 0

chessboards_movement_trajectory = {
    # 'chessboard_front_left': [ 
    #     [], 
    #     ... 
    # ],

    'chessboard_front': [ 
        ['Y', -3.25], 
        ['X', 0.6], 
        ['X', 0.1], 
        ['X', -0.4], 
        ['X', 0.1], 
        ['Z', [-0.0926917, -0.704062, -0.704063, -2.95674]], 
        ['Z', [-0.186157, -0.694746, -0.694747, -2.7735]], 
        ['Z', [-0.281085, -0.678598, -0.678599, -2.59357]], 
        ['Z', [-0.186157, -0.694746, -0.694747, -2.7735]], 
        ['Z', [-0.0926917, -0.704062, -0.704063, -2.95674]], 
        ['Z', [-3.3905e-09, 0.707106, 0.707107, 3.14159]], 
        ['Z', [-0.0926917, 0.704062, 0.704063, -2.95675]], 
        ['Z', [0.186157, -0.694746, -0.694747, 2.7735]], 
        ['Z', [0.281085, -0.678598, -0.678599, 2.59357]], 
        ['Z', [0.186157, -0.694746, -0.694747, 2.7735]], 
        ['Z', [-0.0926917, 0.704062, 0.704063, -2.95675]], 
        ['Z', [-3.3905e-09, 0.707106, 0.707107, 3.14159]], 
        ['Y', -4.0], 
    ],

    # 'chessboard_front_right': [ 
    #     [], 
    #     ... 
    # ],

    'chessboard_rear_left': [ 
        ['XY', [3.08477, -1.0723]], 
        ['XY', [2.78879, -0.66916]], 
        ['XY', [3.07659, -1.07806]], 
        ['XY', [3.36447, -1.48701]], 
        ['XY', [3.07659, -1.07806]], 
        ['Z', [-0.711475, 0.496892, 0.496892, -1.90483]], 
        ['Z', [0.806145, -0.418408, -0.418408, 1.78464]], 
        ['Z', [0.88983, -0.322647, -0.322647, 1.68726]], 
        ['Z', [0.806145, -0.418408, -0.418408, 1.78464]], 
        ['XY', [4.7122, 0.0730846]], 
    ],

    'chessboard_rear': [ 
        ['Y', 4.75], 
        ['X', -0.4], 
        ['X', 0.1], 
        ['X', 0.6], 
        ['X', 0.1], 
        ['Z', [-0.983106, 0.12943, 0.129426, -1.58784]], 
        ['Z', [-0.935113, 0.250564, 0.250561, -1.63784]], 
        ['Z', [-0.862856, 0.357408, 0.357405, -1.71778]], 
        ['Z', [-0.935113, 0.250564, 0.250561, -1.63784]], 
        ['Z', [-0.983106, 0.12943, 0.129426, -1.58784]], 
        ['Z', [1, -1.87157e-06, 1.87158e-06, 1.5708]], 
        ['Z', [0.983106, 0.129426, 0.12943, 1.58784]], 
        ['Z', [0.935113, 0.25056, 0.250565, 1.63784]], 
        ['Z', [0.862857, 0.357404, 0.357409, 1.71778]], 
        ['Z', [0.935113, 0.25056, 0.250565, 1.63784]], 
        ['Z', [0.983106, 0.129426, 0.12943, 1.58784]], 
        ['Z', [1, -1.87157e-06, 1.87158e-06, 1.5708]], 
        ['Y', 5.0], 
    ],

    'chessboard_rear_right': [ 
        ['XY', [-3.07659, -0.93396]], 
        ['Z', [-0.88983, -0.322647, -0.322647, -1.68726]], 
        ['Z', [-0.954692, -0.210432, -0.210433, -1.61715]], 
        ['Z', [-0.992641, -0.085628, -0.0856291, -1.57819]], 
        ['Z', [-0.954692, -0.210432, -0.210433, -1.61715]], 
        ['Z', [-0.88983, -0.322647, -0.322647, -1.68726]], 
        ['Z', [0.806145, 0.418408, 0.418408, 1.78464]], 
        ['Z', [0.711475, 0.496892, 0.496892, 1.90482]], 
        ['Z', [0.611859, 0.559298, 0.559298, 2.0434]], 
        ['Z', [0.511206, 0.607729, 0.607729, 2.19644]], 
        ['Z', [0.611859, 0.559298, 0.559298, 2.0434]], 
        ['Z', [0.711475, 0.496892, 0.496892, 1.90482]], 
        ['Z', [0.806145, 0.418408, 0.418408, 1.78464]], 
        ['XY', [-4.7122, 0.217176]], 
    ]
}


def get_webots_device(device_name):
    webots_device = supervisor.getDevice(device_name)
    webots_device.enable(int(supervisor.getBasicTimeStep()))

    if webots_device is None:
        node_logger.error('The Webots device with the specified name was not found!')
    return webots_device


class Camera:

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


    def __init__(self, webots_camera_name):
        self.device = get_webots_device(webots_camera_name)
        self.device_name = webots_camera_name

        self.calibration_images_count = CALIBRATION_IMAGES_COUNT

        global cameras_image_width, cameras_image_height

        cameras_image_width = cameras_webots_settings['image_width']
        cameras_image_height = cameras_webots_settings['image_height']

        focal_length = cameras_webots_settings['focal_length']

        distortion_center = cameras_webots_settings['distortion_center']
        radial_distortion = cameras_webots_settings['radial_distortion']
        tangential_distortion = cameras_webots_settings['tangential_distortion']

        self.object_points_3D = []  # Список обнаруженных углов шахматной доски в 3D-пространстве (реальный мир)
        self.image_points_2D = []  # Список обнаруженных углов шахматной доски в 2D-пространстве (плоскость изображения)

        ### Параметры калибровки

        # Матрица внутренних (intrinsics) параметров камеры, таких как: fx и fy - фокусное расстояние в пикселях по соответствующим осям, cx и cy - координаты главной точки (principal point), 
        # которая обычно находится в центре изображения
        self.K = np.zeros((3, 3))

        self.K[0][0] = focal_length  # fx
        self.K[0][1] = 0.0
        self.K[0][2] = cameras_image_width * distortion_center[0]  # cx
        self.K[1][0] = 0.0
        self.K[1][1] = focal_length  # fy
        self.K[1][2] = cameras_image_height * distortion_center[1]  # cy
        self.K[2][0] = 0.0
        self.K[2][1] = 0.0
        self.K[2][2] = 1.0

        # Вектор коэффициентов дисторсии, которая зачастую вызвана радиальными и тангенциальными искажениями из-за линзы объектива видеокамеры
        self.D = np.zeros((5, 1))

        self.D[0] = radial_distortion[0]  # k1
        self.D[1] = radial_distortion[1]  # k2
        self.D[2] = tangential_distortion[0]  # p1
        self.D[3] = tangential_distortion[1]  # p2
        self.D[4] = 0.0  # k3

        # Векторы (ось) вращения для каждого из калибровочных изображений, длина которых - это значение угла поворота камеры по одной из трёх осей относительно мировых координат
        self.rotation_vectors = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(self.calibration_images_count)]  # Ориентация в пространстве относительно сцены

        # Векторы перемещения для каждого из калибровочных изображений, длина которых - это значение смещения камеры по одной из трёх осей относительно центра мировых координат
        self.translation_vectors = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(self.calibration_images_count)]  # Положение в пространстве относительно сцены


    def calibrate_camera(self, calibration_image, debug=False):
        calibration_image_gray = cv2.cvtColor(calibration_image, cv2.COLOR_RGBA2GRAY)

        if self.calibration_images_count > 0:
            node_logger.info(f'[{self.device_name}] Searching for a chessboard pattern corners...')

            found, corners = cv2.findChessboardCorners(calibration_image_gray, CHESSBOARD_PATTERN_SIZE)

            if found:
                self.calibration_images_count -= 1
                node_logger.info(f'[{self.device_name}] The corners was found, {self.calibration_images_count} calibration images remained\n')

                # 0.1 - порог точности в пикселях +/ИЛИ 30 - максимальное количество итераций
                term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                # 5x5 пикселей - размер окна внутри которого будут уточняться координаты найденных углов шахматной доски с субпиксельной точностью
                cv2.cornerSubPix(calibration_image_gray, corners, (5, 5), (-1, -1), term_criteria)  # (-1, -1) - уточнение будет производиться по всему окну поиска (без так называемой "мёртвой зоны")

                self.object_points_3D.append(Chessboard.pattern_points_3D)
                self.image_points_2D.append(corners.reshape(-1, 2))  # Убираем дополнительное измерение (массив), возникшее от того, что библиотека OpenCV использует трёхмерные структуры данных

                if debug:
                    calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_RGBA2RGB)
                    cv2.drawChessboardCorners(calibration_image, CHESSBOARD_PATTERN_SIZE, corners, found)
                    cv2.imwrite(os.path.join(
                        os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                        f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{self.device_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
                    ), calibration_image)
        elif self.calibration_images_count == 0:
            self.calibration_images_count -= 1
            node_logger.info(f'[{self.device_name}] Calculation of camera parameters...')

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
                Camera.calibration_flags
            )

            if debug:
                image_height, image_width = calibration_image.shape[:2]

                self_K_new, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (image_width, image_height), 1, (image_width, image_height))
                map_x, map_y = cv2.initUndistortRectifyMap(self.K, self.D, np.eye(3), self_K_new, (image_width, image_height), cv2.CV_32FC1)

                # Применяем карты преобразования для устранения искажений и создания выровненного высококачественного изображения
                calibration_image_undistorted = cv2.remap(calibration_image, map_x, map_y, cv2.INTER_CUBIC)

                x, y, w, h = roi  # Обрезаем искажённые края исправленного изображения
                calibration_image_undistorted = calibration_image_undistorted[y:y+h, x:x+w]  #

                cv2.imwrite(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{self.device_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
                ), calibration_image_undistorted)

            camera_parameters_file = cv2.FileStorage(os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml'
            ), cv2.FILE_STORAGE_WRITE)

            camera_parameters_file.write('image_resolution', np.int32([calibration_image.shape[1], calibration_image.shape[0]]))
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

            node_logger.info(f'[{self.device_name}] Successfully saved on the path "../configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/parameters/{self.device_name}.yaml"!\n')
            node_logger.info(f'[{self.device_name}] Re-projection Error: {mean_error / len(self.object_points_3D)}\n')


class Chessboard:

    pattern_size = (np.prod(CHESSBOARD_PATTERN_SIZE), 3)  # CHESSBOARD_PATTERN_SIZE[0] * CHESSBOARD_PATTERN_SIZE[1] квадратов с 3-мя реальными координатами (X, Y, Z)

    # Создание массива координат углов паттерна шахматной доски в 3D-пространстве
    pattern_points_3D = np.zeros(pattern_size, np.float32)

    # Создание матриц индексов точек пересечения квадратов паттерна шахматной доски по осям y и x
    # и их объединение в один массив размерностью CHESSBOARD_PATTERN_SIZE[0] * CHESSBOARD_PATTERN_SIZE[1] * 2 за счёт применения операции транспонирования
    pattern_points_indices = np.indices(CHESSBOARD_PATTERN_SIZE).T

    # Матрица размера (N, 2), где N - это общее количество точек пересечения (CHESSBOARD_PATTERN_SIZE[0] * CHESSBOARD_PATTERN_SIZE[1]) 
    # и 2 - количество координат (x и y), которыми эти точки задаются. Элементы данной матрицы - это индексы строк и столбцов внутренних пересечений квадратов шахматной доски
    pattern_points_indices = pattern_points_indices.reshape(-1, 2)

    # Операторами среза исключаем координату Z для каждой такой точки, так как предполагается, что доска плоская, и все они лежат в одной плоскости
    pattern_points_3D[:,:2] = pattern_points_indices
    pattern_points_3D *= CHESSBOARD_SQUARE_SIZE  # Конвертируем индексы точек пересечения паттерна в их координаты X и Y с реальной размерностью CHESSBOARD_SQUARE_SIZE сантиметров


    def __init__(self, webots_def_name):
        self.pose_node = supervisor.getFromDef(webots_def_name)

        if self.pose_node:
            self.translation_field = self.pose_node.getField('translation')
            self.rotation_field = self.pose_node.getField('rotation')

        self.trajectory = chessboards_movement_trajectory[webots_def_name]


    def update_position(self):
        current_translation_value = self.translation_field.getSFVec3f()  # x, y, z
        current_rotation_value = self.rotation_field.getSFRotation()  # x, y, z, angle

        if len(self.trajectory) > 0:
            match self.trajectory[0][0]:
                case 'X':
                    ctv_x = round(current_translation_value[0], 1)
                    trajectory_x = round(self.trajectory[0][1], 1)

                    if ctv_x != trajectory_x:
                        # Подразумевается, что разность значений по модулю в данном случае >= CHESSBOARDS_MOVEMENT_SENSITIVITY
                        if ctv_x < trajectory_x:
                            ctv_x += CHESSBOARDS_MOVEMENT_SENSITIVITY  # Движение влево
                        else:
                            ctv_x -= CHESSBOARDS_MOVEMENT_SENSITIVITY  # Движение вправо

                        current_translation_value[0] = ctv_x
                        return self.translation_field.setSFVec3f(current_translation_value)
                    else:
                        # Удаляем достигнутую координату из их общего списка, чтобы не перебирать её на следующей итерации
                        self.trajectory.pop(0)
                case 'Y':
                    ctv_y = round(current_translation_value[1], 1)
                    trajectory_y = round(self.trajectory[0][1], 1)

                    if ctv_y != trajectory_y:
                        if ctv_y < trajectory_y:
                            ctv_y += CHESSBOARDS_MOVEMENT_SENSITIVITY  # Движение вперёд
                        else:
                            ctv_y -= CHESSBOARDS_MOVEMENT_SENSITIVITY  # Движение назад

                        current_translation_value[1] = ctv_y
                        return self.translation_field.setSFVec3f(current_translation_value)
                    else:
                        self.trajectory.pop(0)
                case 'Z':
                    current_rotation_value = self.trajectory[0][1]  # Вращение
                    self.trajectory.pop(0)
                    return self.rotation_field.setSFRotation(current_rotation_value)
                
                case 'XY':
                    ctv_x = round(current_translation_value[0], 1)
                    ctv_y = round(current_translation_value[1], 1)

                    trajectory_x = round(self.trajectory[0][1][0], 1)
                    trajectory_y = round(self.trajectory[0][1][1], 1)

                    if ctv_x != trajectory_x or ctv_y != trajectory_y:
                        if ctv_x < trajectory_x:
                            ctv_x += CHESSBOARDS_MOVEMENT_SENSITIVITY
                        elif ctv_x > trajectory_x:
                            ctv_x -= CHESSBOARDS_MOVEMENT_SENSITIVITY

                        # Корректировки диагонального движения, подобранные опытным путём
                        if ctv_y < trajectory_y:
                            if trajectory_x > 0:
                                ctv_x += 0.05
                            else:
                                ctv_x -= 0.05
                            ctv_y += 0.0525
                        elif ctv_y > trajectory_y:
                            if trajectory_x > 0:
                                ctv_x -= 0.05
                            else:
                                ctv_x += 0.05
                            ctv_y -= 0.0525

                        current_translation_value[:2] = ctv_x, ctv_y
                        return self.translation_field.setSFVec3f(current_translation_value)
                    else:
                        self.trajectory.pop(0)


class ChessboardsControllerNode(Node):

    def __init__(self):
        try:
            super().__init__('chessboards_controller_node')
            self._logger.info('Successfully launched!')

            global node_logger
            node_logger = self._logger

            if supervisor:
                with open(os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
                    f'configs/cameras/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/webots_settings.yaml'
                )) as webots_settings_yaml_file:
                    try:
                        global cameras_webots_settings
                        cameras_webots_settings = yaml.safe_load(webots_settings_yaml_file)
                        webots_settings_yaml_file.close()
                    except yaml.YAMLError as e:
                        self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))

                # self.camera_front_left = Camera('camera_front_left')
                self.camera_front = Camera('camera_front')
                # self.camera_front_right = Camera('camera_front_right')

                self.camera_rear_left = Camera('camera_rear_left')
                self.camera_rear = Camera('camera_rear')
                self.camera_rear_right = Camera('camera_rear_right')

                # self.chessboard_front_left = Chessboard('chessboard_front_left')
                self.chessboard_front = Chessboard('chessboard_front')
                # self.chessboard_front_right = Chessboard('chessboard_front_right')

                self.chessboard_rear_left = Chessboard('chessboard_rear_left')
                self.chessboard_rear = Chessboard('chessboard_rear')
                self.chessboard_rear_right = Chessboard('chessboard_rear_right')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


def image_bytes_to_numpy_array(image_bytes, camera_name='', debug=False):
    image_array = np.frombuffer(image_bytes, np.uint8).reshape((cameras_image_height, cameras_image_width, 4))

    if debug:
        cv2.imwrite(os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)), 
            f'resource/images/{global_settings.USED_CAMERA_MODEL_FOLDER_NAME}/{camera_name}/debug/{time.strftime("%Y%m%d-%H%M%S")}.png'
        ), cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB))

    return image_array


def main(args=None):
    try:
        rclpy.init(args=args)

        global supervisor
        supervisor = Supervisor()

        node = ChessboardsControllerNode()

        while supervisor.step(TIME_STEP) != -1:
            # cfl_image_color = image_bytes_to_numpy_array(node.camera_front_left.getImage(), camera_name=node.camera_front_left.device_name)
            cf_image_color = image_bytes_to_numpy_array(node.camera_front.device.getImage(), camera_name=node.camera_front.device_name)
            # cfr_image_color = image_bytes_to_numpy_array(node.camera_front_right.getImage(), camera_name=node.camera_front_right.device_name)

            crl_image_color = image_bytes_to_numpy_array(node.camera_rear_left.device.getImage(), camera_name=node.camera_rear_left.device_name)
            cr_image_color = image_bytes_to_numpy_array(node.camera_rear.device.getImage(), camera_name=node.camera_rear.device_name)
            crr_image_color = image_bytes_to_numpy_array(node.camera_rear_right.device.getImage(), camera_name=node.camera_rear_right.device_name)

            # node.camera_front_left.calibrate_camera(cfl_image_color)
            # node.chessboard_front_left.update_position()

            node.camera_front.calibrate_camera(cf_image_color)
            node.chessboard_front.update_position()

            # node.camera_front_right.calibrate_camera(cfr_image_color)
            # node.chessboard_front_right.update_position()

            node.camera_rear_left.calibrate_camera(crl_image_color)
            node.chessboard_rear_left.update_position()

            node.camera_rear.calibrate_camera(cr_image_color)
            node.chessboard_rear.update_position()

            node.camera_rear_right.calibrate_camera(crr_image_color)
            node.chessboard_rear_right.update_position()

            rclpy.spin_once(node, timeout_sec=0.1)
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
