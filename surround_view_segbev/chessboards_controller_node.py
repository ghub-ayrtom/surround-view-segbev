from configs import global_settings
from rclpy.node import Node
import numpy as np
import rclpy
from controller import Supervisor
import traceback
from .scripts.CameraModel import CameraModel
from .scripts.utils import image_bytes_to_numpy_array


CHESSBOARD_SQUARE_SIZE = 2.5  # Размеры квадрата шахматной доски в сантиметрах
CHESSBOARD_PATTERN_SIZE = (7, 7)  # Размерность внутренних пересечений (паттерна) шахматной доски
CHESSBOARDS_MOVEMENT_SENSITIVITY = 0.1  # Минимальное значение изменения координат траектории движения шахматных досок в Webots (одинаковое для всех)

supervisor = None
node_logger = None

chessboards_movement_trajectory = {
    'chessboard_front_left': [ 
        ['X', 3.0], 
        ['Z', [-0.476905, 0.621515, 0.621515, -2.25159]], 
        ['Z', [-0.377964, 0.654654, 0.654654, -2.41886]], 
        ['Z', [-0.476905, 0.621515, 0.621515, -2.25159]], 
        ['Z', [0.57735, -0.57735, -0.57735, 2.0944]], 
        ['Z', [0.677661, -0.519988, -0.519988, 1.95044]], 
        ['Z', [0.774596, -0.447214, -0.447214, 1.82348]], 
        ['Z', [0.677661, -0.519988, -0.519988, 1.95044]], 
        ['Z', [0.57735, -0.57735, -0.57735, 2.0944]], 
        ['Y', -0.9675], 
        ['Y', -1.9675], 
    ],

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

    'chessboard_front_blind': [ 
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

    'chessboard_front_right': [ 
        ['X', -3.0], 
        ['Z', [-0.677661, -0.519988, -0.519988, -1.95044]], 
        ['Z', [-0.774596, -0.447214, -0.447214, -1.82348]], 
        ['Z', [-0.677661, -0.519988, -0.519988, -1.95044]], 
        ['Z', [0.57735, 0.57735, 0.57735, 2.0944]], 
        ['Z', [0.476905, 0.621515, 0.621515, 2.25159]], 
        ['Z', [0.377964, 0.654654, 0.654654, 2.41886]], 
        ['Z', [0.476905, 0.621515, 0.621515, 2.25159]], 
        ['Z', [0.57735, 0.57735, 0.57735, 2.0944]], 
        ['Y', -2.9675], 
        ['Y', -1.9675], 
    ],

    # 'chessboard_rear_left': [ 
    #     ['XY', [3.08477, -1.0723]], 
    #     ['XY', [2.78879, -0.66916]], 
    #     ['XY', [3.07659, -1.07806]], 
    #     ['XY', [3.36447, -1.48701]], 
    #     ['XY', [3.07659, -1.07806]], 
    #     ['Z', [-0.711475, 0.496892, 0.496892, -1.90483]], 
    #     ['Z', [0.806145, -0.418408, -0.418408, 1.78464]], 
    #     ['Z', [0.88983, -0.322647, -0.322647, 1.68726]], 
    #     ['Z', [0.806145, -0.418408, -0.418408, 1.78464]], 
    #     ['XY', [4.7122, 0.0730846]], 
    # ],

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

    # 'chessboard_rear_right': [ 
    #     ['XY', [-3.07659, -0.93396]], 
    #     ['Z', [-0.88983, -0.322647, -0.322647, -1.68726]], 
    #     ['Z', [-0.954692, -0.210432, -0.210433, -1.61715]], 
    #     ['Z', [-0.992641, -0.085628, -0.0856291, -1.57819]], 
    #     ['Z', [-0.954692, -0.210432, -0.210433, -1.61715]], 
    #     ['Z', [-0.88983, -0.322647, -0.322647, -1.68726]], 
    #     ['Z', [0.806145, 0.418408, 0.418408, 1.78464]], 
    #     ['Z', [0.711475, 0.496892, 0.496892, 1.90482]], 
    #     ['Z', [0.611859, 0.559298, 0.559298, 2.0434]], 
    #     ['Z', [0.511206, 0.607729, 0.607729, 2.19644]], 
    #     ['Z', [0.611859, 0.559298, 0.559298, 2.0434]], 
    #     ['Z', [0.711475, 0.496892, 0.496892, 1.90482]], 
    #     ['Z', [0.806145, 0.418408, 0.418408, 1.78464]], 
    #     ['XY', [-4.7122, 0.217176]], 
    # ]
}


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

            global node_logger
            node_logger = self._logger

            if supervisor:
                self.chessboard_front_left = Chessboard('chessboard_front_left')
                self.chessboard_front = Chessboard('chessboard_front')
                self.chessboard_front_blind = Chessboard('chessboard_front_blind')
                self.chessboard_front_right = Chessboard('chessboard_front_right')

                # self.chessboard_rear_left = Chessboard('chessboard_rear_left')
                self.chessboard_rear = Chessboard('chessboard_rear')
                # self.chessboard_rear_right = Chessboard('chessboard_rear_right')

                self.camera_front_left = CameraModel('camera_front_left', node_logger, related_chessboard=self.chessboard_front_left, load_parameters=False)
                self.camera_front = CameraModel('camera_front', node_logger, related_chessboard=self.chessboard_front, load_parameters=False)
                self.camera_front_blind = CameraModel('camera_front_blind', node_logger, related_chessboard=self.chessboard_front_blind, load_parameters=False)
                self.camera_front_right = CameraModel('camera_front_right', node_logger, related_chessboard=self.chessboard_front_right, load_parameters=False)

                # self.camera_rear_left = CameraModel('camera_rear_left', node_logger, related_chessboard=self.chessboard_rear_left, load_parameters=False)
                self.camera_rear = CameraModel('camera_rear', node_logger, related_chessboard=self.chessboard_rear, load_parameters=False)
                # self.camera_rear_right = CameraModel('camera_rear_right', node_logger, related_chessboard=self.chessboard_rear_right, load_parameters=False)

                self._logger.info('Successfully launched!')
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


def main(args=None):
    try:
        rclpy.init(args=args)

        global supervisor
        supervisor = Supervisor()

        node = ChessboardsControllerNode()

        while supervisor.step(global_settings.SIMULATION_TIME_STEP) != -1:
            cfl_image_color = image_bytes_to_numpy_array(node.camera_front_left.device.getImage(), node.camera_front_left.image_shape, camera_name=node.camera_front_left.device_name)
            cf_image_color = image_bytes_to_numpy_array(node.camera_front.device.getImage(), node.camera_front.image_shape, camera_name=node.camera_front.device_name)
            cfb_image_color = image_bytes_to_numpy_array(node.camera_front_blind.device.getImage(), node.camera_front_blind.image_shape, camera_name=node.camera_front_blind.device_name)
            cfr_image_color = image_bytes_to_numpy_array(node.camera_front_right.device.getImage(), node.camera_front_right.image_shape, camera_name=node.camera_front_right.device_name)

            # crl_image_color = image_bytes_to_numpy_array(node.camera_rear_left.device.getImage(), node.camera_rear_left.image_shape, camera_name=node.camera_rear_left.device_name)
            cr_image_color = image_bytes_to_numpy_array(node.camera_rear.device.getImage(), node.camera_rear.image_shape, camera_name=node.camera_rear.device_name)
            # crr_image_color = image_bytes_to_numpy_array(node.camera_rear_right.device.getImage(), node.camera_rear_right.image_shape, camera_name=node.camera_rear_right.device_name)

            node.camera_front_left.calibrate_camera(cfl_image_color)
            node.chessboard_front_left.update_position()

            node.camera_front.calibrate_camera(cf_image_color)
            node.chessboard_front.update_position()

            node.camera_front_blind.calibrate_camera(cfb_image_color)
            node.chessboard_front_blind.update_position()

            node.camera_front_right.calibrate_camera(cfr_image_color)
            node.chessboard_front_right.update_position()

            # node.camera_rear_left.calibrate_camera(crl_image_color)
            # node.chessboard_rear_left.update_position()

            node.camera_rear.calibrate_camera(cr_image_color)
            node.chessboard_rear.update_position()

            # node.camera_rear_right.calibrate_camera(crr_image_color)
            # node.chessboard_rear_right.update_position()

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
