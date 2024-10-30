from rclpy.node import Node
import rclpy
from .scripts.CameraModel import CameraModel
import traceback
from controller import Supervisor
from configs import global_settings
from .scripts.utils import image_bytes_to_numpy_array


class ProjectionMatricesNode(Node):

    def __init__(self):
        try:
            super().__init__('projection_matrices_node')
            self._logger.info('Successfully launched!')

            self.camera_front_left = CameraModel('camera_front_left', self._logger)
            self.camera_front = CameraModel('camera_front', self._logger)
            self.camera_front_right = CameraModel('camera_front_right', self._logger)

            # self.camera_rear_left = CameraModel('camera_rear_left', self._logger)
            self.camera_rear = CameraModel('camera_rear', self._logger)
            # self.camera_rear_right = CameraModel('camera_rear_right', self._logger)
        except Exception as e:
            self._logger.error(''.join(traceback.TracebackException.from_exception(e).format()))


def main(args=None):
    try:
        rclpy.init(args=args)

        supervisor = Supervisor()
        node = ProjectionMatricesNode()

        while supervisor.step(global_settings.SIMULATION_TIME_STEP) != -1:
            cfl_image_color = image_bytes_to_numpy_array(node.camera_front_left.device.getImage(), node.camera_front_left.image_shape, camera_name=node.camera_front_left.device_name)
            cf_image_color = image_bytes_to_numpy_array(node.camera_front.device.getImage(), node.camera_front.image_shape, camera_name=node.camera_front.device_name)
            cfr_image_color = image_bytes_to_numpy_array(node.camera_front_right.device.getImage(), node.camera_front_right.image_shape, camera_name=node.camera_front_right.device_name)

            # crl_image_color = image_bytes_to_numpy_array(node.camera_rear_left.device.getImage(), node.camera_rear_left.image_shape, camera_name=node.camera_rear_left.device_name)
            cr_image_color = image_bytes_to_numpy_array(node.camera_rear.device.getImage(), node.camera_rear.image_shape, camera_name=node.camera_rear.device_name)
            # crr_image_color = image_bytes_to_numpy_array(node.camera_rear_right.device.getImage(), node.camera_rear_right.image_shape, camera_name=node.camera_rear_right.device_name)

            node.camera_front_left.get_projection_matrix(cfl_image_color)
            node.camera_front.get_projection_matrix(cf_image_color)
            node.camera_front_right.get_projection_matrix(cfr_image_color)

            # node.camera_rear_left.get_projection_matrix(crl_image_color)
            node.camera_rear.get_projection_matrix(cr_image_color)
            # node.camera_rear_right.get_projection_matrix(crr_image_color)

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
