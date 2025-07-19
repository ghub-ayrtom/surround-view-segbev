from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Imu
import traceback
from surround_view_segbev.configs import qos_profiles
from surround_view_segbev.scripts.utils import *
from geometry_msgs.msg import Quaternion, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class EgoVehicleOdometryNode(Node):
    def __init__(self):
        try:
            super().__init__('ego_vehicle_odometry_node')

            self.imu_data = Quaternion()
            self.create_subscription(Imu, '/ego_vehicle/imu', self.__imu_callback, qos_profiles.imu_qos)

            self.create_subscription(PointStamped, '/ego_vehicle/gps', self.__gps_callback, qos_profiles.gps_qos)

            self.tfs = TransformStamped()
            self.tfs.header.frame_id = 'odom'
            self.tfs.child_frame_id = 'base_link'

            self.tf_broadcaster = TransformBroadcaster(self)

            self.odom = Odometry()
            self.odom.header.frame_id = 'odom'
            self.odom.child_frame_id = 'base_link'

            # В течение нескольких секунд после запуска решения, эго-автомобиль будет находиться в центре глобального фрейма, 
            # однако, из-за его смещения, данные одометрии в этот момент не будут соответствовать фактическому месторасположению робота на карте
            self.skip_odom_publishes_count = 100

            self.odom_publisher = self.create_publisher(Odometry, '/odom', qos_profiles.odometry_qos)

            self.get_logger().info('Successfully launched!')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __imu_callback(self, message):
        self.imu_data = message.orientation

    def __gps_callback(self, message):
        # self.get_logger().info(f'[False, {message.point.y}, {message.point.x}, 0.0], ')

        self.tfs.header.stamp = self.get_clock().now().to_msg()

        self.tfs.transform.translation.x = message.point.x
        self.tfs.transform.translation.y = message.point.y
        self.tfs.transform.rotation = self.imu_data

        self.tf_broadcaster.sendTransform(self.tfs)

        self.odom.header.stamp = self.get_clock().now().to_msg()

        self.odom.pose.pose.position.x = message.point.x
        self.odom.pose.pose.position.y = message.point.y
        self.odom.pose.pose.orientation = self.imu_data

        if self.skip_odom_publishes_count == 0:
            self.odom_publisher.publish(self.odom)
        else:
            self.skip_odom_publishes_count -= 1


def main(args=None):
    try:
        rclpy.init(args=args)

        node = EgoVehicleOdometryNode()
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
