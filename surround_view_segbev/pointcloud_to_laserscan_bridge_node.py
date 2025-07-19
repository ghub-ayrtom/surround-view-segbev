import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
import traceback
from surround_view_segbev.configs import qos_profiles


class PointCloudToLaserScanBridgeNode(Node):
    def __init__(self):
        super().__init__('pointcloud_to_laserscan_bridge_node')

        self.create_subscription(PointCloud2, '/cloud_in', self.__point_cloud_dummy_callback, qos_profiles.bridge_qos)
        self.create_subscription(LaserScan, '/scan', self.__scan_callback, qos_profiles.scan_qos)

        self.laserscan_bridge_publisher = self.create_publisher(LaserScan, '/scan_reliable', qos_profiles.bridge_qos)
        
        self.get_logger().info('Successfully launched!')

    def __point_cloud_dummy_callback(self, message):
        pass

    def __scan_callback(self, message):
        self.laserscan_bridge_publisher.publish(message)


def main(args=None):
    try:
        rclpy.init(args=args)

        node = PointCloudToLaserScanBridgeNode()
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
