from rclpy.node import Node
import rclpy
import traceback
from surround_view_segbev.configs import qos_profiles
from surround_view_segbev.scripts.utils import *
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import TaskResult, BasicNavigator


class Nav2PathPlanningNode(Node):
    def __init__(self):
        try:
            super().__init__('nav2_path_planning_node')

            self.current_route_point_index = 0

            self.route = [
                [-76.7706, -236.233],  # x, y
                [-64.066, -234.443], 
                [0.300003, 14.1], 
            ]

            self.goal_poses = []
            self.navigator = BasicNavigator()

            self.go_through_poses = False
            self.nav_through_poses_task = None

            self.initial_pose_subscriber = self.create_subscription(
                PoseWithCovarianceStamped, 
                '/initialpose', 
                self.__initial_pose_callback, 
                qos_profiles.pose_qos, 
            )

            self.create_timer(0.2, self.__navigate)

            self.get_logger().info('Successfully launched!')
        except Exception as e:
            self.get_logger().error(''.join(traceback.TracebackException.from_exception(e).format()))

    def __initial_pose_callback(self, message):
        initial_pose = PoseStamped()
        self.destroy_subscription(self.initial_pose_subscriber)

        initial_pose.header.frame_id = message.header.frame_id
        initial_pose.header.stamp = message.header.stamp
        initial_pose.pose = message.pose.pose

        self.navigator.setInitialPose(initial_pose)
        self.navigator.waitUntilNav2Active()

        goal_pose = PoseStamped()

        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

        goal_pose.pose.position.x = self.route[self.current_route_point_index][0]
        goal_pose.pose.position.y = self.route[self.current_route_point_index][1]

        self.goal_poses.append(goal_pose)
        self.current_route_point_index += 1

        if self.navigator.getPathThroughPoses(initial_pose, self.goal_poses):
            self.nav_through_poses_task = self.navigator.goThroughPoses(self.goal_poses)

    def __navigate(self):
        if not self.navigator.isTaskComplete():
            self.go_through_poses = False

            feedback = self.navigator.getFeedback()

            # if feedback:
            #     self.get_logger().info(f'{feedback.distance_remaining, feedback.number_of_poses_remaining}')
        else:
            result = self.navigator.getResult()

            if not self.go_through_poses and self.current_route_point_index <= len(self.route) - 1:
                if result == TaskResult.SUCCEEDED:
                    self.goal_poses.clear()
                    goal_pose = PoseStamped()

                    goal_pose.header.frame_id = 'map'
                    goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

                    goal_pose.pose.position.x = self.route[self.current_route_point_index][0]
                    goal_pose.pose.position.y = self.route[self.current_route_point_index][1]

                    self.goal_poses.append(goal_pose)
                    self.current_route_point_index += 1

                    self.nav_through_poses_task = self.navigator.goThroughPoses(self.goal_poses)
                    self.go_through_poses = True


def main(args=None):
    try:
        rclpy.init(args=args)

        node = Nav2PathPlanningNode()
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
