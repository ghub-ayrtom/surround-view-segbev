#ifndef ASYNC_POINT_CLOUD_MERGE_NODE_HPP
#define ASYNC_POINT_CLOUD_MERGE_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <deque>
#include <mutex>
#include "rclcpp/qos.hpp"

using namespace std;
using placeholders::_1;

class AsyncPointCloudMergeNode : public rclcpp::Node
{
public:
    AsyncPointCloudMergeNode();

private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    mutex mutex_;

    string lidar_first_frame_ = this->declare_parameter<string>("lidar_first_frame", "");
    string lidar_first_topic_ = this->declare_parameter<string>("lidar_first_topic", "");

    string lidar_second_frame_ = this->declare_parameter<string>("rotating_lidar_second_frame", "");
    string lidar_second_topic_ = this->declare_parameter<string>("rotating_lidar_second_topic", "");

    string target_frame_ = this->declare_parameter<string>("target_frame", "");
    string point_cloud_merged_topic_ = this->declare_parameter<string>("point_cloud_merged_topic", "");

    double point_cloud_merged_ttl_sec_ = this->declare_parameter<double>("point_cloud_merged_ttl_sec", 0.0);
    double lookup_transform_timeout_sec_ = this->declare_parameter<double>("lookup_transform_timeout_sec", 0.0);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_first_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_second_subscriber_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_merged_publisher_;

    rclcpp::TimerBase::SharedPtr timer_;

    deque<pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_first_buffer_;
    deque<pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_second_buffer_;

    void lidar_first_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& message);
    void lidar_second_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message);

    void rotating_lidar_crop_rear_180_degrees(
        sensor_msgs::msg::PointCloud2::SharedPtr input_point_cloud, 
        sensor_msgs::msg::PointCloud2::SharedPtr output_point_cloud
    );
    
    void prune(deque<pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> &buffer);

    pcl::PointCloud<pcl::PointXYZ>::Ptr transform(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &message, 
        const string &lidar_frame
    );
    
    void merge();
};

#endif
