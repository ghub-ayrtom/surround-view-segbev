#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <deque>
#include <string>
#include <vector>
#include <mutex>
#include "rclcpp/qos.hpp"

using std::placeholders::_1;

class AsyncPointCloudMergeNode : public rclcpp::Node
{
public:
    AsyncPointCloudMergeNode()
        : Node("async_pointcloud_merge_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        target_frame_ = this->declare_parameter<std::string>("target_frame", "lidar_front");
        cloud_ttl_sec_ = this->declare_parameter<double>("cloud_ttl_sec", 0.1);
        
        rclcpp::QoS lidar_qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));

        lidar_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
        lidar_qos.history(rclcpp::HistoryPolicy::KeepLast);
        lidar_qos.keep_last(1);

        lidar_front_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ego_vehicle/lidar_front/point_cloud", 
            lidar_qos, 
            std::bind(&AsyncPointCloudMergeNode::lidar_front_callback, this, _1)
        );
        lidar_rear_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/ego_vehicle/Ouster_OS1_32/point_cloud", 
            lidar_qos, 
            std::bind(&AsyncPointCloudMergeNode::lidar_rear_callback, this, _1)
        );

        point_cloud_merged_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/cloud_in", 
            lidar_qos
        );

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), 
            std::bind(&AsyncPointCloudMergeNode::merge, this)
        );

        RCLCPP_INFO(this->get_logger(), "Successfully launched!");
    }

private:
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::mutex mutex_;

    std::string target_frame_;
    double cloud_ttl_sec_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_front_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_rear_subscriber_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_merged_publisher_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_front_buffer_;
    std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_rear_buffer_;

    void lidar_front_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        lidar_front_buffer_.emplace_back(
            rclcpp::Time(message->header.stamp.sec, message->header.stamp.nanosec, this->get_clock()->get_clock_type()), 
            message
        );

        prune(lidar_front_buffer_);
    }

    void lidar_rear_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        lidar_rear_buffer_.emplace_back(
            rclcpp::Time(message->header.stamp.sec, message->header.stamp.nanosec, this->get_clock()->get_clock_type()), 
            message
        );

        prune(lidar_rear_buffer_);
    }

    void prune(std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> &buffer) {
        if (buffer.empty()) return;

        rclcpp::Time now(
            buffer.back().second->header.stamp.sec, 
            buffer.back().second->header.stamp.nanosec, 
            this->get_clock()->get_clock_type()
        );
        rclcpp::Time earliest_cloud_stamp(
            buffer.front().second->header.stamp.sec, 
            buffer.front().second->header.stamp.nanosec, 
            this->get_clock()->get_clock_type()
        );

        while (!buffer.empty() && (now - earliest_cloud_stamp).seconds() > cloud_ttl_sec_) {
            buffer.pop_front();

            if (!buffer.empty()) 
                earliest_cloud_stamp = rclcpp::Time(
                    buffer.front().second->header.stamp.sec, 
                    buffer.front().second->header.stamp.nanosec, 
                    this->get_clock()->get_clock_type()
                );
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr transform(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &message, const std::string &sensor_frame) {
        geometry_msgs::msg::TransformStamped tfs;

        try {
            if (!tf_buffer_.canTransform(target_frame_, sensor_frame, message->header.stamp, rclcpp::Duration::from_seconds(0.1))) {
                RCLCPP_WARN(this->get_logger(), "[%s] Transform not available yet...", sensor_frame.c_str());
                return nullptr;
            } else {
                tfs = tf_buffer_.lookupTransform(
                    target_frame_, 
                    sensor_frame, 
                    message->header.stamp, 
                    rclcpp::Duration::from_seconds(0.01) // 0.05 / 0.1 / 0.2 / 0.3 / 1.0
                );
            }
        }
        catch (const tf2::TransformException &e) {
            RCLCPP_ERROR(this->get_logger(), "[%s] %s", sensor_frame.c_str(), e.what());
            return nullptr;
        }

        sensor_msgs::msg::PointCloud2 message_transformed;

        try {
            tf2::doTransform(*message, message_transformed, tfs);
        }
        catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "[%s] %s", sensor_frame.c_str(), e.what());
            return nullptr;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(message_transformed, *transformed_point_cloud);

        return transformed_point_cloud;
    }

    void merge() {
        std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_front_buffer_copy, lidar_rear_buffer_copy;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lidar_front_buffer_copy = lidar_front_buffer_;
            lidar_rear_buffer_copy = lidar_rear_buffer_;
        }

        rclcpp::Time latest_pointcloud_stamp(0, 0, this->get_clock()->get_clock_type());
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_point_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto &[timestamp, pointcloud] : lidar_front_buffer_copy) {
            if (timestamp > latest_pointcloud_stamp) 
                latest_pointcloud_stamp = timestamp;

            auto front_point_cloud = transform(pointcloud, "lidar_front");
            if (front_point_cloud) *merged_point_cloud += *front_point_cloud;
        }

        for (const auto &[timestamp, pointcloud] : lidar_rear_buffer_copy) {
            if (timestamp > latest_pointcloud_stamp) 
                latest_pointcloud_stamp = timestamp;

            auto rear_point_cloud = transform(pointcloud, "lidar_rear");
            if (rear_point_cloud) *merged_point_cloud += *rear_point_cloud;
        }

        if (merged_point_cloud->empty()) return;

        sensor_msgs::msg::PointCloud2 point_cloud_merged;
        pcl::toROSMsg(*merged_point_cloud, point_cloud_merged);

        point_cloud_merged.header.stamp = latest_pointcloud_stamp;
        point_cloud_merged.header.frame_id = target_frame_;

        point_cloud_merged_publisher_->publish(point_cloud_merged);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AsyncPointCloudMergeNode>());
    rclcpp::shutdown();

    return 0;
}
