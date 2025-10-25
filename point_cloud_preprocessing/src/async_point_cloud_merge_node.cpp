#include "async_point_cloud_merge_node.hpp"

AsyncPointCloudMergeNode::AsyncPointCloudMergeNode()
    : Node("async_point_cloud_merge_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
    lidar_first_frame_ = this->get_parameter("lidar_first_frame").as_string();
    lidar_first_topic_ = this->get_parameter("lidar_first_topic").as_string();

    lidar_second_frame_ = this->get_parameter("rotating_lidar_second_frame").as_string();
    lidar_second_topic_ = this->get_parameter("rotating_lidar_second_topic").as_string();

    target_frame_ = this->get_parameter("target_frame").as_string();
    point_cloud_merged_topic_ = this->get_parameter("point_cloud_merged_topic").as_string();

    point_cloud_merged_ttl_sec_ = this->get_parameter("point_cloud_merged_ttl_sec").as_double();
    lookup_transform_timeout_sec_ = this->get_parameter("lookup_transform_timeout_sec").as_double();
    
    rclcpp::QoS lidar_qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data));

    lidar_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    lidar_qos.history(rclcpp::HistoryPolicy::KeepLast);
    lidar_qos.keep_last(1);

    lidar_first_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_first_topic_, 
        lidar_qos, 
        bind(&AsyncPointCloudMergeNode::lidar_first_callback, this, _1)
    );
    lidar_second_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_second_topic_, 
        lidar_qos, 
        bind(&AsyncPointCloudMergeNode::lidar_second_callback, this, _1)
    );

    point_cloud_merged_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        point_cloud_merged_topic_, 
        lidar_qos
    );

    timer_ = this->create_wall_timer(
        chrono::milliseconds(100), 
        bind(&AsyncPointCloudMergeNode::merge, this)
    );

    RCLCPP_INFO(this->get_logger(), "Successfully launched!");
}

void AsyncPointCloudMergeNode::lidar_first_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& message) {
    lock_guard<mutex> lock(mutex_);

    lidar_first_buffer_.emplace_back(
        rclcpp::Time(message->header.stamp.sec, message->header.stamp.nanosec, this->get_clock()->get_clock_type()), 
        message
    );

    prune(lidar_first_buffer_);
}

void AsyncPointCloudMergeNode::rotating_lidar_crop_rear_180_degrees(
    sensor_msgs::msg::PointCloud2::SharedPtr input_point_cloud, 
    sensor_msgs::msg::PointCloud2::SharedPtr output_point_cloud
) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr croped_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*input_point_cloud, *pcl_point_cloud);
    
    croped_point_cloud->header = pcl_point_cloud->header;
    croped_point_cloud->is_dense = pcl_point_cloud->is_dense;
    
    for (const auto& point : *pcl_point_cloud) {
        if (point.x > 0) {
            // Добавляем точку из передней полусферы по оси X
            croped_point_cloud->points.push_back(point);
        } else if (point.x < 0) {
            // Проверяем угол точки в задней полусфере для более точного отсечения
            float angle = atan2(point.y, point.x);
            float angle_degrees = (angle < 0 ? angle + 2 * M_PI : angle) * 180.0 / M_PI;
            
            if (angle_degrees >= 270.0 || angle_degrees < 90.0)
                croped_point_cloud->points.push_back(point);
        }
    }

    pcl::toROSMsg(*croped_point_cloud, *output_point_cloud);
    output_point_cloud->header = input_point_cloud->header;
}

void AsyncPointCloudMergeNode::lidar_second_callback(const sensor_msgs::msg::PointCloud2::SharedPtr message) {
    auto point_cloud_croped = std::make_shared<sensor_msgs::msg::PointCloud2>();
    rotating_lidar_crop_rear_180_degrees(message, point_cloud_croped);

    lock_guard<mutex> lock(mutex_);
    
    lidar_second_buffer_.emplace_back(
        rclcpp::Time(message->header.stamp.sec, message->header.stamp.nanosec, this->get_clock()->get_clock_type()), 
        point_cloud_croped
    );

    prune(lidar_second_buffer_);
}

void AsyncPointCloudMergeNode::prune(deque<pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> &buffer) {
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

    while (!buffer.empty() && (now - earliest_cloud_stamp).seconds() > point_cloud_merged_ttl_sec_) {
        buffer.pop_front();

        if (!buffer.empty()) 
            earliest_cloud_stamp = rclcpp::Time(
                buffer.front().second->header.stamp.sec, 
                buffer.front().second->header.stamp.nanosec, 
                this->get_clock()->get_clock_type()
            );
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr AsyncPointCloudMergeNode::transform(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &message, 
    const string &lidar_frame
) {
    geometry_msgs::msg::TransformStamped tfs;

    try {
        if (!tf_buffer_.canTransform(target_frame_, lidar_frame, message->header.stamp, rclcpp::Duration::from_seconds(0.1))) {
            RCLCPP_WARN(this->get_logger(), "[%s] Transform not available yet...", lidar_frame.c_str());
            return nullptr;
        } else {
            tfs = tf_buffer_.lookupTransform(
                target_frame_, 
                lidar_frame, 
                message->header.stamp, 
                rclcpp::Duration::from_seconds(lookup_transform_timeout_sec_)
            );
        }
    }
    catch (const tf2::TransformException &e) {
        RCLCPP_ERROR(this->get_logger(), "[%s] %s", lidar_frame.c_str(), e.what());
        return nullptr;
    }

    sensor_msgs::msg::PointCloud2 message_transformed;

    try {
        tf2::doTransform(*message, message_transformed, tfs);
    }
    catch (const exception &e) {
        RCLCPP_ERROR(this->get_logger(), "[%s] %s", lidar_frame.c_str(), e.what());
        return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_point_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(message_transformed, *transformed_point_cloud);

    return transformed_point_cloud;
}

void AsyncPointCloudMergeNode::merge() {
    deque<pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::ConstSharedPtr>> lidar_first_buffer_copy, lidar_second_buffer_copy;
    {
        lock_guard<mutex> lock(mutex_);
        lidar_first_buffer_copy = lidar_first_buffer_;
        lidar_second_buffer_copy = lidar_second_buffer_;
    }

    rclcpp::Time latest_point_cloud_stamp(0, 0, this->get_clock()->get_clock_type());
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_point_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto &[timestamp, point_cloud] : lidar_first_buffer_copy) {
        if (timestamp > latest_point_cloud_stamp) 
            latest_point_cloud_stamp = timestamp;

        auto point_cloud_transformed = transform(point_cloud, lidar_first_frame_);
        if (point_cloud_transformed) *merged_point_cloud += *point_cloud_transformed;
    }

    for (const auto &[timestamp, point_cloud] : lidar_second_buffer_copy) {
        if (timestamp > latest_point_cloud_stamp) 
            latest_point_cloud_stamp = timestamp;

        auto point_cloud_transformed = transform(point_cloud, lidar_second_frame_);
        if (point_cloud_transformed) *merged_point_cloud += *point_cloud_transformed;
    }

    if (merged_point_cloud->empty()) return;

    sensor_msgs::msg::PointCloud2 point_cloud_merged;
    pcl::toROSMsg(*merged_point_cloud, point_cloud_merged);

    point_cloud_merged.header.stamp = latest_point_cloud_stamp;
    point_cloud_merged.header.frame_id = target_frame_;

    point_cloud_merged_publisher_->publish(point_cloud_merged);
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<AsyncPointCloudMergeNode>());
    rclcpp::shutdown();

    return 0;
}
