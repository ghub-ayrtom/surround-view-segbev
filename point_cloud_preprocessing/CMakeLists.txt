cmake_minimum_required(VERSION 3.8)
project(point_cloud_preprocessing)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(include)

add_executable(async_point_cloud_merge_node src/async_point_cloud_merge_node.cpp)

ament_target_dependencies(async_point_cloud_merge_node 
  rclcpp 
  sensor_msgs 
  tf2_ros 
  tf2_sensor_msgs 
  pcl_conversions
)

install(TARGETS 
  async_point_cloud_merge_node 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
