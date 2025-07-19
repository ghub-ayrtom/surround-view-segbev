cmake_minimum_required(VERSION 3.8)
project(pointcloud_preprocessing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

add_executable(async_pointcloud_merge_node src/async_pointcloud_merge_node.cpp)

ament_target_dependencies(async_pointcloud_merge_node 
  rclcpp 
  std_msgs 
  sensor_msgs 
  geometry_msgs 
  tf2 
  tf2_ros 
  tf2_sensor_msgs 
  pcl_ros 
  pcl_conversions
)

target_include_directories(async_pointcloud_merge_node PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
  $<INSTALL_INTERFACE:include>
)

target_compile_features(async_pointcloud_merge_node PUBLIC 
  c_std_99 
  cxx_std_17
)

install(TARGETS 
  async_pointcloud_merge_node 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
