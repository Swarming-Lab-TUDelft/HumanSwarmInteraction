cmake_minimum_required(VERSION 3.5)
project(keypoints_tracking)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(human_swarm_interaction_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
include_directories(include)

# Single keypoint tracking node
add_executable(single_keypoint_tracking_node
  src/single_keypoint_tracking_node.cpp
  src/linear_kalman_filter.cpp
)

target_include_directories(single_keypoint_tracking_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(single_keypoint_tracking_node
  rclcpp
  std_msgs
  sensor_msgs
  human_swarm_interaction_interfaces
  Eigen3
)

# Multiple keypoint tracking node
add_executable(multiple_keypoints_tracking_node
  src/multiple_keypoints_tracker_node.cpp
  src/linear_kalman_filter.cpp
)

target_include_directories(multiple_keypoints_tracking_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(multiple_keypoints_tracking_node
  rclcpp
  std_msgs
  sensor_msgs
  human_swarm_interaction_interfaces
  Eigen3
)

# Install executables
install(TARGETS
  single_keypoint_tracking_node
  multiple_keypoints_tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files.
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
