cmake_minimum_required(VERSION 3.8)
project(image_recognition_jetson)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_recognition_interfaces REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

find_package(jetson-utils QUIET)
find_package(jetson-inference QUIET)
if(jetson-inference_FOUND)
  find_package(CUDAToolkit REQUIRED)
  add_library(detect_net STATIC src/detect_net.cpp)
  target_link_libraries(detect_net PRIVATE jetson-inference CUDA::cudart)
else()
  message(STATUS "Jetson inference unavailable, building the mock detector")
  add_library(detect_net STATIC src/detect_net_mock.cpp)
endif()
target_link_libraries(detect_net PUBLIC
  ${image_recognition_interfaces_TARGETS}
  opencv_core
  opencv_imgproc
)

add_executable(detect_net_ros src/detect_net_ros.cpp)
target_link_libraries(detect_net_ros PRIVATE
  detect_net
  cv_bridge::cv_bridge
  image_transport::image_transport
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
)

install(TARGETS detect_net_ros DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/install_jetson_inference.bash DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
