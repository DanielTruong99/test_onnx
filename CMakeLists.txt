cmake_minimum_required(VERSION 3.8)
project(test_onnx)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}; /home/humanoid/DanielWorkspace/test_onnx_runtime/src/test_onnx/onnxruntime-linux-x64-1.21.0/lib/cmake)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(onnxruntime REQUIRED CONFIG)

include_directories(test_onnx_node /home/humanoid/DanielWorkspace/test_onnx_runtime/src/test_onnx/onnxruntime-linux-x64-1.21.0/include)
link_directories(test_onnx_node /home/humanoid/DanielWorkspace/test_onnx_runtime/src/test_onnx/onnxruntime-linux-x64-1.21.0/lib)
add_executable(test_onnx_node src/main.cpp)
target_link_libraries(test_onnx_node onnxruntime)
ament_target_dependencies(test_onnx_node rclcpp sensor_msgs geometry_msgs)





install(
  TARGETS
  test_onnx_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package() 
