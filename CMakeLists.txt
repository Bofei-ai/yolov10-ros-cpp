cmake_minimum_required(VERSION 3.0.2)
project(det_pkg)
catkin_package()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# -------------- ONNXRUNTIME  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.19.2)
if (USE_CUDA)
    set(ONNXRUNTIME_ROOT "/home/bofia/下载/onnxruntime")
else ()
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
endif ()
include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include/onnxruntime/core)

# -------------- Yolov10  ------------------#
include_directories(
  include/det_pkg
  ${catkin_INCLUDE_DIRS}
)

add_library(yolov10_lib
  src/yolov10.cpp
)

add_dependencies(yolov10_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(yolov10_lib ${OpenCV_LIBRARIES} ${ONNXRUNTIME_ROOT}/build/Linux/Release/libonnxruntime.so)
if (USE_CUDA)
    target_link_libraries(yolov10_lib ${CUDA_LIBRARIES})
endif ()

# -------------- detect  ------------------#
add_executable(detect
  src/detect.cpp 
)

add_dependencies(detect ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(detect
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_ROOT}/build/Linux/Release/libonnxruntime.so
  yolov10_lib
)
