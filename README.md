# yolov10-ros-cpp

### 使用C++在ROS中部署yolov10

## 安装要求

请提前在电脑中安装以下软件：

- Nvidia Driver
- CUDA Toolkit
- cuDNN
- C++ 版本的 Onnx Runtime

> **注意**：确保上述四个软件的版本匹配。

### 本机版本信息：

1. 显卡：RTX 3060
2. 系统：Ubuntu 22.04
3. 驱动：nvidia-driver-550
4. CUDA：12.4.0
5. cuDNN：8.9.7
6. OpenCV：4.9.0
7. ROS-noetic

## 运行方法

1. 修改 `yolov10.yaml` 中的 `model_path` 和 `detec.cpp` 中的 `image_path`，然后运行 `catkin_make` 编译。
2. 在终端中输入 `roscore`。
3. 新开一个终端，输入 `roslaunch det_pkg detect`。
