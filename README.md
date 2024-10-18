# yolov10-ros-cpp

# 项目名称

## 安装要求

请提前在电脑中安装以下软件：

- Nvidia Driver
- CUDA Toolkit
- cuDNN
- C++ 版本的 Onnx Runtime

> **注意**：确保上述四个软件的版本匹配。

### 本机版本信息：

1. 显卡：RTX 3030
2. 系统：Ubuntu 22.04
3. 驱动：nvidia-driver-550
4. CUDA：12.4.0
5. cuDNN：8.9.7
6. OpenCV：4.9.0

## 主流部署方式

目前主流的部署方式有以下 4 种：

1. **OpenCV's DNN API**：调用 onnx 格式的模型文件
2. **LibTorch**：即 PyTorch 的 C++ 版本，可以直接使用 `.pt` 后缀的模型文件
3. **Onnx Runtime**：需使用 `.onnx` 后缀的模型文件
4. **TensorRT**：需使用 `.engine` 后缀的模型文件（专门用于 NVIDIA GPU）

使用 Onnx Runtime 调用 onnx 模型速度比 TensorRT 慢，占用资源相对较多，但不需要转换 engine 格式，且转换的 engine 文件只能在对应算力的显卡上运行。onnx 可实现跨 X86/ARM 架构的迁移应用。

## 参考链接

1. [链接1](https://zhuanlan.zhihu.com/p/444350949)
2. [链接2](https://github.com/THU-MIG/yolov10/tree/main/examples/YOLOv8-ONNXRuntime-CPP)
3. [链接3](https://github.com/DanielSarmiento04/yolov10cpp)
4. [链接4](https://www.cnblogs.com/water-wells/p/18304361)

> **主要参考**：链接3

## 运行方法

1. 修改 `yolov10.yaml` 中的 `model_path` 和 `detec.cpp` 中的 `image_path`，然后运行 `catkin_make` 编译。
2. 在终端中输入 `roscore`。
3. 新开一个终端，输入 `roslaunch det_pkg detect`。
