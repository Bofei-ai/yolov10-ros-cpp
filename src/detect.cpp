#include "yolov10.h"
#include <iostream>

#include "ros/ros.h"

using namespace std;

int main(int argc, char *argv[]){
    ros::init(argc, argv, "detect_node");
    ros::NodeHandle nh("~");

    InitParam params;
    params.class_names = nh.param("class_names", vector<string>{});
    params.model_path = nh.param("model_path", string(""));
    params.confidence_threshold = nh.param("confidence_threshold", 0.3);
    params.iou_threshold = nh.param("iou_threshold", 0.3);
    params.nms_sigma = nh.param("nms_sigma", 0.5);
    vector<double> input_shape = nh.param("input_shape", vector<double>{1, 3, 640, 640});
    params.input_shape = vector<int64_t>(input_shape.begin(), input_shape.end());
    params.cuda_enable = nh.param("cuda_enable", true);
    params.intra_opNum_threads = nh.param("intra_opNum_threads", 1);
    params.log_severity_level = nh.param("log_severity_level", 3);

    Yolov10* yolo = new Yolov10(params);

    Mat image = cv::imread("/home/bofia/powerGrid_ws/src/test_pkg/src/first_test/images/test.jpg");
    
    std::vector<float> input_tensor_values = yolo->preprocess(image);
    std::vector<float> results = yolo->runInference(input_tensor_values);
    std::vector<OdResult> detections = yolo->postprocess(results);
    cv::Mat output = yolo->drawLabels(image, detections);

    std::cout 
        << "[YOLO_V10(CUDA)]: " 
        << yolo->getWarmupTime() << "ms warm-up, "
        << yolo->getPreprocessTime() << "ms pre-process, " 
        << yolo->getInferenceTime() << "ms inference, " 
        << yolo->getPostprocessTime() << "ms post-process." 
        << std::endl;

    cv::imshow("Result of Detection", output);
    cv::waitKey(0);
    cv::destroyAllWindows();

    delete yolo;

    ros::spin();
    return 0;
}