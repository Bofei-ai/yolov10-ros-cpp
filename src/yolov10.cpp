#include "yolov10.h"
#include <regex>
#include <algorithm>
#include <cmath>


Yolov10::~Yolov10(){
    delete _session;
}


/**
 * @brief Letterbox an image to fit into the target size without changing its aspect ratio.
 * Adds padding to the shorter side to match the target dimensions.
 *
 * @param src Image to be letterboxed.
 * @param target_size Desired output size (width and height should be the same).
 * @param color_padding Color of the padding (default is black).
 * @return Letterboxed image with padding.
 */
Mat Yolov10::letterbox(const Mat& src, const Size& target_size, const Scalar& color_padding){
    // Calculate scale and padding
    float scale = std::min(target_size.width / (float)src.cols, target_size.height / (float)src.rows);
    int new_width = static_cast<int>(src.cols * scale);
    int new_height = static_cast<int>(src.rows * scale);

    // Resize the image with the computed scale
    cv::Mat resized_image;
    cv::resize(src, resized_image, cv::Size(new_width, new_height));

    // Create the output image with the target size and fill it with the padding color
    cv::Mat dst = cv::Mat::zeros(target_size.height, target_size.width, src.type());
    dst.setTo(color_padding);

    // Calculate the top-left corner where the resized image will be placed
    int top = (target_size.height - new_height) / 2;
    int left = (target_size.width - new_width) / 2;

    // Place the resized image onto the center of the letterboxed image
    resized_image.copyTo(dst(cv::Rect(left, top, resized_image.cols, resized_image.rows)));

    return dst;
}

/**
 * @brief Apply Histogram Equalization to an image.
 *
 * @param src Input image in BGR format.
 * @return Image with enhanced contrast.
 */
cv::Mat Yolov10::applyHistogramEqualization(const cv::Mat &src)
{
    cv::Mat ycrcb_image;
    cv::cvtColor(src, ycrcb_image, cv::COLOR_BGR2YCrCb);  // Convert to YCrCb color space

    std::vector<cv::Mat> channels;
    cv::split(ycrcb_image, channels);

    // Apply histogram equalization to the Y channel (intensity)
    cv::equalizeHist(channels[0], channels[0]);

    // Merge back the channels and convert to BGR
    cv::merge(channels, ycrcb_image);
    cv::Mat result;
    cv::cvtColor(ycrcb_image, result, cv::COLOR_YCrCb2BGR);

    return result;
}

/**
 * @brief Apply CLAHE to an image for adaptive contrast enhancement.
 *
 * @param src Input image in BGR format.
 * @return Image with enhanced local contrast.
 */
Mat Yolov10::applyCLAHE(const Mat& src){
    cv::Mat lab_image;
    cv::cvtColor(src, lab_image, cv::COLOR_BGR2Lab);  // Convert to LAB color space

    std::vector<cv::Mat> lab_planes;
    cv::split(lab_image, lab_planes);

    // Apply CLAHE to the L channel (lightness)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);  // Set the clip limit for contrast enhancement
    clahe->apply(lab_planes[0], lab_planes[0]);

    // Merge the planes back and convert to BGR
    cv::merge(lab_planes, lab_image);
    cv::Mat result;
    cv::cvtColor(lab_image, result, cv::COLOR_Lab2BGR);

    return result;
}

/**
 * @brief Apply Gamma Correction to an image.
 *
 * @param src Input image in BGR format.
 * @param gamma Gamma value for correction. Values < 1 will lighten the image, values > 1 will darken it.
 * @return Image with gamma correction applied.
 */
cv::Mat Yolov10::applyGammaCorrection(const cv::Mat &src, float gamma)
{
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* p = lut.ptr();
    for (int i = 0; i < 256; ++i)
    {
        p[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }

    cv::Mat result;
    cv::LUT(src, lut, result);  // Apply the gamma lookup table to the image

    return result;
}

/*
 * Function to preprocess the image
 *
 * @param image: input image as cv::Mat
 * @return: vector of floats representing the preprocessed image
 */
vector<float> Yolov10::preprocess(const Mat& image){
    clock_t start_time = clock();

    if (image.empty())
    {
        throw std::runtime_error("Could not read the image");
    }

    _orig_width = image.cols;
    _orig_height = image.rows;

    // Step 1: Apply image enhancement techniques
    cv::Mat enhanced_image = applyCLAHE(image);  // Use CLAHE as an example
    // cv::Mat enhanced_image = applyHistogramEqualization(image);  // Or use Histogram Equalization
    // cv::Mat enhanced_image = applyGammaCorrection(image, 1.2);  // Or use Gamma Correction

    // Step 2: Apply letterbox to the enhanced image
    cv::Mat letterboxed_image = letterbox(enhanced_image, Size(_input_shape[2], _input_shape[3]), _color_padding);

    // Step 3: Convert image to float and normalize
    letterboxed_image.convertTo(letterboxed_image, CV_32F, 1.0 / 255);

    // Step 4: Convert from BGR to RGB
    cv::cvtColor(letterboxed_image, letterboxed_image, cv::COLOR_BGR2RGB);

    // Step 5: Prepare the input tensor values as a 1D vector
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(_input_shape[1] * _input_shape[2] * _input_shape[3]);

    // Convert Mat to vector of floats (HWC to CHW)
    std::vector<cv::Mat> channels(3);
    cv::split(letterboxed_image, channels);

    for (int c = 0; c < 3; ++c)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float *)channels[c].data, (float *)channels[c].data + _input_shape[2] * _input_shape[3]);
    }

    clock_t end_time = clock();
    _preprocess_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000;


    return input_tensor_values;
}

/*
 * 初始化参数、加载模型文件、并且完成预热
 *
 * @param params: 初始化信息结构体 
 * @return: 报错信息
 */
Yolov10::Yolov10(InitParam& params){
    // 识别路径中是否有中文字符
    regex pattern("[\u4e00-\u9fa5]");
    bool result = regex_search(params.model_path, pattern);
    if(result){
        string ret = "[YOLO_V10]:Your model path is error.Change your model path without chinese characters.";
        cout << ret << endl;
        return;
    }

    _confidence_threshold = params.confidence_threshold;
    _iou_threshold = params.iou_threshold;
    _input_shape = params.input_shape;
    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "second_test");
    Ort::SessionOptions session_options;
    // 使用CUDA加速
    if(params.cuda_enable){
        _cuda_enable = params.cuda_enable;
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cudaOption);
    }
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(params.intra_opNum_threads);
    session_options.SetLogSeverityLevel(params.log_severity_level);

    const char* model_path = params.model_path.c_str();

    // 读取模型文件
    _session = new Ort::Session(_env, model_path, session_options);
    if(!_session){
        throw runtime_error("[YOLOv10-CUDA]: Failed to create ONNX Runtime session.");
    }

    _input_node_names = getInputNodeNames();
    _output_node_names = getOutputNodeNames();
    _options = Ort::RunOptions{nullptr};

    _class_names = params.class_names;
    _nms_sigma = params.nms_sigma;
    warmUpInference();
    return;
}


/*
 * Function to run inference
 *
 * @param input_tensor_values: vector of floats representing the input tensor
 * @return: vector of floats representing the output tensor
 */
vector<float> Yolov10::runInference(const vector<float>& input_tensor_values){
    clock_t start_time = clock();

    const char *input_name_ptr = _input_node_names.c_str();
    const char *output_name_ptr = _output_node_names.c_str();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float *>(input_tensor_values.data()), input_tensor_values.size(), _input_shape.data(), _input_shape.size()
    );
    auto output_tensors = _session->Run(_options, &input_name_ptr, &input_tensor, 1, &output_name_ptr, 1);

    float *floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    clock_t end_time = clock();
    _inference_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000;

    return std::vector<float>(floatarr, floatarr + output_tensor_size);
}

/*
 * Function to get the input name
 *
 * @return: name of the input tensor
 */
string Yolov10::getInputNodeNames(){
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = _session->GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

/*
 * Function to get the output name
 *
 * @return: name of the output tensor
 */
std::string Yolov10::getOutputNodeNames()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = _session->GetOutputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

/*
 * Function to filter the detections based on the confidence threshold
 *
 * @param results: vector of floats representing the output tensor
 * @return: vector of Detection objects
 */
vector<OdResult> Yolov10::postprocess(const std::vector<float> &results){
    clock_t start_time = clock();

    std::vector<OdResult> detections;
    const int num_detections = results.size() / 6;
    
    // Calculate scale and padding factors
    float scale = std::min(_input_shape[2] / (float)_orig_width, _input_shape[3] / (float)_orig_height);
    int new_width = static_cast<int>(_orig_width * scale);
    int new_height = static_cast<int>(_orig_height * scale);
    int pad_x = (_input_shape[2] - new_width) / 2;
    int pad_y = (_input_shape[3] - new_height) / 2;

    detections.reserve(num_detections);
    
    for(int i = 0; i < num_detections; ++i){
        float left = results[i * 6 + 0];
        float top = results[i * 6 + 1];
        float right = results[i * 6 + 2];
        float bottom = results[i * 6 + 3];
        float confidence = results[i * 6 + 4];
        int class_id = static_cast<int>(results[i * 6 + 5]);

        if (confidence >= _confidence_threshold)
        {
            // Remove padding and rescale to original image dimensions
            left = (left - pad_x) / scale;
            top = (top - pad_y) / scale;
            right = (right - pad_x) / scale;
            bottom = (bottom - pad_y) / scale;

            int x = static_cast<int>(left);
            int y = static_cast<int>(top);
            int width = static_cast<int>(right - left);
            int height = static_cast<int>(bottom - top);

            detections.push_back(
                {class_id,
                 confidence,
                 cv::Rect(x, y, width, height),
                 _class_names[class_id]}
            );
        }
    }

    // Apply Soft-NMS to refine detections
    applySoftNMS(detections); // You can tweak the sigma and IoU threshold values as needed

    clock_t end_time = clock();
    double _postprocess_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000;
    return detections;
}

/**
 * @brief Applies Soft-NMS to a set of detected bounding boxes to reduce overlapping detections.
 *
 * @param detections Vector of detections to process.
 */
void Yolov10::applySoftNMS(std::vector<OdResult> &detections){
    for (size_t i = 0; i < detections.size(); ++i)
    {
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            float iou = computeIOU(detections[i].bbox, detections[j].bbox);
            if (iou > _iou_threshold)
            {
                // Apply the Soft-NMS score decay formula
                detections[j].confidence *= std::exp(-iou * iou / _nms_sigma);
            }
        }
    }

    // Remove detections with low confidence scores
    detections.erase(
        std::remove_if(
            detections.begin(), 
            detections.end(),
            [](const OdResult &det) { return det.confidence < 0.001; }
        ),
        detections.end()
    );
}

/**
 * @brief Computes the Intersection over Union (IoU) between two bounding boxes.
 *
 * @param boxA First bounding box.
 * @param boxB Second bounding box.
 * @return IoU value between 0 and 1.
 */
float Yolov10::computeIOU(const cv::Rect &boxA, const cv::Rect &boxB)
{
    int xA = std::max(boxA.x, boxB.x);
    int yA = std::max(boxA.y, boxB.y);
    int xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);

    int boxAArea = boxA.width * boxA.height;
    int boxBArea = boxB.width * boxB.height;

    float iou = static_cast<float>(interArea) / (boxAArea + boxBArea - interArea);
    return iou;
}

// 预先推理一次
void Yolov10::warmUpInference() {
    clock_t start_time = clock();

    cv::Mat tmp_img = cv::Mat(cv::Size(_input_shape.at(2), _input_shape.at(3)), CV_8UC3);
    vector<float> input_tensor_values = preprocess(tmp_img); 
    runInference(input_tensor_values);

    clock_t end_time = clock();
    double warmup_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000;
}

/*
 * Function to draw the labels on the image
 *
 * @param image: input image
 * @param detections: vector of Detection objects
 * @return: image with labels drawn
 */
cv::Mat Yolov10::drawLabels(const cv::Mat &image, const std::vector<OdResult> &detections)
{
    cv::Mat result = image.clone();

    for (const auto &detection : detections)
    {
        cv::rectangle(result, detection.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = detection.class_name + ": " + std::to_string(detection.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(
            result,
            cv::Point(detection.bbox.x, detection.bbox.y - labelSize.height),
            cv::Point(detection.bbox.x + labelSize.width, detection.bbox.y + baseLine),
            cv::Scalar(255, 255, 255),
            cv::FILLED);

        cv::putText(
            result,
            label,
            cv::Point(detection.bbox.x, detection.bbox.y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
    }

    return result;
}

