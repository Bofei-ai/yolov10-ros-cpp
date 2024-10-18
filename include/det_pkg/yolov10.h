#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "session/onnxruntime_cxx_api.h"

using namespace std;
using namespace cv;

struct InitParam
{
    string model_path;
    float confidence_threshold = 0.3;
    float iou_threshold = 0.3;
    float nms_sigma = 0.5;
    vector<int64_t> input_shape = {1, 3, 640, 640};
    bool cuda_enable = true;
    int intra_opNum_threads = 1;
    int log_severity_level = 3;
    vector<string> class_names;
};

struct OdResult
{
    int class_id;
    float confidence;
    cv::Rect bbox;
    string class_name;
};

class Yolov10
{
    public:
        Yolov10(InitParam& params);
        ~Yolov10();

    public:      
        vector<float> preprocess(const Mat& image);
        vector<float> runInference(const vector<float>& input_tensor_values);
        vector<OdResult> postprocess(const std::vector<float> &results);
        Mat drawLabels(const Mat& image, const std::vector<OdResult> &detections);
        double getPreprocessTime(){return _preprocess_time;}
        double getInferenceTime(){return _inference_time;}
        double getPostprocessTime(){return _postprocess_time;}
        double getWarmupTime(){return _warmup_time;}

    private:
        Mat letterbox(const Mat& src, const Size& target_size, const Scalar& color_padding);
        Mat applyCLAHE(const Mat& src);
        Mat applyHistogramEqualization(const Mat& src);
        Mat applyGammaCorrection(const Mat& src, float gamma);
        string getInputNodeNames();
        string getOutputNodeNames();
        void warmUpInference();
        void applySoftNMS(std::vector<OdResult> &detections);
        float computeIOU(const Rect& boxA, const Rect& boxB);

        Ort::Env _env;
        Ort::Session* _session;
        bool _cuda_enable;
        Ort::RunOptions _options;
        vector<string> _class_names;

        Scalar _color_padding = Scalar(0, 0, 0);
        float _confidence_threshold;
        float _iou_threshold;
        float _nms_sigma;
        int _orig_width;
        int _orig_height;

        string _input_node_names;
        string  _output_node_names;
        vector<int64_t> _input_shape;

        double _preprocess_time;
        double _inference_time;
        double _postprocess_time;
        double _warmup_time;
};






