#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection {
    int classId;
    std::string className;
    float confidence;
    cv::Rect box;
};

class YOLOv8Detector {
public:
    YOLOv8Detector(const std::string& modelPath, 
                   const cv::Size& inputSize = cv::Size(640, 640),
                   float confThreshold = 0.25f, 
                   float nmsThreshold = 0.45f);
    ~YOLOv8Detector();

    bool loadModel();
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    std::string modelPath;
    cv::Size inputSize;
    float confThreshold;
    float nmsThreshold;
    
    cv::dnn::Net net;
    std::vector<std::string> classNames;

    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    std::vector<Detection> postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, float scale, int pad_x, int pad_y);
    cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, cv::Scalar pad_color, float& scale, int& pad_x, int& pad_y);
};
