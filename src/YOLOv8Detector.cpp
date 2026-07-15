#include "YOLOv8Detector.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp> // NMSBoxes
#include <numeric>

YOLOv8Detector::YOLOv8Detector(const std::string& modelPath, 
                               const std::string& className,
                               const cv::Size& inputSize,
                               float confThreshold, 
                               float nmsThreshold)
    : modelPath(modelPath), targetClassName(className), inputSize(inputSize), 
      confThreshold(confThreshold), nmsThreshold(nmsThreshold) {
    // Populate the first class with the dynamic class name. 
    // If it's a multi-class model, other classes will be auto-generated in postprocess or can be loaded later.
    classNames = {targetClassName};
}

YOLOv8Detector::~YOLOv8Detector() {}

bool YOLOv8Detector::loadModel() {
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputNodes = session->GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(i, allocator);
            inputNames.push_back(inputName.get());
        }
        for(auto& name : inputNames) { inputNodeNames.push_back(name.c_str()); }

        size_t numOutputNodes = session->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(i, allocator);
            outputNames.push_back(outputName.get());
        }
        for(auto& name : outputNames) { outputNodeNames.push_back(name.c_str()); }
        
        std::cout << "[YOLOv8Detector] Successfully loaded model via ONNX Runtime: " << modelPath << std::endl;
        
        // Load class names from labels.txt
        std::string directory = "";
        size_t last_slash = modelPath.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            directory = modelPath.substr(0, last_slash + 1);
        }
        std::string labelsPath = directory + "labels.txt";
        std::ifstream labelsFile(labelsPath);
        if (labelsFile.is_open()) {
            classNames.clear();
            std::string line;
            while (std::getline(labelsFile, line)) {
                if (!line.empty()) {
                    classNames.push_back(line);
                }
            }
            labelsFile.close();
            std::cout << "[YOLOv8Detector] Loaded " << classNames.size() << " classes from " << labelsPath << std::endl;
        } else {
            std::cout << "[YOLOv8Detector] Warning: Could not open " << labelsPath << ". Using default class: " << targetClassName << std::endl;
            classNames = {targetClassName};
        }

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[YOLOv8Detector] ORT Exception loading model: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[YOLOv8Detector] Standard Exception loading model: " << e.what() << std::endl;
        return false;
    }
}

void YOLOv8Detector::preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues) {
    cv::Mat rgbImage;
    cv::cvtColor(frame, rgbImage, cv::COLOR_BGR2RGB);
    
    cv::Mat floatImage;
    rgbImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    
    inputTensorValues.resize(3 * inputSize.width * inputSize.height);
    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i) {
        chw[i] = cv::Mat(inputSize, CV_32FC1, inputTensorValues.data() + i * inputSize.width * inputSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YOLOv8Detector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (!session) {
        std::cerr << "[YOLOv8Detector] Model not loaded!" << std::endl;
        return detections;
    }

    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    cv::Mat letterboxed = letterbox(frame, inputSize, cv::Scalar(114, 114, 114), scale, pad_x, pad_y);

    std::vector<float> inputTensorValues;
    preprocess(letterboxed, inputTensorValues);

    std::vector<int64_t> inputDims = {1, 3, inputSize.height, inputSize.width};
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputDims.data(), inputDims.size());

    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);

    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputTypeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto outputShape = outputTypeInfo.GetShape(); 
    
    int dimensions = static_cast<int>(outputShape[1]);
    int rows = static_cast<int>(outputShape[2]);

    return postprocess(frame, outputData, dimensions, rows, scale, pad_x, pad_y);
}

cv::Mat YOLOv8Detector::letterbox(const cv::Mat& src, cv::Size target_size, cv::Scalar pad_color, float& scale, int& pad_x, int& pad_y) {
    int src_w = src.cols;
    int src_h = src.rows;
    int target_w = target_size.width;
    int target_h = target_size.height;

    scale = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;

    cv::Mat dst(target_size, src.type(), pad_color);
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return dst;
}

std::vector<Detection> YOLOv8Detector::postprocess(const cv::Mat& frame, float* outputData, int dimensions, int rows, float scale, int pad_x, int pad_y) {
    std::vector<Detection> detections;
    
    cv::Mat output(dimensions, rows, CV_32F, outputData);
    cv::Mat transposed;
    cv::transpose(output, transposed);

    int classesCount = dimensions - 4;
    
    static bool printedClasses = false;
    if (!printedClasses) {
        std::cout << "[YOLOv8Detector] Model output dimensions: " << dimensions 
                  << " (classes count: " << classesCount << ")" << std::endl;
        printedClasses = true;
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float* data = transposed.ptr<float>(i);
        
        float* classesScores = data + 4;
        cv::Mat scores(1, classesCount, CV_32FC1, classesScores);
        cv::Point classIdPoint;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

        if (maxClassScore >= confThreshold) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = static_cast<int>((cx - pad_x - 0.5f * w) / scale);
            int top = static_cast<int>((cy - pad_y - 0.5f * h) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            classIds.push_back(classIdPoint.x);
            confidences.push_back(static_cast<float>(maxClassScore));
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsIndices);

    for (int idx : nmsIndices) {
        Detection d;
        d.classId = classIds[idx];
        d.className = (d.classId < classNames.size()) ? classNames[d.classId] : "unknown";
        d.confidence = confidences[idx];
        d.box = boxes[idx];
        detections.push_back(d);
    }

    return detections;
}
