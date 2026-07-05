#include "YOLOv8Detector.hpp"
#include <iostream>
#include <fstream>

YOLOv8Detector::YOLOv8Detector(const std::string& modelPath, 
                               const cv::Size& inputSize,
                               float confThreshold, 
                               float nmsThreshold)
    : modelPath(modelPath), inputSize(inputSize), 
      confThreshold(confThreshold), nmsThreshold(nmsThreshold) {
    // Populate some default classes (e.g. COCO classes) just in case
    classNames = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                  "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
}

YOLOv8Detector::~YOLOv8Detector() {}

bool YOLOv8Detector::loadModel() {
    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        
        // Optional: Try to use CUDA if available, otherwise fallback to CPU
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "[YOLOv8Detector] Successfully loaded model: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[YOLOv8Detector] OpenCV Exception loading model: " << e.what() << std::endl;
        return false;
    }
}

void YOLOv8Detector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    // Mô hình YOLOv8 chuẩn hóa pixel về khoảng [0, 1] (chia cho 255.0)
    // Ảnh 'frame' đưa vào đây đã được letterbox về kích thước 640x640, nên không cần resize thêm.
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, inputSize, cv::Scalar(), true, false);
}

std::vector<Detection> YOLOv8Detector::detect(const cv::Mat& frame) {
    std::vector<Detection> detections;
    if (net.empty()) {
        std::cerr << "[YOLOv8Detector] Model not loaded!" << std::endl;
        return detections;
    }

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    // Thực hiện letterbox ảnh đầu vào về kích thước inputSize (640x640) với màu nền xám (114, 114, 114)
    cv::Mat letterboxed = letterbox(frame, inputSize, cv::Scalar(114, 114, 114), scale, pad_x, pad_y);

    cv::Mat blob;
    preprocess(letterboxed, blob);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return postprocess(frame, outputs, scale, pad_x, pad_y);
}

cv::Mat YOLOv8Detector::letterbox(const cv::Mat& src, cv::Size target_size, cv::Scalar pad_color, float& scale, int& pad_x, int& pad_y) {
    int src_w = src.cols;
    int src_h = src.rows;
    int target_w = target_size.width;
    int target_h = target_size.height;

    // Tính tỉ lệ scale tốt nhất (giữ nguyên aspect ratio)
    scale = std::min((float)target_w / src_w, (float)target_h / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    // Resize ảnh gốc theo scale mới
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    // Tính toán lượng padding cần bù vào mỗi bên (để căn giữa ảnh)
    pad_x = (target_w - new_w) / 2;
    pad_y = (target_h - new_h) / 2;

    // Tạo ảnh đích có kích thước target_size (640x640) và tô màu viền xám
    cv::Mat dst(target_size, src.type(), pad_color);
    // Sao chép ảnh đã resize vào giữa ảnh đích
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return dst;
}

std::vector<Detection> YOLOv8Detector::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, float scale, int pad_x, int pad_y) {
    std::vector<Detection> detections;
    if (outputs.empty()) return detections;

    // YOLOv8 output is usually 1 x (4 + num_classes) x 8400
    cv::Mat output = outputs[0];
    if (output.dims == 3) {
        // Reshape if needed: 3D to 2D matrix of shape [rows = 8400, cols = 4 + num_classes]
        int sizes[] = {output.size[1], output.size[2]};
        output = cv::Mat(2, sizes, CV_32F, output.data);
    }
    
    // We transpose it so we have 8400 rows, each row has [x, y, w, h, class0_conf, class1_conf, ...]
    cv::transpose(output, output);

    int rows = output.rows;
    int dimensions = output.cols;
    int classesCount = dimensions - 4;

    static bool printedClasses = false;
    if (!printedClasses) {
        std::cout << "[YOLOv8Detector] Model output dimensions: " << dimensions 
                  << " (classes count: " << classesCount << ")" << std::endl;
        if (classesCount == 1) {
            classNames = {"rack"};
        } else if (classesCount != 80) {
            classNames.clear();
            for (int i = 0; i < classesCount; ++i) {
                classNames.push_back("class_" + std::to_string(i));
            }
        }
        printedClasses = true;
    }

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
        float* data = output.ptr<float>(i);
        
        // Tìm xác suất phân lớp lớn nhất
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

            // Ánh xạ tọa độ từ không gian letterbox (640x640) về khung hình gốc:
            // 1. Trừ đi lượng padding đã bù vào mỗi bên
            // 2. Chia cho tỉ lệ scale ban đầu
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
