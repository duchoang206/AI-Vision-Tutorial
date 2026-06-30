#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

// Global variables for mouse drawing simulation
cv::Rect g_drawn_box;
bool g_is_drawing = false;
bool g_has_box = false;
cv::Size g_frame_size(1, 1);

// Mouse callback function (Left click to draw simulated box, Right click to print coordinates)
void onMouse(int event, int x, int y, int flags, void* userdata) {
    int width = g_frame_size.width;
    int height = g_frame_size.height;

    if (event == cv::EVENT_LBUTTONDOWN) {
        g_is_drawing = true;
        g_drawn_box = cv::Rect(x, y, 0, 0);
        g_has_box = false;
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (g_is_drawing) {
            g_drawn_box.width = x - g_drawn_box.x;
            g_drawn_box.height = y - g_drawn_box.y;
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        g_is_drawing = false;
        if (g_drawn_box.width < 0) {
            g_drawn_box.x += g_drawn_box.width;
            g_drawn_box.width = -g_drawn_box.width;
        }
        if (g_drawn_box.height < 0) {
            g_drawn_box.y += g_drawn_box.height;
            g_drawn_box.height = -g_drawn_box.height;
        }
        if (g_drawn_box.width > 5 && g_drawn_box.height > 5) {
            g_has_box = true;
        }
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        double norm_x = (double)x / width;
        double norm_y = (double)y / height;
        std::cout << "Clicked Point: Pixels=(" << x << ", " << y 
                  << ") | Code Format: cv::Point(width * " << norm_x << ", height * " << norm_y << ")" << std::endl;
    }
}

// Structure to store YOLO detection results
struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

// YOLOv8 inference function using OpenCV DNN
std::vector<Detection> runYOLOv8(cv::dnn::Net& net, const cv::Mat& frame, float conf_threshold, float nms_threshold, int num_classes = 1) {
    std::vector<Detection> detections;

    // YOLOv8 input size is 640x640, normalizes by dividing by 255.0, swapRB=true
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Output shape of YOLOv8 is [1, 4 + num_classes, 8400]
    cv::Mat output = outputs[0];
    int rows = output.size[1]; // 4 bounding box coords + num_classes
    int cols = output.size[2]; // 8400 candidate anchors

    // Reshape and transpose to shape [8400, 4 + num_classes]
    cv::Mat output_reshaped = output.reshape(1, rows);
    cv::Mat output_transposed;
    cv::transpose(output_reshaped, output_transposed);

    float* data = (float*)output_transposed.data;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float x_factor = (float)frame.cols / 640.0f;
    float y_factor = (float)frame.rows / 640.0f;

    for (int i = 0; i < cols; ++i) {
        float* row = data + i * rows;
        
        // Find class with highest score
        float max_score = 0;
        int class_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = row[4 + c];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= conf_threshold) {
            float x_center = row[0];
            float y_center = row[1];
            float w = row[2];
            float h = row[3];

            // Convert normalized box to frame size
            int x = static_cast<int>((x_center - w / 2.0f) * x_factor);
            int y = static_cast<int>((y_center - h / 2.0f) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(x, y, width, height));
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    // Run Non-Maximum Suppression to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        Detection d;
        d.box = boxes[idx];
        d.confidence = confidences[idx];
        d.class_id = class_ids[idx];
        detections.push_back(d);
    }

    return detections;
}

int main() {
    std::cout << "=== RTSP Camera AI Object & Custom ROI Detection ===" << std::endl;

    std::string onnx_model_path = "weights/best.onnx";
    std::string rtsp_url = "rtsp://admin:rtc%402025@192.168.5.201:554/cam/realmonitor?channel=1&subtype=0";

    // 1. Try to load YOLO ONNX model
    cv::dnn::Net net;
    bool is_ai_mode = false;
    try {
        net = cv::dnn::readNetFromONNX(onnx_model_path);
        // Try enabling GPU acceleration if available
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Change to DNN_TARGET_CUDA if GPU configured
        is_ai_mode = true;
        std::cout << "SUCCESS: Loaded YOLO ONNX model from: " << onnx_model_path << std::endl;
        std::cout << "Running in AUTOMATIC AI detection mode." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "WARNING: Could not load " << onnx_model_path << " (Error: " << e.what() << ")" << std::endl;
        std::cout << "Running in INTERACTIVE SIMULATION mode (Left-click & drag mouse to draw a box)..." << std::endl;
    }

    // Connect to RTSP stream
    cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to connect to RTSP stream!" << std::endl;
        return -1;
    }

    std::string window_name = "RTSP Real-time Stream & ROI Detection";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name, onMouse, nullptr);

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Warning: Failed to grab frame, camera offline..." << std::endl;
            cv::waitKey(100);
            continue;
        }

        g_frame_size = frame.size();
        int width = frame.cols;
        int height = frame.rows;

        // Custom ROI points focused on the yellow-and-black floor markings under the metal stand
        std::vector<cv::Point> roi_points;
        roi_points.push_back(cv::Point(width * 0.605, height * 0.360)); // Top-Left
        roi_points.push_back(cv::Point(width * 0.750, height * 0.400)); // Top-Right
        roi_points.push_back(cv::Point(width * 0.810, height * 0.540)); // Bottom-Right
        roi_points.push_back(cv::Point(width * 0.680, height * 0.440)); // Bottom-Left

        std::vector<cv::Point> check_points; // Points to check inside ROI
        std::vector<cv::Rect> draw_boxes;    // Bounding boxes to draw
        std::vector<std::string> labels;     // Label text for boxes

        if (is_ai_mode) {
            // Run real YOLOv8 object detection
            // confidence_threshold = 0.35, nms_threshold = 0.45
            std::vector<Detection> detections = runYOLOv8(net, frame, 0.35f, 0.45f, 1);
            for (const auto& det : detections) {
                draw_boxes.push_back(det.box);
                
                // Calculate bottom-center for floor region checking
                cv::Point bp(det.box.x + det.box.width / 2, det.box.y + det.box.height);
                check_points.push_back(bp);

                // Create label
                char label_buf[100];
                snprintf(label_buf, sizeof(label_buf), "Metal Rack %.2f", det.confidence);
                labels.push_back(std::string(label_buf));
            }
        } else {
            // Interactive mouse simulation fallback mode
            if (g_is_drawing || g_has_box) {
                draw_boxes.push_back(g_drawn_box);
                
                cv::Point bp(g_drawn_box.x + g_drawn_box.width / 2, g_drawn_box.y + g_drawn_box.height);
                check_points.push_back(bp);
                labels.push_back("Simulated BBox");
            }
        }

        // Run ROI Point Polygon check for all active points
        bool is_inside = false;
        for (const auto& pt : check_points) {
            double test = cv::pointPolygonTest(roi_points, cv::Point2f(pt.x, pt.y), false);
            if (test >= 0) {
                is_inside = true;
                break; // Trigger alert if any object lies in the zone
            }
        }

        // Render the ROI Zone (Red if object inside, Green if safe)
        cv::Scalar roi_color = is_inside ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::Mat overlay = frame.clone();
        std::vector<std::vector<cv::Point>> polygons = { roi_points };
        
        cv::fillPoly(overlay, polygons, roi_color);
        cv::addWeighted(overlay, 0.25, frame, 0.75, 0, frame);
        cv::polylines(frame, polygons, true, roi_color, 3);

        // Draw Bounding Boxes and Reference Points
        for (size_t i = 0; i < draw_boxes.size(); ++i) {
            cv::rectangle(frame, draw_boxes[i], cv::Scalar(255, 255, 0), 2);
            cv::circle(frame, check_points[i], 6, cv::Scalar(0, 0, 255), -1);

            // Draw text labels
            cv::putText(frame, labels[i], 
                        cv::Point(draw_boxes[i].x, draw_boxes[i].y - 8), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
            cv::putText(frame, "Ref Point", 
                        cv::Point(check_points[i].x + 10, check_points[i].y - 10), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        }

        // Display general status on screen
        std::string status_text = "STATUS: SAFE (No stand detected in zone)";
        if (is_inside) {
            status_text = "WARNING: Metal Rack detected in Zone!";
        }
        cv::putText(frame, status_text, cv::Point(30, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, roi_color, 3);

        // Helper label at the bottom
        std::string mode_label = is_ai_mode ? "AI Mode: Running YOLOv8 detection" : "Simulation Mode (No best.onnx found). Right-click to calibrate.";
        cv::putText(frame, mode_label, cv::Point(15, height - 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

        // Display frame
        cv::imshow(window_name, frame);

        // Press ESC to exit
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Program closed." << std::endl;
    return 0;
}
