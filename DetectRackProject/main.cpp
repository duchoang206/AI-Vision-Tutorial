#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "CameraStream.hpp"
#include "YOLOv8Detector.hpp"

// Cấu trúc lưu trữ thông tin vẽ hộp bằng chuột
struct MouseCallbackParams {
    cv::Rect box;
    bool drawing = false;
};

// Hàm callback sự kiện chuột
void onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* params = reinterpret_cast<MouseCallbackParams*>(userdata);
    if (event == cv::EVENT_LBUTTONDOWN) {
        params->drawing = true;
        params->box = cv::Rect(x, y, 0, 0);
    } else if (event == cv::EVENT_MOUSEMOVE && params->drawing) {
        params->box.width = x - params->box.x;
        params->box.height = y - params->box.y;
    } else if (event == cv::EVENT_LBUTTONUP && params->drawing) {
        params->drawing = false;
        if (params->box.width < 0) {
            params->box.x += params->box.width;
            params->box.width = -params->box.width;
        }
        if (params->box.height < 0) {
            params->box.y += params->box.height;
            params->box.height = -params->box.height;
        }
    }
}

// Biến chia sẻ đa luồng giữa GUI và Inference
std::mutex g_mutex;
cv::Mat g_inferenceFrame;
bool g_newJob = false;
std::vector<Detection> g_detections;
std::atomic<bool> g_running(true);

// Kiểm tra xem vị trí phát hiện có nằm trong phân khu đặt rack hợp lệ hay không
bool isValidRackArea(const cv::Rect& box) {
    // Tránh khu vực di chuyển của AGV bên trái (x < 750) và vạch kẻ sọc phía trước (y + h > 600)
    return box.x >= 750 && (box.y + box.height) <= 600;
}

// Tăng cường độ tương phản cục bộ giúp làm nổi bật vân kim loại của rack
cv::Mat enhanceContrast(const cv::Mat& src) {
    if (src.empty()) return src;
    cv::Mat lab, dst;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> planes(3);
    cv::split(lab, planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
    clahe->apply(planes[0], planes[0]);
    cv::merge(planes, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    return dst;
}

// Luồng nhận diện YOLOv8 chạy ngầm trên toàn bộ khung hình (Full Frame)
void runInference(YOLOv8Detector* detector) {
    cv::Mat localFrame;
    while (g_running) {
        bool process = false;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_newJob) {
                localFrame = g_inferenceFrame.clone();
                g_newJob = false;
                process = true;
            }
        }

        if (process && !localFrame.empty()) {
            // Thực hiện tăng tương phản ảnh gốc ở luồng ngầm để tránh block luồng GUI
            cv::Mat enhanced = enhanceContrast(localFrame);
            
            // Chạy nhận diện YOLOv8 trên toàn bộ khung hình đã tiền xử lý
            std::vector<Detection> dets = detector->detect(enhanced);
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_detections = dets;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

int main(int argc, char* argv[]) {
    std::string videoSource = "rtsp://admin:rtc%402025@192.168.5.201:554/cam/realmonitor?channel=1&subtype=0";
    std::string modelPath = "weights/best.onnx";
    if (argc > 1) videoSource = argv[1];
    if (argc > 2) modelPath = argv[2];
    if (!std::filesystem::exists(modelPath) && std::filesystem::exists("../" + modelPath)) {
        modelPath = "../" + modelPath;
    }

    CameraStream camera(videoSource);
    if (!camera.start()) return -1;

    // Load YOLOv8 model
    YOLOv8Detector detector(modelPath, cv::Size(640, 640), 0.20f, 0.45f);
    if (!detector.loadModel()) {
        camera.stop();
        return -1;
    }

    std::thread infThread(runInference, &detector);
    cv::Mat frame, outputFrame;
    std::string winName = "DetectRackProject - Custom ROI Demo";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    MouseCallbackParams mouseParams;
    cv::setMouseCallback(winName, onMouse, &mouseParams);

    double fps = 0.0;
    int frameCount = 0;
    auto lastFpsUpdate = std::chrono::steady_clock::now();

    while (true) {
        if (camera.retrieveFrame(frame)) {
            frame.copyTo(outputFrame);
            cv::Rect userBox = mouseParams.box;

            // Gửi khung hình thô sang luồng nhận diện ngầm (cực kỳ nhanh, không block GUI)
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_inferenceFrame = frame;
                g_newJob = true;
            }

            // Tính toán FPS của luồng hiển thị giao diện
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - lastFpsUpdate;
            if (elapsed.count() >= 0.5) {
                fps = frameCount / elapsed.count();
                frameCount = 0;
                lastFpsUpdate = now;
            }

            // Lấy danh sách kết quả nhận diện từ luồng ngầm
            std::vector<Detection> rawDets;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                rawDets = g_detections;
            }

            // Lọc các phát hiện nằm trong vùng đặt rack hợp lệ
            std::vector<Detection> validDets;
            for (const auto& d : rawDets) {
                if (isValidRackArea(d.box)) {
                    validDets.push_back(d);
                }
            }

            // Kiểm tra giao cắt giữa hộp vẽ của người dùng và các phát hiện hợp lệ
            bool detected = false;
            if (userBox.width > 10 && userBox.height > 10) {
                for (const auto& d : validDets) {
                    cv::Rect intersection = userBox & d.box;
                    if (intersection.area() > 0 && (double)intersection.area() / d.box.area() >= 0.40) {
                        detected = true;
                        break;
                    }
                }

                // Vẽ hộp của người dùng (Xanh lá nếu có rack thật, Vàng nếu không có)
                cv::Scalar boxColor = detected ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
                cv::rectangle(outputFrame, userBox, boxColor, 2);

                if (detected) {
                    std::string txt = "Phat hien duoc rack!";
                    int baseLine = 0;
                    cv::Size sz = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseLine);
                    cv::rectangle(outputFrame, cv::Point(userBox.x, userBox.y - sz.height - 8),
                                  cv::Point(userBox.x + sz.width + 10, userBox.y), cv::Scalar(0, 255, 0), cv::FILLED);
                    cv::putText(outputFrame, txt, cv::Point(userBox.x + 5, userBox.y - 4),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
                }
            }

            // Hiển thị FPS lên màn hình
            if (fps > 0.0) {
                std::string fpsText = cv::format("FPS: %.1f", fps);
                cv::putText(outputFrame, fpsText, cv::Point(16, 36), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
                cv::putText(outputFrame, fpsText, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }

            cv::imshow(winName, outputFrame);
        }

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    g_running = false;
    if (infThread.joinable()) infThread.join();

    camera.stop();
    cv::destroyAllWindows();
    return 0;
}
