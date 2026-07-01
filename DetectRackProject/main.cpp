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
#include "RegionMonitor.hpp"

// Biến chia sẻ đa luồng
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

// Tăng cường độ tương phản cục bộ để làm nổi bật vân kim loại của rack
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

// Luồng nhận diện YOLOv8 chạy ngầm (Thread 2)
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
            cv::Mat enhanced = enhanceContrast(localFrame);
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

// Hàm callback sự kiện chuột chuyển tiếp tới RegionMonitor
void onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* monitor = reinterpret_cast<RegionMonitor*>(userdata);
    if (monitor) {
        monitor->handleMouseCallback(event, x, y, flags);
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

    // Khởi tạo luồng đọc camera (Thread 1)
    CameraStream camera(videoSource);
    if (!camera.start()) return -1;

    // Load YOLOv8 model
    YOLOv8Detector detector(modelPath, cv::Size(640, 640), 0.20f, 0.45f);
    if (!detector.loadModel()) {
        camera.stop();
        return -1;
    }

    // Khởi động luồng xử lý AI (Thread 2)
    std::thread infThread(runInference, &detector);

    // Cấu hình giao diện và RegionMonitor
    std::string winName = "DetectRackProject - Custom ROI Demo";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    RegionMonitor monitor;
    cv::setMouseCallback(winName, onMouse, &monitor);

    cv::Mat frame, outputFrame;
    double fps = 0.0;
    int frameCount = 0;
    auto lastFpsUpdate = std::chrono::steady_clock::now();

    // Vòng lặp giao diện GUI chính
    while (true) {
        if (camera.retrieveFrame(frame)) {
            frame.copyTo(outputFrame);

            // Gửi khung hình thô sang luồng nhận diện ngầm (cực kỳ nhanh, không block GUI)
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_inferenceFrame = frame;
                g_newJob = true;
            }

            // Tính toán FPS hiển thị
            frameCount++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - lastFpsUpdate;
            if (elapsed.count() >= 0.5) {
                fps = frameCount / elapsed.count();
                frameCount = 0;
                lastFpsUpdate = now;
            }

            // Lấy kết quả nhận diện mới nhất
            std::vector<Detection> latestDets;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                latestDets = g_detections;
            }

            // Lọc các phát hiện nằm trong vùng đặt rack hợp lệ
            std::vector<Detection> validDets;
            for (const auto& d : latestDets) {
                if (isValidRackArea(d.box)) {
                    validDets.push_back(d);
                }
            }

            // Kiểm tra va chạm bằng RegionMonitor và vẽ giao diện tùy chỉnh
            bool isOccupied = monitor.checkIntersection(validDets);
            
            // Vẽ hộp giám sát vẽ bằng chuột (Đỏ nếu có rack, Xanh lá nếu an toàn)
            cv::Rect roi = monitor.getROI();
            if (roi.width > 0 && roi.height > 0) {
                cv::Scalar boxColor = isOccupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(outputFrame, roi, boxColor, 2);
                
                std::string statusText = isOccupied ? "STATUS: OCCUPIED (RACK IN ZONE)" : "STATUS: SAFE (EMPTY)";
                cv::Scalar textColor = isOccupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::putText(outputFrame, statusText, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);
            } else {
                cv::putText(outputFrame, "Dung chuot trai keo de ve vung can giam sat...", 
                            cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            }

            // Hiển thị FPS
            if (fps > 0.0) {
                std::string fpsText = cv::format("FPS: %.1f", fps);
                cv::putText(outputFrame, fpsText, cv::Point(16, outputFrame.rows - 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
                cv::putText(outputFrame, fpsText, cv::Point(15, outputFrame.rows - 17), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }

            cv::imshow(winName, outputFrame);
        }

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 'Q' || key == 27) break;
    }

    // Dọn dẹp luồng
    g_running = false;
    if (infThread.joinable()) infThread.join();

    camera.stop();
    cv::destroyAllWindows();
    return 0;
}
