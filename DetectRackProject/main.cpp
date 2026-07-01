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
    // Tránh khu vực di chuyển của AGV bên trái (x < 700) và vạch kẻ sọc phía trước (y + h > 600)
    return box.x >= 700 && (box.y + box.height) <= 600;
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

// Hàm gửi báo cáo lên Server khi trạng thái thay đổi
void reportToServer(const std::string& roiName, bool hasRack) {
    // Hàm mẫu để bạn kết nối Socket, HTTP API hoặc Modbus TCP gửi dữ liệu về Server sau này
    std::cout << "\n[SERVER REPORT] " << roiName << " state changed: " 
              << (hasRack ? "RACK DETECTED (OCCUPIED)" : "NO RACK (EMPTY)") << std::endl;
}

// Cấu trúc dùng làm công cụ phụ trợ lấy tọa độ mới bằng chuột
struct MouseCallbackParams {
    cv::Rect box;
    bool drawing = false;
};

void onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* params = reinterpret_cast<MouseCallbackParams*>(userdata);
    if (!params) return;
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
        std::cout << "[Config Helper] Drawn Box: cv::Rect(" 
                  << params->box.x << ", " << params->box.y << ", " 
                  << params->box.width << ", " << params->box.height << ")" << std::endl;
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

    // Cấu hình giao diện GUI
    std::string winName = "DetectRackProject - Predefined ROI Demo";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

    // Đăng ký chuột để làm công cụ phụ lấy tọa độ khi cần cấu hình lại
    MouseCallbackParams mouseParams;
    cv::setMouseCallback(winName, onMouse, &mouseParams);

    // Định cấu hình 2 vùng ROI giám sát mặc định
    cv::Rect roiRect1(1160, 390, 390, 200); // ROI 1 (phía phải dưới camera)
    cv::Rect roiRect2(760, 390, 380, 200);  // ROI 2 (ở giữa cạnh ROI 1)

    // Khởi tạo 2 monitor tương ứng để quản lý va chạm
    RegionMonitor monitor1;
    monitor1.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect1.x, roiRect1.y, 0);
    monitor1.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect1.x + roiRect1.width, roiRect1.y + roiRect1.height, 0);
    monitor1.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect1.x + roiRect1.width, roiRect1.y + roiRect1.height, 0);

    RegionMonitor monitor2;
    monitor2.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect2.x, roiRect2.y, 0);
    monitor2.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect2.x + roiRect2.width, roiRect2.y + roiRect2.height, 0);
    monitor2.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect2.x + roiRect2.width, roiRect2.y + roiRect2.height, 0);

    // Lưu trạng thái trước đó để chỉ gửi báo cáo khi trạng thái thay đổi
    bool prevOccupied1 = false;
    bool prevOccupied2 = false;

    cv::Mat frame, outputFrame;
    double fps = 0.0;
    int frameCount = 0;
    auto lastFpsUpdate = std::chrono::steady_clock::now();

    // Vòng lặp giao diện GUI chính
    while (true) {
        if (camera.retrieveFrame(frame)) {
            frame.copyTo(outputFrame);

            // Gửi khung hình thô sang luồng nhận diện ngầm
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

            // Kiểm tra va chạm cho từng ROI sử dụng đối tượng RegionMonitor tương ứng
            bool isOccupied1 = monitor1.checkIntersection(validDets);
            bool isOccupied2 = monitor2.checkIntersection(validDets);

            // Báo cáo thay đổi trạng thái lên Server
            if (isOccupied1 != prevOccupied1) {
                reportToServer("ROI1", isOccupied1);
                prevOccupied1 = isOccupied1;
            }
            if (isOccupied2 != prevOccupied2) {
                reportToServer("ROI2", isOccupied2);
                prevOccupied2 = isOccupied2;
            }

            // Vẽ hộp ROI 1 (Đỏ nếu có rack, Xanh lá nếu trống)
            cv::Rect r1 = monitor1.getROI();
            cv::Scalar color1 = isOccupied1 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(outputFrame, r1, color1, 2);
            cv::putText(outputFrame, "ROI 1", cv::Point(r1.x + 5, r1.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color1, 2);

            // Vẽ hộp ROI 2 (Đỏ nếu có rack, Xanh lá nếu trống)
            cv::Rect r2 = monitor2.getROI();
            cv::Scalar color2 = isOccupied2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(outputFrame, r2, color2, 2);
            cv::putText(outputFrame, "ROI 2", cv::Point(r2.x + 5, r2.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, color2, 2);

            // Hiển thị trạng thái của cả 2 ROI ở góc trên bên trái
            std::string statusText1 = "ROI 1: " + std::string(isOccupied1 ? "OCCUPIED (RACK IN ZONE)" : "SAFE (EMPTY)");
            std::string statusText2 = "ROI 2: " + std::string(isOccupied2 ? "OCCUPIED (RACK IN ZONE)" : "SAFE (EMPTY)");
            cv::putText(outputFrame, statusText1, cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, color1, 2);
            cv::putText(outputFrame, statusText2, cv::Point(30, 70), cv::FONT_HERSHEY_SIMPLEX, 0.75, color2, 2);

            // Vẽ hộp đang vẽ dở bằng chuột (Công cụ đo đạc phụ trợ)
            if (mouseParams.drawing || (mouseParams.box.width > 0 && mouseParams.box.height > 0)) {
                cv::rectangle(outputFrame, mouseParams.box, cv::Scalar(0, 255, 255), 1, cv::LINE_8);
                cv::putText(outputFrame, "Temp Box", cv::Point(mouseParams.box.x, mouseParams.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
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
