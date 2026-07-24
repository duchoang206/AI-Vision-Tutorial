#include "CameraStream.hpp"
#include <iostream>

CameraStream::CameraStream(const std::string& source)
    : videoSource(source), running(false), hasNewFrame(false) {}

CameraStream::~CameraStream() {
    stop();
}

bool CameraStream::start() {
    if (running) return true;

    bool isInt = true;
    for (char c : videoSource) {
        if (c < '0' || c > '9') {
            isInt = false;
            break;
        }
    }

    if (isInt && !videoSource.empty()) {
        cap.open(std::stoi(videoSource));
    } else {
        cap.open(videoSource);
    }
    
    // Đặt bộ đệm OpenCV = 1 để tránh tích tụ frame cũ khi chạy 24/7 (gây trễ hoặc đầy RAM)
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) {
        std::cerr << "[CameraStream] Error: Could not open video source: " << videoSource << std::endl;
        return false;
    }

    running = true;
    captureThread = std::thread(&CameraStream::captureLoop, this);
    return true;
}

void CameraStream::stop() {
    running = false;
    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (cap.isOpened()) {
        cap.release();
    }
}

bool CameraStream::isOpened() const {
    return cap.isOpened();
}

bool CameraStream::retrieveFrame(cv::Mat& frame) {
    if (!hasNewFrame) return false;

    std::lock_guard<std::mutex> lock(frameMutex);
    currentFrame.copyTo(frame);
    hasNewFrame = false;
    return true;
}

void CameraStream::captureLoop() {
    cv::Mat tempFrame;
    int failCount = 0;
    while (running) {
        if (!cap.read(tempFrame) || tempFrame.empty()) {
            failCount++;
            std::cerr << "[CameraStream] Warning: Failed to grab frame. Fail count: " << failCount << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Nếu mất kết nối quá 3 giây (30 frames x 100ms), thử kết nối lại
            if (failCount > 30) {
                std::cerr << "[CameraStream] Attempting to reconnect to stream..." << std::endl;
                cap.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                
                bool isInt = true;
                for (char c : videoSource) {
                    if (c < '0' || c > '9') {
                        isInt = false;
                        break;
                    }
                }

                if (isInt && !videoSource.empty()) {
                    cap.open(std::stoi(videoSource));
                } else {
                    cap.open(videoSource);
                }
                
                cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                
                if (cap.isOpened()) {
                    std::cerr << "[CameraStream] Reconnected successfully." << std::endl;
                    failCount = 0; // Đặt lại bộ đếm khi kết nối thành công
                } else {
                    std::cerr << "[CameraStream] Reconnect failed. Will try again." << std::endl;
                    // Reset 1 phần failCount để không gọi release liên tục mà delay 3 giây nữa
                    failCount = 0; 
                }
            }
            continue;
        }

        failCount = 0;

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            tempFrame.copyTo(currentFrame);
            hasNewFrame = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // throttle to prevent spinning too fast
    }
}
