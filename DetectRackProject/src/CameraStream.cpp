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
    while (running) {
        if (!cap.read(tempFrame)) {
            std::cerr << "[CameraStream] Warning: Failed to grab frame. Reconnecting or ending..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (tempFrame.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            tempFrame.copyTo(currentFrame);
            hasNewFrame = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // throttle to prevent spinning too fast
    }
}
