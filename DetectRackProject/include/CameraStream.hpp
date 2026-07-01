#pragma once
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <string>

class CameraStream {
private:
    std::string videoSource;
    cv::VideoCapture cap;
    cv::Mat currentFrame;
    std::mutex frameMutex;
    std::thread captureThread;
    bool running;
    bool hasNewFrame;
    void captureLoop();

public:
    CameraStream(const std::string& source);
    ~CameraStream();
    bool start();
    void stop();
    bool isOpened() const;
    bool retrieveFrame(cv::Mat& frame);
};