#pragma once
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

class CameraStream {
private:
  std::string rtspUrl;
  cv::VideoCapture cap;
  std::queue<cv::Mat> frameQueue;
  std::mutex mtx;
  std::thread capThread;
  bool isRunning;
  void update();

public:
  CameraStream(const std::string &url);
  ~CameraStream();
  void start();
  void stop();
  bool getLatestFrame(cv::Mat &frame);
};