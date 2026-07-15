#pragma once
#include "YOLOv8Detector.hpp"
#include <opencv2/opencv.hpp>

class RegionMonitor {
private:
  cv::Rect roiBox;
  bool isDrawing;
  cv::Point startPoint;
  float intersectionThreshold;

public:
  RegionMonitor(float intersectionThreshold = 0.25f);
  void handleMouseCallback(int event, int x, int y, int flags);
  cv::Rect getROI() const;
  bool checkIntersection(const std::vector<Detection> &detections);
  void drawUI(cv::Mat &frame, const std::vector<Detection> &detections,
              bool isOccupied, const std::string& className);
};