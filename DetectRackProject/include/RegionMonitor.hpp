#pragma once
#include "YOLOv8Detector.hpp"
#include <opencv2/opencv.hpp>

class RegionMonitor {
private:
  cv::Rect roiBox;
  bool isDrawing;
  cv::Point startPoint;

public:
  RegionMonitor();
  void handleMouseCallback(int event, int x, int y, int flags);
  cv::Rect getROI() const;
  bool checkIntersection(const std::vector<Detection> &detections);
  void drawUI(cv::Mat &frame, const std::vector<Detection> &detections,
              bool isOccupied);
};