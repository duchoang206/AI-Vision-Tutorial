#include "RegionMonitor.hpp"

RegionMonitor::RegionMonitor() : roiBox(0, 0, 0, 0), isDrawing(false) {}

void RegionMonitor::handleMouseCallback(int event, int x, int y, int flags) {
  if (event == cv::EVENT_LBUTTONDOWN) {
    isDrawing = true;
    startPoint = cv::Point(x, y);
    roiBox = cv::Rect(x, y, 0, 0);
  } else if (event == cv::EVENT_MOUSEMOVE && isDrawing) {
    roiBox.x = std::min(startPoint.x, x);
    roiBox.y = std::min(startPoint.y, y);
    roiBox.width = std::abs(startPoint.x - x);
    roiBox.height = std::abs(startPoint.y - y);
  } else if (event == cv::EVENT_LBUTTONUP) {
    isDrawing = false;
  }
}

cv::Rect RegionMonitor::getROI() const { return roiBox; }

bool RegionMonitor::checkIntersection(
    const std::vector<Detection> &detections) {
  if (roiBox.width <= 0 || roiBox.height <= 0)
    return false;

  for (const auto &det : detections) {
    cv::Rect inter = det.box & roiBox;
    if (inter.area() > 0) {
      // Nếu diện tích phần giao cắt vượt quá 25% diện tích vùng vẽ hoặc vùng
      // vật thể
      if ((float)inter.area() / roiBox.area() > 0.25f ||
          (float)inter.area() / det.box.area() > 0.25f) {
        return true;
      }
    }
  }
  return false;
}

void RegionMonitor::drawUI(cv::Mat &frame,
                           const std::vector<Detection> &detections,
                           bool isOccupied) {
  // 1. Vẽ các vật thể AI nhận diện được (Màu xanh lá)
  for (const auto &det : detections) {
    cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
    std::string label =
        "Rack: " + std::to_string(int(det.confidence * 100)) + "%";
    cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }

  // 2. Vẽ trạng thái vùng ROI được cấu hình bằng kéo chuột
  if (roiBox.width > 0 && roiBox.height > 0) {
    if (isOccupied) {
      cv::rectangle(frame, roiBox, cv::Scalar(0, 0, 255),
                    3); // ĐỎ khi có rack xâm nhập
      cv::putText(frame, "STATUS: OCCUPIED (RACK IN ZONE)", cv::Point(30, 40),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    } else {
      cv::rectangle(frame, roiBox, cv::Scalar(0, 255, 255),
                    2); // VÀNG khi an toàn trống trải
      cv::putText(frame, "STATUS: SAFE (EMPTY)", cv::Point(30, 40),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
  } else {
    cv::putText(frame, "Dung chuot trai keo de ve vung can giam sat...",
                cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);
  }
}