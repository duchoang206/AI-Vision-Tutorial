#include "CameraStream.hpp"
#include "RegionMonitor.hpp"
#include "YOLOv8Detector.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <modbus/modbus.h>

#ifdef _WIN32
#include <winsock2.h>
#endif

// ============================================================================
// ============================================================================

struct AppConfig {
    std::string weights_path;
    std::string class_name;
    float confidence_threshold;
    float nms_threshold;
    
    float intersection_threshold;
    float polygon_intersection_threshold;
    int ignore_left_x;
    int ignore_bottom_y;

    std::vector<cv::Point> roi_1;
    std::vector<cv::Point> roi_2;
    
    std::string camera_source;
};

AppConfig g_config;

const int MODBUS_PORT = 502; 
const int WEBSOCKET_PORT = 8082;

// ============================================================================

// --- Trạng thái vi phạm ROI toàn cục ---
std::mutex g_alarmMutex;
bool roi_1_alarm = false; 
bool roi_2_alarm = false;

std::mutex g_detsMutex;
std::vector<Detection> g_latestDets;

// --- Lịch sử phát hiện Rack ---
std::mutex g_historyMutex;
std::vector<std::string> g_entryHistory;

// --- Vùng đệm chia sẻ (Camera -> AI & WS) ---
std::mutex g_bufferMutex;
cv::Mat g_sharedFrame;
bool g_hasNewFrame = false; 

// --- Vùng đệm hiển thị GUI OpenCV cục bộ ---
std::mutex g_guiMutex;
cv::Mat g_guiFrame;
bool g_guiNewFrame = false;

// --- Quản lý các kết nối Web Clients ---
std::mutex g_wsClientsMutex;
std::set<std::shared_ptr<ix::WebSocket>> g_wsClients;

std::atomic<bool> g_running(true);

std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
#ifdef _WIN32
  struct tm buf;
  localtime_s(&buf, &in_time_t);
  ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
#else
  struct tm buf;
  localtime_r(&in_time_t, &buf);
  ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
#endif
  return ss.str();
}

bool isValidRackArea(const cv::Rect &box) {
  if (g_config.ignore_left_x == 0 && g_config.ignore_bottom_y == 0) return true;
  return box.x >= g_config.ignore_left_x && (box.y + box.height) <= g_config.ignore_bottom_y;
}

cv::Mat enhanceContrast(const cv::Mat &src) {
  if (src.empty())
    return src;
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

void reportToServer(const std::string &roiName, bool hasObject) {
  std::string upperClass = g_config.class_name;
  std::transform(upperClass.begin(), upperClass.end(), upperClass.begin(), ::toupper);
  std::cout << "[REPORT] " << roiName << " thay đổi trạng thái: "
            << (hasObject ? "CÓ " + upperClass + " (OCCUPIED)" : "TRỐNG (EMPTY)") << std::endl;
}

bool checkPolygonIntersection(const std::vector<cv::Point> &polygon,
                              const std::vector<Detection> &detections) {
  if (polygon.empty() || detections.empty())
    return false;

  std::vector<cv::Point2f> polyF;
  polyF.reserve(polygon.size());
  for (const auto &p : polygon)
    polyF.push_back(cv::Point2f(p.x, p.y));

  double polyArea = cv::contourArea(polyF);
  if (polyArea <= 0)
    return false;

  for (const auto &det : detections) {
    cv::Point bottomCenter(det.box.x + det.box.width / 2,
                           det.box.y + det.box.height);
    if (cv::pointPolygonTest(
            polygon, cv::Point2f(bottomCenter.x, bottomCenter.y), false) >= 0) {
      return true;
    }

    // Đọc cả 4 chân ở 4 viền của ROI (Đã comment lại vì dễ gây báo động nhầm khi bounding box quá to)
    // for (const auto &p : polygon) {
    //   if (det.box.contains(p)) {
    //     return true;
    //   }
    // }

    std::vector<cv::Point2f> rectF = {
        cv::Point2f(det.box.x, det.box.y),
        cv::Point2f(det.box.x + det.box.width, det.box.y),
        cv::Point2f(det.box.x + det.box.width, det.box.y + det.box.height),
        cv::Point2f(det.box.x, det.box.y + det.box.height)};
    std::vector<cv::Point2f> intersection;
    float intersectArea =
        cv::intersectConvexConvex(polyF, rectF, intersection, true);
    if ((intersectArea / polyArea) > g_config.polygon_intersection_threshold) {
      return true;
    }
  }
  return false;
}

// ==========================================
// ==========================================
void cameraThreadFunc(CameraStream *camera) {
  cv::Mat tempFrame;
  while (g_running) {
    if (camera->retrieveFrame(tempFrame)) {
      if (!tempFrame.empty()) {
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        g_sharedFrame = tempFrame.clone();
        g_hasNewFrame = true;
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
}

// ==========================================
// ==========================================
void aiCoreThreadFunc(YOLOv8Detector *detector, RegionMonitor *monitor1,
                      RegionMonitor *monitor2) {
  cv::Mat localFrame;
  bool prevOccupied1 = false;
  bool prevOccupied2 = false;
  int emptyFrames1 = 0;
  int emptyFrames2 = 0;

  const int releaseFrameThreshold = 3;

  while (g_running) {
    bool process = false;
    {
      std::lock_guard<std::mutex> lock(g_bufferMutex);
      if (g_hasNewFrame) {
        localFrame = g_sharedFrame.clone();
        g_hasNewFrame = false;
        process = true;
      }
    }

    if (process && !localFrame.empty()) {
      cv::Mat enhanced = enhanceContrast(localFrame);
      std::vector<Detection> dets = detector->detect(enhanced);

      std::vector<Detection> validDets;
      for (const auto &d : dets) {
        if (isValidRackArea(d.box)) {
          validDets.push_back(d);
        }
      }

      bool rawOccupied1 = checkPolygonIntersection(g_config.roi_1, validDets);
      bool rawOccupied2 = checkPolygonIntersection(g_config.roi_2, validDets);

      bool isOccupied1 = prevOccupied1;
      bool isOccupied2 = prevOccupied2;

      if (rawOccupied1) {
        emptyFrames1 = 0;
        isOccupied1 = true;
      } else {
        if (++emptyFrames1 >= releaseFrameThreshold) {
          isOccupied1 = false;
        }
      }

      if (rawOccupied2) {
        emptyFrames2 = 0;
        isOccupied2 = true;
      } else {
        if (++emptyFrames2 >= releaseFrameThreshold) {
          isOccupied2 = false;
        }
      }

      monitor1->checkIntersection(validDets);
      monitor2->checkIntersection(validDets);

      if (isOccupied1 != prevOccupied1) {
        reportToServer("ROI1", isOccupied1);
        if (isOccupied1) {
          std::lock_guard<std::mutex> lock(g_historyMutex);
          if (g_entryHistory.size() >= 20) {
            g_entryHistory.erase(g_entryHistory.begin());
          }
          g_entryHistory.push_back("ROI 1 | " + getCurrentTimestamp());
        }
        prevOccupied1 = isOccupied1;
      }
      if (isOccupied2 != prevOccupied2) {
        reportToServer("ROI2", isOccupied2);
        if (isOccupied2) {
          std::lock_guard<std::mutex> lock(g_historyMutex);
          if (g_entryHistory.size() >= 20) {
            g_entryHistory.erase(g_entryHistory.begin());
          }
          g_entryHistory.push_back("ROI 2 | " + getCurrentTimestamp());
        }
        prevOccupied2 = isOccupied2;
      }

      {
        std::lock_guard<std::mutex> lock(g_alarmMutex);
        roi_1_alarm = isOccupied1;
        roi_2_alarm = isOccupied2;
      }

      {
        std::lock_guard<std::mutex> lock(g_detsMutex);
        g_latestDets = validDets;
      }

      {
        std::lock_guard<std::mutex> lock(g_guiMutex);
        g_guiFrame = localFrame.clone();
        g_guiNewFrame = true;
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

// ==========================================
// ==========================================
void websocketThreadFunc() {
  ix::WebSocketServer server(WEBSOCKET_PORT, "0.0.0.0");

  server.setOnConnectionCallback(
      [](std::weak_ptr<ix::WebSocket> webSocket,
         std::shared_ptr<ix::ConnectionState> connectionState) {
        auto ws = webSocket.lock();
        if (ws) {
          {
            std::lock_guard<std::mutex> lock(g_wsClientsMutex);
            g_wsClients.insert(ws);
          }
          std::cout << "[WebSocket] Thiết bị kết nối: "
                    << connectionState->getRemoteIp() << std::endl;

          ws->setOnMessageCallback([webSocket](
                                       const ix::WebSocketMessagePtr &msg) {
            if (msg->type == ix::WebSocketMessageType::Close ||
                msg->type == ix::WebSocketMessageType::Error) {
              std::lock_guard<std::mutex> lock(g_wsClientsMutex);
              auto wsShared = webSocket.lock();
              if (wsShared)
                g_wsClients.erase(wsShared);
              std::cout << "[WebSocket] Thiết bị ngắt kết nối." << std::endl;
            }
          });
        }
      });

  auto res = server.listen();
  if (!res.first) {
    std::cerr << "[WebSocket] Lỗi khởi động server: " << res.second
              << std::endl;
    return;
  }

  server.start();

  double streamFps = 0.0;
  int frameCount = 0;
  auto lastFpsUpdate = std::chrono::steady_clock::now();

  while (g_running) {
    bool r1 = false, r2 = false;
    {
      std::lock_guard<std::mutex> lock(g_alarmMutex);
      r1 = roi_1_alarm;
      r2 = roi_2_alarm;
    }

    std::string historyJson = "";
    {
      std::lock_guard<std::mutex> lock(g_historyMutex);
      for (size_t i = 0; i < g_entryHistory.size(); ++i) {
        historyJson += "\"" + g_entryHistory[i] + "\"";
        if (i + 1 < g_entryHistory.size()) {
          historyJson += ", ";
        }
      }
    }

    std::string jsonStr =
        "{\"roi_1_alarm\": " + std::string(r1 ? "true" : "false") +
        ", \"roi_2_alarm\": " + std::string(r2 ? "true" : "false") +
        ", \"history\": [" + historyJson + "]}";



    {
      std::lock_guard<std::mutex> lock(g_wsClientsMutex);
      if (!g_wsClients.empty()) {
        for (auto it = g_wsClients.begin(); it != g_wsClients.end(); ) {
          auto ws = *it;

          if (ws->getReadyState() != ix::ReadyState::Open) {
            it = g_wsClients.erase(it);
            continue;
          }

          if (ws->bufferedAmount() > 500 * 1024) {
            ++it;
            continue;
          }

          ws->sendText(jsonStr);
          ++it;
        }
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 FPS for JSON status
  }

  server.stop();
}

// ==========================================
// ==========================================
void modbusThreadFunc() {
  modbus_t *ctx = modbus_new_tcp("0.0.0.0", MODBUS_PORT);
  if (ctx == NULL) {
    std::cerr << "[Modbus] Không thể tạo Modbus TCP context." << std::endl;
    return;
  }

  modbus_mapping_t *mb_mapping = modbus_mapping_new(2, 0, 0, 0);
  if (mb_mapping == NULL) {
    std::cerr << "[Modbus] Lỗi cấp phát bộ nhớ Modbus." << std::endl;
    modbus_free(ctx);
    return;
  }

  int server_socket = modbus_tcp_listen(ctx, 1);
  if (server_socket == -1) {
    std::cerr << "[Modbus] Lỗi lắng nghe cổng " << MODBUS_PORT << std::endl;
    modbus_mapping_free(mb_mapping);
    modbus_free(ctx);
    return;
  }

  std::cout << "[Modbus] Server đang lắng nghe ở cổng " << MODBUS_PORT << "..."
            << std::endl;

  while (g_running) {
    int client_socket = modbus_tcp_accept(ctx, &server_socket);
    if (client_socket == -1) {
      if (!g_running)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    std::cout << "[Modbus] Đã kết nối thiết bị điều khiển." << std::endl;

    uint8_t query[MODBUS_TCP_MAX_ADU_LENGTH];
    while (g_running) {
      modbus_set_response_timeout(ctx, 0, 200000);

      int rc = modbus_receive(ctx, query);
      if (rc > 0) {
        {
          std::lock_guard<std::mutex> lock(g_alarmMutex);
          mb_mapping->tab_bits[0] = roi_1_alarm ? 1 : 0;
          mb_mapping->tab_bits[1] = roi_2_alarm ? 1 : 0;
        }
        modbus_reply(ctx, query, rc, mb_mapping);
      } else if (rc == -1) {
        break;
      }
    }
    std::cout << "[Modbus] Thiết bị điều khiển ngắt kết nối." << std::endl;
    modbus_close(ctx);
  }

  if (server_socket != -1) {
#ifdef _WIN32
    closesocket(server_socket);
#else
    close(server_socket);
#endif
  }
  modbus_mapping_free(mb_mapping);
  modbus_free(ctx);
}

// ============================================================================
// ============================================================================
int main(int argc, char *argv[]) {
  ix::initNetSystem();

  std::ifstream f("config.json");
  if (!f.is_open()) {
      std::cerr << "Failed to open config.json. Exiting...\n";
      return -1;
  }
  nlohmann::json j;
  try {
      f >> j;
      g_config.weights_path = j["model_settings"]["weights_path"];
      g_config.class_name = j["model_settings"]["class_name"];
      g_config.confidence_threshold = j["model_settings"]["confidence_threshold"];
      g_config.nms_threshold = j["model_settings"]["nms_threshold"];
      
      g_config.intersection_threshold = j["logic_settings"]["intersection_threshold"];
      g_config.polygon_intersection_threshold = j["logic_settings"]["polygon_intersection_threshold"];
      g_config.ignore_left_x = j["logic_settings"]["ignore_left_x"];
      g_config.ignore_bottom_y = j["logic_settings"]["ignore_bottom_y"];
      
      for (auto& pt : j["roi_settings"]["roi_1"]) {
          g_config.roi_1.push_back(cv::Point(pt[0], pt[1]));
      }
      for (auto& pt : j["roi_settings"]["roi_2"]) {
          g_config.roi_2.push_back(cv::Point(pt[0], pt[1]));
      }
      g_config.camera_source = j["camera_source"];
  } catch(const std::exception& e) {
      std::cerr << "Error parsing config.json: " << e.what() << "\n";
      return -1;
  }

  std::string videoSource = g_config.camera_source;
  std::string modelPath = g_config.weights_path;

  // --- TỰ ĐỘNG TÌM FILE .ONNX TRONG THƯ MỤC weights ---
  std::string weightsDir = "weights";
  if (!std::filesystem::exists(weightsDir) && std::filesystem::exists("../weights")) {
    weightsDir = "../weights";
  }
  
  if (std::filesystem::exists(weightsDir)) {
    for (const auto& entry : std::filesystem::directory_iterator(weightsDir)) {
      if (entry.path().extension() == ".onnx") {
        modelPath = entry.path().string();
        std::cout << "[INFO] Tự động load file weights: " << modelPath << std::endl;
        break; // Lấy file .onnx đầu tiên tìm thấy
      }
    }
  }
  // ---------------------------------------------------

  if (argc > 1)
    videoSource = argv[1];
  if (argc > 2)
    modelPath = argv[2];

  if (!std::filesystem::exists(modelPath) &&
      std::filesystem::exists("../" + modelPath)) {
    modelPath = "../" + modelPath;
  }

  CameraStream camera(videoSource);
  if (!camera.start()) {
    ix::uninitNetSystem();
    return -1;
  }

  YOLOv8Detector detector(modelPath, g_config.class_name, cv::Size(640, 640), g_config.confidence_threshold, g_config.nms_threshold);
  if (!detector.loadModel()) {
    camera.stop();
    ix::uninitNetSystem();
    return -1;
  }
  
  // Tự động gán tên lớp từ file labels.txt để in log và không cần sửa config.json
  g_config.class_name = detector.getPrimaryClassName();

  RegionMonitor monitor1(g_config.intersection_threshold);
  cv::Rect roiRect1 = cv::boundingRect(g_config.roi_1);
  monitor1.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect1.x, roiRect1.y,
                               0);
  monitor1.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect1.x + roiRect1.width,
                               roiRect1.y + roiRect1.height, 0);
  monitor1.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect1.x + roiRect1.width,
                               roiRect1.y + roiRect1.height, 0);

  RegionMonitor monitor2(g_config.intersection_threshold);
  cv::Rect roiRect2 = cv::boundingRect(g_config.roi_2);
  monitor2.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect2.x, roiRect2.y,
                               0);
  monitor2.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect2.x + roiRect2.width,
                               roiRect2.y + roiRect2.height, 0);
  monitor2.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect2.x + roiRect2.width,
                               roiRect2.y + roiRect2.height, 0);

  std::thread grabThread(cameraThreadFunc, &camera);
  std::thread aiThread(aiCoreThreadFunc, &detector, &monitor1, &monitor2);
  std::thread wsThread(websocketThreadFunc);
  std::thread modbusThread(modbusThreadFunc);

  std::string winName = "DetectRackProject - Live Server Debug Monitor";
  cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);

  cv::Mat localFrame;

  while (g_running) {
    bool hasNewGui = false;
    {
      std::lock_guard<std::mutex> lock(g_guiMutex);
      if (g_guiNewFrame) {
        localFrame = g_guiFrame.clone();
        g_guiNewFrame = false;
        hasNewGui = true;
      }
    }

    if (hasNewGui && !localFrame.empty()) {
      bool r1 = false, r2 = false;
      {
        std::lock_guard<std::mutex> lock(g_alarmMutex);
        r1 = roi_1_alarm;
        r2 = roi_2_alarm;
      }

      std::vector<Detection> localDets;
      {
        std::lock_guard<std::mutex> lock(g_detsMutex);
        localDets = g_latestDets;
      }

      for (const auto& d : localDets) {
        cv::rectangle(localFrame, d.box, cv::Scalar(255, 165, 0), 2);
        cv::putText(localFrame, d.className + " " + cv::format("%.2f", d.confidence), 
                    cv::Point(d.box.x, d.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 165, 0), 2);
      }

      cv::Scalar color1 = r1 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      std::vector<std::vector<cv::Point>> polys1 = {g_config.roi_1};
      cv::polylines(localFrame, polys1, true, color1, 2);
      cv::putText(localFrame, "ROI 1", cv::Point(g_config.roi_1[0].x, g_config.roi_1[0].y - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color1, 2);

      cv::Scalar color2 = r2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      std::vector<std::vector<cv::Point>> polys2 = {g_config.roi_2};
      cv::polylines(localFrame, polys2, true, color2, 2);
      cv::putText(localFrame, "ROI 2", cv::Point(g_config.roi_2[0].x, g_config.roi_2[0].y - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color2, 2);

      std::string statusText1 =
          "ROI 1: " + std::string(r1 ? "OCCUPIED" : "SAFE");
      std::string statusText2 =
          "ROI 2: " + std::string(r2 ? "OCCUPIED" : "SAFE");
      cv::putText(localFrame, statusText1, cv::Point(30, 40),
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, color1, 2);
      cv::putText(localFrame, statusText2, cv::Point(30, 70),
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, color2, 2);

      cv::imshow(winName, localFrame);
    }

    char key = static_cast<char>(cv::waitKey(1));
    if (key == 'q' || key == 'Q' || key == 27) {
      g_running = false;
      break;
    }
  }

  g_running = false;

  std::cout << "[Shutdown] Đang dừng tất cả các luồng..." << std::endl;
  if (grabThread.joinable())
    grabThread.join();
  if (aiThread.joinable())
    aiThread.join();
  if (wsThread.joinable())
    wsThread.join();
  if (modbusThread.joinable())
    modbusThread.join();

  camera.stop();
  cv::destroyAllWindows();
  ix::uninitNetSystem();

  std::cout << "[Shutdown] Hoàn thành dọn dẹp hệ thống." << std::endl;
  return 0;
}
