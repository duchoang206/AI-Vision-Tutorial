// Thư viện xử lý luồng luân chuyển hình ảnh từ Camera (lấy khung hình)
#include "CameraStream.hpp"
// Thư viện quản lý việc giám sát khu vực ROI (Region of Interest)
#include "RegionMonitor.hpp"
// Thư viện tích hợp mô hình YOLOv8 để phát hiện đối tượng (Rack)
#include "YOLOv8Detector.hpp"
// Thư viện cung cấp các biến nguyên tử (atomic) hỗ trợ đồng bộ hóa đa luồng an toàn
#include <atomic>
// Thư viện đo thời gian, tính toán FPS và khoảng thời gian trễ (delay)
#include <chrono>
// Thư viện thao tác với hệ thống tệp tin (kiểm tra file weights sự tồn tại của mô hình)
#include <filesystem>
// Thư viện định dạng dữ liệu luồng vào/ra (định dạng thời gian)
#include <iomanip>
// Thư viện nhập xuất chuẩn của C++ (in log ra console)
#include <iostream>
// Thư viện cung cấp khóa loại trừ tương hỗ (mutex) để đồng bộ hóa tài nguyên dùng chung giữa các luồng
#include <mutex>
// Thư viện xử lý ảnh chính OpenCV (vẽ hình, đọc luồng, ma trận ảnh cv::Mat)
#include <opencv2/opencv.hpp>
// Thư viện tập hợp (set) quản lý các kết nối duy nhất (các Web Clients kết nối)
#include <set>
// Thư viện luồng chuỗi (stringstream) hỗ trợ ghép nối định dạng chuỗi
#include <sstream>
// Thư viện chuỗi ký tự chuẩn C++
#include <string>
// Thư viện lập trình đa luồng (std::thread, std::this_thread)
#include <thread>
// Thư viện mảng động (vector) của C++
#include <vector>

// Các thư viện phục vụ truyền thông mạng (Networking Headers)
// Khởi tạo và giải phóng hệ thống mạng mạng cho thư viện IXWebSocket
#include <ixwebsocket/IXNetSystem.h>
// Khởi tạo các đối tượng Client kết nối WebSocket
#include <ixwebsocket/IXWebSocket.h>
// Khởi tạo WebSocket Server để truyền phát dữ liệu hình ảnh/trạng thái lên Web Dashboard
#include <ixwebsocket/IXWebSocketServer.h>
// Thư viện giao thức truyền thông Modbus TCP kết nối với PLC/AGV
#include <modbus/modbus.h>

// Nếu hệ thống chạy trên hệ điều hành Windows, nạp thư viện socket Winsock2
#ifdef _WIN32
#include <winsock2.h>
#endif

// ============================================================================
// KHU VỰC CẤU HÌNH HỆ THỐNG - DÀNH CHO KHÁCH HÀNG SỬA IP / CỔNG MẠNG
// ============================================================================

// 1. Đường dẫn RTSP mặc định của Camera AI (Địa chỉ IP camera là 192.168.5.201, cổng 554, luồng chính subtype=0)
const std::string DEFAULT_CAMERA_RTSP =
    "rtsp://admin:rtc%402025@192.168.5.201:554/cam/"
    "realmonitor?channel=1&subtype=0";

// 2. Đường dẫn mặc định đến file weights (trọng số) ONNX của mô hình YOLOv8 đã train
const std::string DEFAULT_MODEL_PATH = "weights/best.onnx";

// 3. Khai báo cổng kết nối mạng LAN cho thiết bị ngoại vi
// Cổng Modbus TCP mặc định (502) dùng để truyền trạng thái cảnh báo đến PLC hoặc AGV
const int MODBUS_PORT = 502; 
// Cổng WebSocket Server mặc định (8082) dùng để truyền video và alarm dạng JSON lên giao diện Web
const int WEBSOCKET_PORT = 8082; 

// 4. Khai báo tọa độ các điểm đa giác cho vùng ROI 1 (Hình bình hành giám sát khu vực đặt Rack số 1)
std::vector<cv::Point> g_pts1 = {cv::Point(1168, 391), cv::Point(1296, 344),
                                 cv::Point(1444, 405), cv::Point(1300, 460)};
// Khai báo tọa độ các điểm đa giác cho vùng ROI 2 (Hình bình hành giám sát khu vực đặt Rack số 2)
std::vector<cv::Point> g_pts2 = {cv::Point(1476, 427), cv::Point(1655, 512),
                                 cv::Point(1520, 575), cv::Point(1341, 490)};

// ============================================================================

// --- Trạng thái vi phạm ROI toàn cục ---
// Mutex bảo vệ truy cập đồng thời vào các biến trạng thái báo động ROI
std::mutex g_alarmMutex;
// Biến cờ (Flag) báo động vùng ROI 1: true nếu có Rack bên trong, false nếu an toàn/trống
bool roi_1_alarm = false; 
// Biến cờ (Flag) báo động vùng ROI 2: true nếu có Rack bên trong, false nếu an toàn/trống
bool roi_2_alarm = false;

// --- Lịch sử phát hiện Rack ---
// Mutex bảo vệ truy cập đồng thời vào danh sách lịch sử phát hiện Rack
std::mutex g_historyMutex;
// Danh sách (vector) lưu trữ lịch sử các sự kiện Rack đi vào vùng ROI (định dạng string: ROI_ID | Timestamp)
std::vector<std::string> g_entryHistory;

// --- Vùng đệm chia sẻ (Camera -> AI & WS) ---
// Mutex bảo vệ tài nguyên ảnh gốc nhận được từ luồng camera sang luồng xử lý AI và WebSocket
std::mutex g_bufferMutex;
// Biến lưu trữ khung hình ảnh gốc dùng chung giữa các luồng
cv::Mat g_sharedFrame;
// Cờ báo hiệu đã nhận được khung hình mới từ luồng Camera, luồng AI cần lấy ra để xử lý
bool g_hasNewFrame = false; 

// --- Vùng đệm hiển thị GUI OpenCV cục bộ ---
// Mutex bảo vệ việc truyền ảnh đã xử lý vẽ ROI từ luồng AI sang luồng giao diện GUI OpenCV cục bộ
std::mutex g_guiMutex;
// Biến lưu trữ khung hình đã qua xử lý để hiển thị trên màn hình OpenCV cục bộ
cv::Mat g_guiFrame;
// Cờ báo hiệu có khung hình mới cho việc hiển thị giao diện đồ họa cục bộ
bool g_guiNewFrame = false;

// --- Quản lý các kết nối Web Clients ---
// Mutex bảo vệ danh sách các client kết nối qua WebSocket
std::mutex g_wsClientsMutex;
// Tập hợp (set) lưu trữ các phiên kết nối WebSocket đang hoạt động để broadcast dữ liệu
std::set<std::shared_ptr<ix::WebSocket>> g_wsClients;

// Biến cờ nguyên tử kiểm soát vòng lặp chạy của toàn bộ ứng dụng (chạy khi true, dừng hệ thống khi set false)
std::atomic<bool> g_running(true);

// Hàm lấy mốc thời gian hiện tại dưới dạng chuỗi kí tự chính xác đến từng giây
std::string getCurrentTimestamp() {
  // Lấy thời điểm hiện tại từ đồng hồ hệ thống (system_clock)
  auto now = std::chrono::system_clock::now();
  // Chuyển đổi thời điểm hiện tại sang kiểu thời gian truyền thống time_t của C
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  // Khởi tạo đối tượng stream để định dạng chuỗi
  std::stringstream ss;
#ifdef _WIN32
  // Sử dụng cấu trúc lưu trữ thông tin thời gian biểu (tm) trong Windows
  struct tm buf;
  // Sử dụng hàm an toàn localtime_s của Windows để chuyển đổi time_t sang cấu trúc tm cục bộ
  localtime_s(&buf, &in_time_t);
  // Định dạng thời gian dạng YYYY-MM-DD HH:MM:SS và ghi vào luồng stream
  ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
#else
  // Sử dụng cấu trúc lưu trữ thông tin thời gian biểu (tm) trong Linux/macOS
  struct tm buf;
  // Sử dụng hàm an toàn localtime_r của Linux để chuyển đổi time_t sang cấu trúc tm cục bộ
  localtime_r(&in_time_t, &buf);
  // Định dạng thời gian dạng YYYY-MM-DD HH:MM:SS và ghi vào luồng stream
  ss << std::put_time(&buf, "%Y-%m-%d %H:%M:%S");
#endif
  // Trả về chuỗi kết quả thu được từ luồng stream
  return ss.str();
}

// Hàm lọc và kiểm tra xem vị trí Box phát hiện bởi YOLO có nằm trong khu vực giám sát thực tế hay không
// (Tránh nhận diện nhầm các xe AGV di chuyển ngoài khu vực đỗ Rack)
bool isValidRackArea(const cv::Rect &box) {
  // Điều kiện 1: Tọa độ x của box phải lớn hơn hoặc bằng 700 (nằm bên phải)
  // Điều kiện 2: Biên dưới cùng của box (y + height) phải nhỏ hơn hoặc bằng 600
  return box.x >= 700 && (box.y + box.height) <= 600;
}

// Hàm cải thiện độ tương phản của khung hình camera bằng thuật toán CLAHE (tối ưu khi ánh sáng yếu/lóa)
cv::Mat enhanceContrast(const cv::Mat &src) {
  // Nếu ảnh đầu vào rỗng (không hợp lệ), trả về chính nó
  if (src.empty())
    return src;
  // Khởi tạo các ma trận lưu trữ ảnh tạm thời
  cv::Mat lab, dst;
  // Chuyển đổi không gian màu từ BGR sang Lab để tách biệt kênh độ sáng (L) và kênh màu (a, b)
  cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);
  // Khởi tạo vector chứa 3 kênh màu riêng biệt
  std::vector<cv::Mat> planes(3);
  // Phân tách ảnh không gian Lab thành 3 kênh độc lập (planes[0] là kênh độ sáng L)
  cv::split(lab, planes);
  // Tạo bộ lọc CLAHE (Contrast Limited Adaptive Histogram Equalization) với giới hạn clip là 4.0 và kích thước ô 8x8
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
  // Áp dụng cân bằng lược đồ màu CLAHE lên kênh độ sáng planes[0]
  clahe->apply(planes[0], planes[0]);
  // Trộn các kênh màu đã xử lý độ sáng trở lại thành ảnh không gian màu Lab
  cv::merge(planes, lab);
  // Chuyển đổi ngược lại ảnh từ không gian màu Lab về BGR để hiển thị
  cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
  // Trả về ảnh đã được tối ưu độ tương phản
  return dst;
}

// Hàm in thông tin log báo cáo trạng thái của ROI khi có sự thay đổi (Có Rack <=> Trống)
void reportToServer(const std::string &roiName, bool hasRack) {
  // In ra màn hình console dòng log sự kiện thay đổi trạng thái
  std::cout << "[REPORT] " << roiName << " thay đổi trạng thái: "
            << (hasRack ? "CÓ RACK (OCCUPIED)" : "TRỐNG (EMPTY)") << std::endl;
}

// Hàm kiểm tra va chạm giữa đa giác ROI và danh sách các đối tượng Rack phát hiện bởi YOLOv8
bool checkPolygonIntersection(const std::vector<cv::Point> &polygon,
                              const std::vector<Detection> &detections) {
  // Nếu tọa độ đa giác hoặc danh sách phát hiện rỗng thì coi như không va chạm
  if (polygon.empty() || detections.empty())
    return false;

  // Chuyển đổi tọa độ đa giác kiểu số nguyên (cv::Point) sang kiểu số thực (cv::Point2f) để tính toán chính xác
  std::vector<cv::Point2f> polyF;
  // Đặt trước dung lượng bộ nhớ cho vector để tối ưu hiệu năng
  polyF.reserve(polygon.size());
  // Lặp qua từng điểm đa giác để sao chép sang định dạng số thực
  for (const auto &p : polygon)
    polyF.push_back(cv::Point2f(p.x, p.y));

  // Tính toán diện tích của vùng đa giác ROI
  double polyArea = cv::contourArea(polyF);
  // Nếu diện tích đa giác nhỏ hơn hoặc bằng 0 (đa giác không hợp lệ), trả về false
  if (polyArea <= 0)
    return false;

  // Lặp qua từng đối tượng được phát hiện bởi mô hình AI
  for (const auto &det : detections) {
    // Phương pháp 1: Kiểm tra xem điểm chính giữa cạnh dưới của bounding box đối tượng có nằm trong đa giác ROI không
    // Cạnh dưới (bottom center) phản ánh vị trí tiếp đất của chân kệ Rack
    cv::Point bottomCenter(det.box.x + det.box.width / 2,
                           det.box.y + det.box.height);
    // Sử dụng hàm pointPolygonTest kiểm tra điểm đối với đa giác (trả về >= 0 nếu nằm trong hoặc trên cạnh)
    if (cv::pointPolygonTest(
            polygon, cv::Point2f(bottomCenter.x, bottomCenter.y), false) >= 0) {
      return true; // Phát hiện va chạm thành công
    }

    // Phương pháp 2: Kiểm tra diện tích phần giao nhau giữa bounding box của đối tượng và đa giác ROI
    // Tạo danh sách 4 đỉnh của bounding box của đối tượng dưới dạng Point2f
    std::vector<cv::Point2f> rectF = {
        cv::Point2f(det.box.x, det.box.y),
        cv::Point2f(det.box.x + det.box.width, det.box.y),
        cv::Point2f(det.box.x + det.box.width, det.box.y + det.box.height),
        cv::Point2f(det.box.x, det.box.y + det.box.height)};
    // Vector lưu trữ các đỉnh của đa giác phần giao cắt
    std::vector<cv::Point2f> intersection;
    // Tính diện tích phần giao cắt giữa đa giác ROI và bounding box đối tượng
    float intersectArea =
        cv::intersectConvexConvex(polyF, rectF, intersection, true);
    // Nếu tỷ lệ diện tích giao cắt so với diện tích đa giác ROI vượt quá 40%
    if ((intersectArea / polyArea) > 0.40f) {
      return true; // Xác nhận có đối tượng đỗ trong vùng ROI
    }
  }
  // Trả về false nếu không phát hiện bất kỳ sự va chạm hay đè lấp nào
  return false;
}

// ==========================================
// LUỒNG 1: CAMERA GRABBING (Lấy khung hình thô từ Camera)
// ==========================================
void cameraThreadFunc(CameraStream *camera) {
  // Biến tạm để lưu trữ khung hình vừa chụp
  cv::Mat tempFrame;
  // Vòng lặp chạy liên tục cho đến khi ứng dụng có lệnh tắt (g_running chuyển false)
  while (g_running) {
    // Gọi hàm lấy khung hình từ camera stream
    if (camera->retrieveFrame(tempFrame)) {
      // Nếu khung hình lấy được không rỗng
      if (!tempFrame.empty()) {
        // Sử dụng lock_guard để khóa mutex g_bufferMutex bảo vệ vùng đệm dùng chung
        std::lock_guard<std::mutex> lock(g_bufferMutex);
        // Sao chép sâu (clone) khung hình vào vùng đệm chia sẻ g_sharedFrame
        g_sharedFrame = tempFrame.clone();
        // Đặt cờ thông báo đã có khung hình mới
        g_hasNewFrame = true;
      }
    } else {
      // Nếu camera chưa trả về khung hình mới, tạm dừng luồng 2ms để tránh quá tải CPU
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
}

// ==========================================
// LUỒNG 2: AI CORE (Chạy mô hình YOLOv8 và kiểm tra ROI)
// ==========================================
void aiCoreThreadFunc(YOLOv8Detector *detector, RegionMonitor *monitor1,
                      RegionMonitor *monitor2) {
  // Khung hình cục bộ dùng để xử lý AI trong luồng này
  cv::Mat localFrame;
  // Biến lưu trữ trạng thái có Rack trước đó của ROI 1 để phát hiện sự thay đổi trạng thái
  bool prevOccupied1 = false;
  // Biến lưu trữ trạng thái có Rack trước đó của ROI 2 để phát hiện sự thay đổi trạng thái
  bool prevOccupied2 = false;

  // Vòng lặp chạy liên tục xử lý AI cho đến khi ứng dụng dừng hẳn
  while (g_running) {
    // Biến cờ xác định xem luồng có lấy được khung hình mới để xử lý hay không
    bool process = false;
    {
      // Khóa mutex g_bufferMutex để đọc khung hình từ vùng đệm chia sẻ an toàn
      std::lock_guard<std::mutex> lock(g_bufferMutex);
      // Nếu có khung hình mới từ luồng Camera
      if (g_hasNewFrame) {
        // Sao chép khung hình sang biến cục bộ localFrame
        localFrame = g_sharedFrame.clone();
        // Thiết lập lại cờ báo hiệu đã xử lý khung hình hiện tại
        g_hasNewFrame = false;
        // Đặt cờ xử lý là true
        process = true;
      }
    }

    // Nếu lấy được khung hình mới và ảnh không bị rỗng
    if (process && !localFrame.empty()) {
      // Tăng cường độ tương phản ảnh trước khi đưa vào mô hình phát hiện để cải thiện độ chính xác
      cv::Mat enhanced = enhanceContrast(localFrame);
      // Đưa ảnh đã tăng cường vào đối tượng YOLOv8 để thực hiện nhận diện Rack
      std::vector<Detection> dets = detector->detect(enhanced);

      // Lọc danh sách các đối tượng phát hiện nằm trong vùng hợp lệ
      std::vector<Detection> validDets;
      for (const auto &d : dets) {
        // Nếu bounding box thuộc vùng hợp lệ thì thêm vào danh sách validDets
        if (isValidRackArea(d.box)) {
          validDets.push_back(d);
        }
      }

      // Kiểm tra va chạm đa giác cho ROI 1 với danh sách đối tượng hợp lệ
      bool isOccupied1 = checkPolygonIntersection(g_pts1, validDets);
      // Kiểm tra va chạm đa giác cho ROI 2 với danh sách đối tượng hợp lệ
      bool isOccupied2 = checkPolygonIntersection(g_pts2, validDets);

      // Cập nhật trạng thái giao cắt vào đối tượng monitor1 để quản lý logic nghiệp vụ
      monitor1->checkIntersection(validDets);
      // Cập nhật trạng thái giao cắt vào đối tượng monitor2 để quản lý logic nghiệp vụ
      monitor2->checkIntersection(validDets);

      // Nếu trạng thái của ROI 1 thay đổi so với lần lặp trước đó
      if (isOccupied1 != prevOccupied1) {
        // Gửi log/báo cáo thay đổi trạng thái
        reportToServer("ROI1", isOccupied1);
        // Nếu chuyển từ KHÔNG RACK sang CÓ RACK (tức là vừa đỗ kệ vào)
        if (isOccupied1) {
          // Khóa mutex lịch sử để ghi nhận sự kiện an toàn
          std::lock_guard<std::mutex> lock(g_historyMutex);
          // Giới hạn lịch sử lưu tối đa 20 bản ghi để tránh tràn bộ nhớ
          if (g_entryHistory.size() >= 20) {
            g_entryHistory.clear();
          }
          // Thêm thông tin sự kiện kèm nhãn thời gian hiện tại vào lịch sử
          g_entryHistory.push_back("ROI 1 | " + getCurrentTimestamp());
        }
        // Cập nhật trạng thái trước đó
        prevOccupied1 = isOccupied1;
      }
      // Nếu trạng thái của ROI 2 thay đổi so với lần lặp trước đó
      if (isOccupied2 != prevOccupied2) {
        // Gửi log/báo cáo thay đổi trạng thái
        reportToServer("ROI2", isOccupied2);
        // Nếu chuyển từ KHÔNG RACK sang CÓ RACK
        if (isOccupied2) {
          // Khóa mutex lịch sử để ghi nhận sự kiện an toàn
          std::lock_guard<std::mutex> lock(g_historyMutex);
          // Giới hạn lịch sử lưu tối đa 20 bản ghi
          if (g_entryHistory.size() >= 20) {
            g_entryHistory.clear();
          }
          // Thêm thông tin sự kiện kèm nhãn thời gian hiện tại vào lịch sử
          g_entryHistory.push_back("ROI 2 | " + getCurrentTimestamp());
        }
        // Cập nhật trạng thái trước đó
        prevOccupied2 = isOccupied2;
      }

      {
        // Khóa mutex g_alarmMutex để cập nhật biến trạng thái báo động toàn cục dùng chung
        std::lock_guard<std::mutex> lock(g_alarmMutex);
        roi_1_alarm = isOccupied1;
        roi_2_alarm = isOccupied2;
      }

      {
        // Khóa mutex g_guiMutex để truyền khung hình gốc sang vùng đệm vẽ GUI cục bộ
        std::lock_guard<std::mutex> lock(g_guiMutex);
        g_guiFrame = localFrame.clone();
        g_guiNewFrame = true;
      }
    } else {
      // Nếu không có khung hình mới, tạm dừng luồng 5ms để tránh lặp lãng phí CPU
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

// ==========================================
// LUỒNG 3: WEBSOCKET STREAMING (Phát dữ liệu JSON và ảnh 30 FPS lên Web Dashboard)
// ==========================================
void websocketThreadFunc() {
  // Khởi tạo đối tượng WebSocket Server lắng nghe tại cổng WEBSOCKET_PORT (8082) và IP "0.0.0.0" (chấp nhận mọi card mạng)
  ix::WebSocketServer server(WEBSOCKET_PORT, "0.0.0.0");

  // Thiết lập hàm Callback khi có một Web Client mới thực hiện kết nối vào Server
  server.setOnConnectionCallback(
      [](std::weak_ptr<ix::WebSocket> webSocket,
         std::shared_ptr<ix::ConnectionState> connectionState) {
        // Chuyển đổi con trỏ yếu (weak_ptr) thành con trỏ chia sẻ (shared_ptr) để làm việc an toàn
        auto ws = webSocket.lock();
        if (ws) {
          {
            // Khóa danh sách Clients bằng mutex g_wsClientsMutex
            std::lock_guard<std::mutex> lock(g_wsClientsMutex);
            // Thêm phiên kết nối WebSocket vừa thiết lập vào tập hợp quản lý
            g_wsClients.insert(ws);
          }
          // In log ra console về thông tin IP của client vừa kết nối thành công
          std::cout << "[WebSocket] Thiết bị kết nối: "
                    << connectionState->getRemoteIp() << std::endl;

          // Thiết lập Callback xử lý khi nhận được tin nhắn hoặc thay đổi trạng thái từ client này
          ws->setOnMessageCallback([webSocket](
                                       const ix::WebSocketMessagePtr &msg) {
            // Nếu là thông điệp đóng kết nối (Close) hoặc xảy ra lỗi kết nối (Error)
            if (msg->type == ix::WebSocketMessageType::Close ||
                msg->type == ix::WebSocketMessageType::Error) {
              // Khóa danh sách Clients
              std::lock_guard<std::mutex> lock(g_wsClientsMutex);
              // Lấy con trỏ kết nối
              auto wsShared = webSocket.lock();
              if (wsShared)
                // Xóa kết nối này khỏi tập hợp quản lý để tránh gửi lỗi
                g_wsClients.erase(wsShared);
              // In log thông báo thiết bị ngắt kết nối
              std::cout << "[WebSocket] Thiết bị ngắt kết nối." << std::endl;
            }
          });
        }
      });

  // Server thực hiện lắng nghe trên cổng đã cấu hình
  auto res = server.listen();
  // Nếu lắng nghe thất bại (ví dụ trùng cổng mạng)
  if (!res.first) {
    // In thông báo lỗi ra luồng cerr và kết thúc luồng
    std::cerr << "[WebSocket] Lỗi khởi động server: " << res.second
              << std::endl;
    return;
  }

  // Khởi động WebSocket Server (chạy ngầm xử lý các kết nối)
  server.start();

  // Biến đo lường và theo dõi FPS truyền phát video thực tế lên Web
  double streamFps = 0.0;
  // Biến đếm số lượng khung hình đã truyền trong chu kỳ đo
  int frameCount = 0;
  // Mốc thời gian cập nhật FPS lần cuối
  auto lastFpsUpdate = std::chrono::steady_clock::now();

  // Vòng lặp truyền phát dữ liệu hoạt động cho đến khi tắt ứng dụng
  while (g_running) {
    // Biến lưu trạng thái cảnh báo của ROI1 và ROI2 cục bộ trong luồng này
    bool r1 = false, r2 = false;
    {
      // Khóa mutex báo động toàn cục để lấy dữ liệu trạng thái mới nhất
      std::lock_guard<std::mutex> lock(g_alarmMutex);
      r1 = roi_1_alarm;
      r2 = roi_2_alarm;
    }

    // Biến chuỗi lưu trữ dữ liệu JSON lịch sử đỗ Rack
    std::string historyJson = "";
    {
      // Khóa mutex lịch sử để sao chép thông tin an toàn
      std::lock_guard<std::mutex> lock(g_historyMutex);
      // Lặp qua toàn bộ danh sách lịch sử để ghép chuỗi định dạng mảng JSON
      for (size_t i = 0; i < g_entryHistory.size(); ++i) {
        historyJson += "\"" + g_entryHistory[i] + "\"";
        if (i + 1 < g_entryHistory.size()) {
          historyJson += ", ";
        }
      }
    }

    // Tạo chuỗi JSON hoàn chỉnh chứa trạng thái cảnh báo của 2 ROI và danh sách lịch sử
    std::string jsonStr =
        "{\"roi_1_alarm\": " + std::string(r1 ? "true" : "false") +
        ", \"roi_2_alarm\": " + std::string(r2 ? "true" : "false") +
        ", \"history\": [" + historyJson + "]}";

    // Khởi tạo ma trận ảnh cục bộ để chuẩn bị nén và truyền đi
    cv::Mat localFrame;
    {
      // Khóa mutex bộ đệm dùng chung để lấy khung hình camera mới nhất
      std::lock_guard<std::mutex> lock(g_bufferMutex);
      if (!g_sharedFrame.empty())
        localFrame = g_sharedFrame.clone();
    }

    // Vector chứa byte dữ liệu ảnh sau khi nén JPEG
    std::vector<uchar> localJpeg;
    // Biến cờ xác nhận có ảnh hợp lệ để truyền phát hay không
    bool hasFrame = false;

    // Nếu khung hình cục bộ không rỗng
    if (!localFrame.empty()) {
      // Tăng biến đếm số khung hình
      frameCount++;
      // Lấy thời gian hiện tại để tính FPS
      auto now = std::chrono::steady_clock::now();
      // Tính toán khoảng thời gian trôi qua từ lần cập nhật FPS trước đó
      std::chrono::duration<double> elapsed = now - lastFpsUpdate;
      // Cứ mỗi 0.5 giây thực hiện cập nhật giá trị FPS một lần
      if (elapsed.count() >= 0.5) {
        streamFps = frameCount / elapsed.count();
        frameCount = 0;
        lastFpsUpdate = now;
      }

      // Xác định màu sắc cho ROI 1 vẽ lên màn hình (Đỏ nếu có vi phạm/Occupied, Xanh lá nếu an toàn/Safe)
      cv::Scalar color1 = r1 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      // Đóng gói danh sách điểm của ROI 1 vào vector để vẽ đa giác
      std::vector<std::vector<cv::Point>> polys1 = {g_pts1};
      // Vẽ đường đa giác khép kín của ROI 1 lên khung hình với độ dày nét vẽ là 2
      cv::polylines(localFrame, polys1, true, color1, 2);
      // Viết chữ "ROI 1" lên góc trên của đa giác
      cv::putText(localFrame, "ROI 1", cv::Point(g_pts1[0].x, g_pts1[0].y - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color1, 2);

      // Xác định màu sắc cho ROI 2 vẽ lên màn hình (Đỏ nếu Occupied, Xanh lá nếu Safe)
      cv::Scalar color2 = r2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      // Đóng gói danh sách điểm của ROI 2
      std::vector<std::vector<cv::Point>> polys2 = {g_pts2};
      // Vẽ đường đa giác khép kín của ROI 2 lên khung hình
      cv::polylines(localFrame, polys2, true, color2, 2);
      // Viết chữ "ROI 2" lên góc trên của đa giác
      cv::putText(localFrame, "ROI 2", cv::Point(g_pts2[0].x, g_pts2[0].y - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color2, 2);

      // Định dạng văn bản trạng thái hiển thị góc trái phía trên ảnh
      std::string statusText1 =
          "ROI 1: " + std::string(r1 ? "OCCUPIED" : "SAFE");
      std::string statusText2 =
          "ROI 2: " + std::string(r2 ? "OCCUPIED" : "SAFE");
      // Viết trạng thái ROI 1 lên khung hình ở tọa độ (30, 40)
      cv::putText(localFrame, statusText1, cv::Point(30, 40),
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, color1, 2);
      // Viết trạng thái ROI 2 lên khung hình ở tọa độ (30, 70)
      cv::putText(localFrame, statusText2, cv::Point(30, 70),
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, color2, 2);

      // Nếu chỉ số FPS đo được hợp lệ (> 0)
      if (streamFps > 0.0) {
        // Định dạng chuỗi văn bản hiển thị FPS
        std::string fpsText = cv::format("FPS: %.1f", streamFps);
        // Vẽ bóng chữ màu đen ở phía dưới góc trái để tăng độ tương phản dễ nhìn
        cv::putText(localFrame, fpsText, cv::Point(16, localFrame.rows - 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2,
                    cv::LINE_AA);
        // Vẽ chữ chính màu trắng đè lên bóng chữ
        cv::putText(localFrame, fpsText, cv::Point(15, localFrame.rows - 17),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
      }

      // Khởi tạo ma trận ảnh để resize (giảm độ phân giải truyền lên web để tiết kiệm băng thông)
      cv::Mat webFrame;
      // Thu nhỏ kích thước ảnh về 960x540 (định dạng HD tỷ lệ 16:9)
      cv::resize(localFrame, webFrame, cv::Size(960, 540));
      // Thiết lập chất lượng nén JPEG là 70% để tối ưu hóa tốc độ tải và chất lượng hiển thị
      std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
      // Nén ảnh webFrame thành chuỗi byte định dạng JPEG lưu vào localJpeg
      cv::imencode(".jpg", webFrame, localJpeg, params);
      // Đánh dấu đã nén ảnh thành công để gửi đi
      hasFrame = true;
    }

    // Broadcast truyền thông điệp tới toàn bộ Web Clients đang kết nối
    {
      // Khóa danh sách các client kết nối
      std::lock_guard<std::mutex> lock(g_wsClientsMutex);
      // Nếu có ít nhất một client đang kết nối
      if (!g_wsClients.empty()) {
        // Lặp qua từng client hoạt động để gửi dữ liệu
        for (auto &ws : g_wsClients) {
          // Gửi chuỗi dữ liệu trạng thái JSON text
          ws->sendText(jsonStr);
          // Gửi dữ liệu nhị phân (binary) của khung hình JPEG nếu có
          if (hasFrame && !localJpeg.empty()) {
            std::string binaryData(reinterpret_cast<char *>(localJpeg.data()),
                                   localJpeg.size());
            ws->sendBinary(binaryData);
          }
        }
      }
    }

    // Tạm dừng luồng 33ms để giới hạn tốc độ truyền phát ở mức ~30 khung hình/giây (30 FPS), tiết kiệm tài nguyên
    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
  }

  // Khi ứng dụng thoát vòng lặp, ra lệnh dừng WebSocket Server để giải phóng cổng kết nối
  server.stop();
}

// ==========================================
// LUỒNG 4: MODBUS TCP SERVER (Gửi cảnh báo sang PLC)
// ==========================================
void modbusThreadFunc() {
  // Tạo thực thể Modbus TCP Server lắng nghe trên mọi địa chỉ IP ("0.0.0.0") tại cổng MODBUS_PORT (502)
  modbus_t *ctx = modbus_new_tcp("0.0.0.0", MODBUS_PORT);
  // Nếu không thể khởi tạo đối tượng Modbus TCP context (ví dụ lỗi thiếu bộ nhớ)
  if (ctx == NULL) {
    // In thông báo lỗi ra luồng cerr và kết thúc luồng
    std::cerr << "[Modbus] Không thể tạo Modbus TCP context." << std::endl;
    return;
  }

  // Cấp phát vùng nhớ ánh xạ thanh ghi/bits cho Modbus Server
  // Số lượng Coils (bits): 2, Discrete Inputs: 0, Holding Registers: 0, Input Registers: 0
  modbus_mapping_t *mb_mapping = modbus_mapping_new(2, 0, 0, 0);
  // Nếu lỗi cấp phát vùng nhớ ánh xạ
  if (mb_mapping == NULL) {
    // In thông báo lỗi ra luồng cerr
    std::cerr << "[Modbus] Lỗi cấp phát bộ nhớ Modbus." << std::endl;
    // Giải phóng cấu trúc dữ liệu Modbus TCP context đã tạo
    modbus_free(ctx);
    return;
  }

  // Khởi tạo socket lắng nghe kết nối Modbus TCP, cho phép hàng đợi kết nối tối đa là 1 thiết bị
  int server_socket = modbus_tcp_listen(ctx, 1);
  // Nếu lắng nghe cổng mạng thất bại (ví dụ cổng 502 đã bị phần mềm khác sử dụng)
  if (server_socket == -1) {
    // In thông báo lỗi kèm thông tin cổng mạng ra luồng cerr
    std::cerr << "[Modbus] Lỗi lắng nghe cổng " << MODBUS_PORT << std::endl;
    // Giải phóng vùng nhớ ánh xạ Modbus
    modbus_mapping_free(mb_mapping);
    // Giải phóng cấu trúc dữ liệu Modbus TCP context
    modbus_free(ctx);
    return;
  }

  // In thông báo sẵn sàng lên màn hình console
  std::cout << "[Modbus] Server đang lắng nghe ở cổng " << MODBUS_PORT << "..."
            << std::endl;

  // Vòng lặp nhận kết nối từ các thiết bị điều khiển (PLC/AGV) cho đến khi tắt ứng dụng
  while (g_running) {
    // Chờ và chấp nhận kết nối từ thiết bị điều khiển (client)
    int client_socket = modbus_tcp_accept(ctx, &server_socket);
    // Nếu kết nối bị lỗi hoặc ngắt quãng
    if (client_socket == -1) {
      // Nếu ứng dụng đang trong quá trình tắt, thoát vòng lặp ngay lập tức
      if (!g_running)
        break;
      // Tạm dừng 100ms trước khi thử kết nối lại để tránh lặp lỗi quá nhanh
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    // In thông báo thiết bị điều khiển (ví dụ PLC) đã kết nối thành công
    std::cout << "[Modbus] Đã kết nối thiết bị điều khiển." << std::endl;

    // Mảng chứa dữ liệu truy vấn nhận được từ client (ADU)
    uint8_t query[MODBUS_TCP_MAX_ADU_LENGTH];
    // Vòng lặp giao tiếp trao đổi dữ liệu với client vừa kết nối
    while (g_running) {
      // Thiết lập thời gian chờ phản hồi tối đa (timeout) cho socket là 200 miligiây (200000 microgiây)
      modbus_set_response_timeout(ctx, 0, 200000); // 200ms

      // Đọc gói tin truy vấn Modbus gửi từ Client
      int rc = modbus_receive(ctx, query);
      // Nếu nhận được gói tin truy vấn hợp lệ (kích thước byte > 0)
      if (rc > 0) {
        {
          // Khóa mutex g_alarmMutex để đọc dữ liệu báo động của 2 ROI một cách an toàn đồng bộ
          std::lock_guard<std::mutex> lock(g_alarmMutex);
          // Gán giá trị bit đầu tiên (Coil index 0) đại diện cho ROI 1 Alarm
          mb_mapping->tab_bits[0] = roi_1_alarm ? 1 : 0;
          // Gán giá trị bit thứ hai (Coil index 1) đại diện cho ROI 2 Alarm
          mb_mapping->tab_bits[1] = roi_2_alarm ? 1 : 0;
        }
        // Gửi phản hồi chuẩn Modbus TCP trở lại cho client dựa trên gói tin truy vấn và ánh xạ dữ liệu vừa cập nhật
        modbus_reply(ctx, query, rc, mb_mapping);
      } else if (rc == -1) {
        // Nếu ngắt kết nối (hoặc timeout/lỗi kết nối), thoát vòng lặp giao tiếp để chờ thiết bị khác kết nối lại
        break;
      }
    }
    // In thông báo khi thiết bị điều khiển ngắt kết nối
    std::cout << "[Modbus] Thiết bị điều khiển ngắt kết nối." << std::endl;
    // Đóng socket kết nối với client hiện tại
    modbus_close(ctx);
  }

  // Giải phóng tài nguyên socket lắng nghe của server nếu còn mở
  if (server_socket != -1) {
#ifdef _WIN32
    // Đóng socket trên Windows sử dụng closesocket
    closesocket(server_socket);
#else
    // Đóng socket trên hệ thống UNIX sử dụng close
    close(server_socket);
#endif
  }
  // Giải phóng vùng nhớ ánh xạ Modbus mapping
  modbus_mapping_free(mb_mapping);
  // Giải phóng cấu trúc dữ liệu Modbus TCP context
  modbus_free(ctx);
}

// ============================================================================
// HÀM CHÍNH (Đóng vai trò khởi tạo hệ thống và quản lý giao diện OpenCV cục bộ)
// ============================================================================
int main(int argc, char *argv[]) {
  ix::initNetSystem();

  std::string videoSource = DEFAULT_CAMERA_RTSP;
  std::string modelPath = DEFAULT_MODEL_PATH;
  if (argc > 1)
    videoSource = argv[1];
  if (argc > 2)
    modelPath = argv[2];

  if (!std::filesystem::exists(modelPath) &&
      std::filesystem::exists("../" + modelPath)) {
    modelPath = "../" + modelPath;
  }

  // Khởi động camera
  CameraStream camera(videoSource);
  if (!camera.start()) {
    ix::uninitNetSystem();
    return -1;
  }

  // Khởi động bộ dò tìm YOLOv8
  YOLOv8Detector detector(modelPath, cv::Size(640, 640), 0.20f, 0.45f);
  if (!detector.loadModel()) {
    camera.stop();
    ix::uninitNetSystem();
    return -1;
  }

  // Khởi tạo vùng giám sát cho RegionMonitor từ tọa độ ROI đã khai báo
  RegionMonitor monitor1;
  cv::Rect roiRect1 = cv::boundingRect(g_pts1);
  monitor1.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect1.x, roiRect1.y,
                               0);
  monitor1.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect1.x + roiRect1.width,
                               roiRect1.y + roiRect1.height, 0);
  monitor1.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect1.x + roiRect1.width,
                               roiRect1.y + roiRect1.height, 0);

  RegionMonitor monitor2;
  cv::Rect roiRect2 = cv::boundingRect(g_pts2);
  monitor2.handleMouseCallback(cv::EVENT_LBUTTONDOWN, roiRect2.x, roiRect2.y,
                               0);
  monitor2.handleMouseCallback(cv::EVENT_MOUSEMOVE, roiRect2.x + roiRect2.width,
                               roiRect2.y + roiRect2.height, 0);
  monitor2.handleMouseCallback(cv::EVENT_LBUTTONUP, roiRect2.x + roiRect2.width,
                               roiRect2.y + roiRect2.height, 0);

  // Kích hoạt các luồng chạy song song
  std::thread grabThread(cameraThreadFunc, &camera);
  std::thread aiThread(aiCoreThreadFunc, &detector, &monitor1, &monitor2);
  std::thread wsThread(websocketThreadFunc);
  std::thread modbusThread(modbusThreadFunc);

  // Mở cửa sổ OpenCV hiển thị trực tiếp cục bộ tại Server để debug nhanh
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

      // Vẽ đa giác và hiển thị trạng thái lên cửa sổ GUI cục bộ
      cv::Scalar color1 = r1 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      std::vector<std::vector<cv::Point>> polys1 = {g_pts1};
      cv::polylines(localFrame, polys1, true, color1, 2);
      cv::putText(localFrame, "ROI 1", cv::Point(g_pts1[0].x, g_pts1[0].y - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color1, 2);

      cv::Scalar color2 = r2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
      std::vector<std::vector<cv::Point>> polys2 = {g_pts2};
      cv::polylines(localFrame, polys2, true, color2, 2);
      cv::putText(localFrame, "ROI 2", cv::Point(g_pts2[0].x, g_pts2[0].y - 8),
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
