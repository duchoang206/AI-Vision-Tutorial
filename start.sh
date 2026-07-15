#!/bin/bash
# Script khởi động nhanh Edge AI Server (YOLOv8) và Web Dashboard

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$PROJECT_DIR/third_party/onnxruntime/lib:$LD_LIBRARY_PATH"
cd "$PROJECT_DIR"

echo "[Hệ thống] Đang khởi động Web Dashboard (Port 9000)..."
# Chạy Python HTTP Server ngầm (background)
python3 -m http.server 9000 --bind 0.0.0.0 > /dev/null 2>&1 &
WEB_PID=$!

# Đảm bảo Web Dashboard cũng tự động tắt khi bạn tắt AI Server bằng Ctrl+C
trap "echo -e '\n[Hệ thống] Đang tắt Web Dashboard...'; kill $WEB_PID" EXIT

echo "[Hệ thống] Đang khởi động AI Edge Server..."
./build/DetectRackProject
