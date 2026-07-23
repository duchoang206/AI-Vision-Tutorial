#!/bin/bash
echo "[Hệ thống] Đang kiểm tra môi trường..."

# 1. Tự động tải thư viện ONNX Runtime nếu máy mới chưa có
if [ ! -d "third_party/onnxruntime" ]; then
    echo "[Hệ thống] Đang tải ONNX Runtime C++..."
    mkdir -p third_party
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz -O onnxruntime.tgz
    tar -xvf onnxruntime.tgz -C third_party/
    mv third_party/onnxruntime-linux-x64-gpu-1.16.3 third_party/onnxruntime
    rm onnxruntime.tgz
    echo "[Hệ thống] Tải thư viện hoàn tất!"
fi

# 2. Build dự án C++
echo "[Hệ thống] Tiến hành build project..."
mkdir -p build && cd build
cmake ..
make -j$(nproc)

echo "[Hệ thống] Hoàn tất! Chạy ./start.sh để khởi động hệ thống."