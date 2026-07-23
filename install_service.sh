#!/bin/bash
# Script tạo Systemd Service cho phần mềm Detect-RACK-Project

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME=$USER
SERVICE_FILE="detectrack.service"

echo "[Hệ thống] Đang tạo file cấu hình $SERVICE_FILE..."

cat <<EOF > $SERVICE_FILE
[Unit]
Description=R-SkyView Industrial Edge Server
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory="$PROJECT_DIR"
ExecStart="$PROJECT_DIR/start.sh"
Restart=always
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=detectrack
Environment=DISPLAY=:0
Environment="LD_LIBRARY_PATH=$PROJECT_DIR/third_party/onnxruntime/lib"

[Install]
WantedBy=multi-user.target
EOF

echo "[Hệ thống] Đã tạo file service thành công."
echo "Để cài đặt phần mềm chạy 24/7, hãy chạy các lệnh sau:"
echo "sudo cp $SERVICE_FILE /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable detectrack"
echo "sudo systemctl restart detectrack"
echo ""
echo "Xem log hệ thống bằng lệnh:"
echo "sudo journalctl -u detectrack -f"
