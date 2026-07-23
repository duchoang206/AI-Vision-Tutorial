import urllib.request
import tarfile
import os
import stat
import json

if not os.path.exists("mediamtx"):
    print("[Hệ thống] Đang tải MediaMTX (Video Server)...")
    url = "https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz"
    urllib.request.urlretrieve(url, "mediamtx.tar.gz")
    with tarfile.open("mediamtx.tar.gz", "r:gz") as tar:
        tar.extractall()
    os.remove("mediamtx.tar.gz")
    os.chmod("mediamtx", os.stat("mediamtx").st_mode | stat.S_IEXEC)

with open("config.json", "r") as f:
    config = json.load(f)
rtsp_source = config.get("camera_source", "")

with open("mediamtx.yml", "w") as f:
    f.write(f"paths:\n  cam1:\n    source: {rtsp_source}\n")
print("[Hệ thống] Đã cấu hình mediamtx.yml")
