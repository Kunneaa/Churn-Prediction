"""Tải dữ liệu từ nguồn công khai và đọc dữ liệu thô (không biến đổi)."""

import http.client
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from modules.shared import get_logger, get_settings

log = get_logger(__name__)
# Kaggle trả HTTP 404 cho User-Agent mặc định của urllib và "Mozilla/5.0" -> dùng tên client riêng
_USER_AGENT = "ml-churn-pipeline/1.0"


def raw_csv_path() -> Path:
    settings = get_settings()
    return settings.data_dir / settings.raw_csv_name


def download_data(force: bool = False) -> Path:
    """Tải zip từ Kaggle (public, không cần API key), giải nén vào data_dir; trả về đường dẫn CSV."""
    settings = get_settings()
    csv_path = raw_csv_path()
    if csv_path.exists() and not force:
        log.info("Dữ liệu đã có: %s", csv_path)
        return csv_path
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = settings.data_dir / "dataset.zip"
    log.info("Đang tải dữ liệu từ %s", settings.data_url)
    request = urllib.request.Request(settings.data_url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            zip_path.write_bytes(response.read())
    # mạng chập chờn có thể ném OSError (URLError, timeout) hoặc HTTPException (IncompleteRead)
    except (OSError, http.client.HTTPException) as err:  # dự phòng: curl có sẵn trên Colab
        log.warning("urllib lỗi (%s), thử lại bằng curl", err)
        subprocess.run(
            ["curl", "-sSL", "--fail", "-o", str(zip_path), settings.data_url], check=True
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(settings.data_dir)
    log.info("Đã giải nén vào %s", settings.data_dir)
    return csv_path


def load_raw() -> pd.DataFrame:
    """Đọc dữ liệu thô (chưa xử lý gì)."""
    return pd.read_csv(raw_csv_path(), low_memory=False)
