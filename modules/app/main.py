"""Điểm vào CLI: `uv run churn --config configs/default.yaml` tái tạo mọi kết quả báo cáo (DS-05)."""

import argparse
import warnings
from pathlib import Path

from modules.experiments import run_all
from modules.shared import get_logger, get_settings, setup_logging

log = get_logger(__name__)


def configure_runtime(log_level: str | None = None) -> None:
    """Gọi một lần ở điểm vào (CLI hoặc notebook): logging + tắt cảnh báo đã xử lý theo thiết kế."""
    setup_logging(log_level or get_settings().log_level)
    # Giá trị phân loại lạ ở test được OneHotEncoder(handle_unknown="infrequent_if_exist") xếp vào
    # nhóm "infrequent" như thiết kế -> cảnh báo này không cần hiển thị.
    warnings.filterwarnings("ignore", message="Found unknown categories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline dự đoán churn — chạy toàn bộ thí nghiệm")
    parser.add_argument("--config", type=Path, default=get_settings().experiment_config)
    args = parser.parse_args()
    configure_runtime()
    summary = run_all(args.config)
    log.info("Kết quả cuối trên test:\n%s", summary["final"].to_string(index=False))


if __name__ == "__main__":
    main()
