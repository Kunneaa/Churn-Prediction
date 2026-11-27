"""Cấu hình đọc từ biến môi trường / .env (R-40). Không hardcode đường dẫn hay secret.

Đường dẫn tương đối (mặc định hoặc từ .env) được neo vào gốc repo, nên chạy từ thư mục nào
(gốc repo, notebooks/, Colab) cũng ghi/đọc đúng chỗ.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="CHURN_", extra="ignore"
    )

    app_name: str = "churn"
    env: str = "dev"
    log_level: str = "INFO"
    # Telecom Customer Churn (Teradata Center for CRM, Duke University) — bản public trên Kaggle,
    # endpoint tải được không cần đăng nhập (ADR-0002)
    dataset_page: str = "https://www.kaggle.com/datasets/abhinav89/telecom-customer"
    data_url: str = "https://www.kaggle.com/api/v1/datasets/download/abhinav89/telecom-customer"
    raw_csv_name: str = "Telecom_customer churn.csv"
    data_dir: Path = Path("data/raw")  # dữ liệu thô tải về — chỉ đọc (DS-01)
    features_dir: Path = Path("features")  # đặc trưng .npy (đề bài yêu cầu thư mục features/)
    reports_dir: Path = Path("reports")  # figures/ + results/ dùng trong báo cáo
    experiment_config: Path = Path("configs/default.yaml")  # tham số thí nghiệm (DS-05)

    @field_validator("data_dir", "features_dir", "reports_dir", "experiment_config")
    @classmethod
    def _anchor_to_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.reports_dir / "results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
