"""Đọc tham số thí nghiệm từ configs/*.yaml."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TuningSettings:
    sample: int = 20000
    cv: int = 3
    n_iter: dict[str, int] = field(default_factory=dict)
    spaces: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    # mô hình đưa vào đánh giá cuối với tham số mặc định (đã chọn bằng CV trên train), không tuning
    fixed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 42
    test_size: float = 0.2
    cv: int = 3
    cv_repeats: int = 3  # số lần lặp CV cho kiểm định thống kê
    stage_sample: int | None = 30000  # mẫu con train cho thí nghiệm so sánh 3.1–3.5 và MLP
    threshold_cv: int = 3
    threshold_sample: int | None = 30000  # mẫu con train để chọn ngưỡng out-of-fold
    importance_sample: int = 5000
    tuning: TuningSettings = field(default_factory=TuningSettings)
    mlp: list[dict[str, Any]] = field(default_factory=list)


def load_experiment_settings(path: Path) -> ExperimentSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tuning = TuningSettings(**raw.pop("tuning", {}))
    missing = sorted(set(tuning.spaces) - set(tuning.n_iter))
    if missing:  # báo ngay khi đọc config, không đợi tới giai đoạn tuning
        raise ValueError(f"{path}: tuning.spaces có {missing} nhưng thiếu tuning.n_iter tương ứng")
    return ExperimentSettings(tuning=tuning, **raw)
