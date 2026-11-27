"""Đánh giá cấu hình bằng cross-validation trên train (+ tuỳ chọn đo trên test)."""

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from modules.modeling import RANDOM_STATE, PipelineConfig, build_pipeline
from modules.preprocessing import Split
from modules.shared import get_logger, get_settings

from .metrics import SCORING, evaluate_on_test

log = get_logger(__name__)


def run_experiment(
    cfg: PipelineConfig, data: Split, cv: int = 3, fit_final: bool = True
) -> tuple[dict[str, Any], Pipeline | None]:
    """Đánh giá một cấu hình trên `data = (x_train, x_test, y_train, y_test)`.

    - Stratified k-fold CV trên train -> dùng để CHỌN cấu hình.
    - `fit_final`: fit lại trên toàn bộ train và đo trên test -> chỉ dùng cho báo cáo cuối.
    """
    x_train, x_test, y_train, y_test = data
    pipe = build_pipeline(x_train, cfg)
    row: dict[str, Any] = {"experiment": cfg.name, **cfg.to_dict()}
    t0 = time.perf_counter()
    if cv > 1:
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        res = cross_validate(pipe, x_train, y_train, cv=folds, scoring=SCORING, n_jobs=1)
        for k in SCORING:
            row[f"cv_{k}"] = float(np.mean(res[f"test_{k}"]))
            row[f"cv_{k}_std"] = float(np.std(res[f"test_{k}"]))
    row["cv_time_s"] = time.perf_counter() - t0
    if not fit_final:
        return row, None
    t0 = time.perf_counter()
    fitted = pipe.fit(x_train, y_train)
    row["fit_time_s"] = time.perf_counter() - t0
    row["n_features_after_preprocess"] = int(fitted[:-1].transform(x_train.head(5)).shape[1])
    if x_test is not None:
        row.update({f"test_{k}": v for k, v in evaluate_on_test(fitted, x_test, y_test).items()})
    return row, fitted


def run_grid(
    configs: Iterable[PipelineConfig],
    data: Split,
    cv: int = 3,
    save_as: str | None = None,
    fit_final: bool = False,
) -> pd.DataFrame:
    """Chạy lần lượt nhiều cấu hình, trả về bảng kết quả (lưu `reports/results/<save_as>.csv`)."""
    rows = []
    for cfg in configs:
        row, _ = run_experiment(cfg, data, cv=cv, fit_final=fit_final)
        rows.append(row)
        log.info(
            "[%-40s] CV ROC-AUC=%.4f F1=%.4f (%.0fs)",
            cfg.name,
            row.get("cv_roc_auc", float("nan")),
            row.get("cv_f1", float("nan")),
            row["cv_time_s"],
        )
    df = pd.DataFrame(rows)
    if save_as:
        results_dir = get_settings().results_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / f"{save_as}.csv", index=False)
    return df


def summary_table(df: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Bảng gọn để đưa vào báo cáo: cột cấu hình chọn thêm + mọi chỉ số (không kèm độ lệch chuẩn)."""
    metric_cols = [
        c
        for c in df.columns
        if c.startswith(("cv_", "test_")) and not c.endswith("_std") and c != "cv_time_s"
    ]
    cols = ["experiment", *(extra_cols or []), *metric_cols]
    return df[cols].round(4)
