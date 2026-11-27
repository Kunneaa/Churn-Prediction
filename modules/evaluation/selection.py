"""Chọn ngưỡng phân loại và tinh chỉnh siêu tham số — chỉ dùng tập train."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from modules.modeling import RANDOM_STATE, PipelineConfig, build_pipeline

THRESHOLD_CRITERIA = ("f1", "balanced_accuracy")


def tune_threshold(
    cfg: PipelineConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    criterion: str = "f1",
) -> tuple[float, pd.DataFrame]:
    """Chọn ngưỡng tối ưu `criterion` bằng xác suất out-of-fold trên TRAIN (không nhìn tập test).

    `curve` có đủ cột của mọi tiêu chí trong THRESHOLD_CRITERIA -> đọc ngưỡng theo tiêu chí khác bằng
    `best_threshold(curve, ...)` mà không phải chạy lại CV. balanced_accuracy = (TPR + TNR) / 2,
    tối đa nó tương đương tối đa chỉ số Youden J = TPR − FPR.
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(
        build_pipeline(x_train, cfg), x_train, y_train, cv=folds, method="predict_proba"
    )[:, 1]
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curve = pd.DataFrame(
        {
            "threshold": grid,
            "precision": [precision_score(y_train, oof >= t, zero_division=0) for t in grid],
            "recall": [recall_score(y_train, oof >= t) for t in grid],
            "f1": [f1_score(y_train, oof >= t) for t in grid],
            "balanced_accuracy": [balanced_accuracy_score(y_train, oof >= t) for t in grid],
        }
    )
    return best_threshold(curve, criterion), curve


def best_threshold(curve: pd.DataFrame, criterion: str) -> float:
    """Ngưỡng tối đa `criterion` trên bảng từ `tune_threshold`."""
    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(
            f"criterion không hợp lệ: {criterion}. Chọn một trong {THRESHOLD_CRITERIA}"
        )
    return float(curve.loc[curve[criterion].idxmax(), "threshold"])


def random_search(
    cfg: PipelineConfig,
    param_distributions: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """RandomizedSearchCV (ROC-AUC) trên toàn pipeline; tham số mô hình đặt tên 'model__<param>'."""
    search = RandomizedSearchCV(
        build_pipeline(x_train, cfg),
        param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=False,
    )
    search.fit(x_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")] + [
        "mean_test_score",
        "std_test_score",
    ]
    res = res[cols].sort_values("mean_test_score", ascending=False).reset_index(drop=True)
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return best, res
