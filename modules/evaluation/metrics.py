"""Chỉ số đánh giá phân loại nhị phân."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Tên chỉ số -> scorer của sklearn (dùng trong cross-validation). Precision = 0 khi mô hình không dự đoán
# mẫu dương nào (vd. mốc Dummy ở ngưỡng 0.5) — như evaluate_on_test, thay vì cảnh báo UndefinedMetricWarning.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def evaluate_on_test(
    fitted: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> dict[str, float]:
    """Mọi chỉ số trên một tập dữ liệu ở ngưỡng `threshold` (ROC-AUC, PR-AUC không phụ thuộc ngưỡng)."""
    proba = fitted.predict_proba(x_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
