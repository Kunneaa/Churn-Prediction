"""Phân tích lỗi trên tập test theo các nhóm khách hàng có ý nghĩa nghiệp vụ."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modules.evaluation import subgroup_metrics
from modules.preprocessing import Split

_MIN_GROUP = 300  # nhóm nhỏ hơn gộp vào "khác" để chỉ số ổn định


def customer_groups(x: pd.DataFrame) -> dict[str, pd.Series]:
    """Cách chia nhóm: thời gian sử dụng, hạng tín dụng, tuổi thiết bị, khu vực."""
    credit = x["crclscod"].astype(str).str[0]
    area = x["area"].fillna("không rõ").astype(str)
    return {
        "thời gian sử dụng (tháng)": pd.cut(
            x["months"], [0, 12, 24, 36, np.inf], labels=["≤12", "13–24", "25–36", ">36"]
        ),
        "hạng tín dụng (crclscod[0])": credit.where(
            credit.map(credit.value_counts()) >= _MIN_GROUP, "khác"
        ),
        "tuổi thiết bị (ngày)": pd.cut(
            x["eqpdays"],
            [-np.inf, 180, 365, 730, np.inf],
            labels=["≤180", "181–365", "366–730", ">730"],
        ),
        "khu vực": area.where(area.map(area.value_counts()) >= _MIN_GROUP * 2, "khác"),
    }


def error_analysis(model: Pipeline, data: Split, threshold: float) -> dict[str, pd.DataFrame]:
    """Bảng chỉ số theo nhóm cho từng cách chia, trên TẬP TEST (mô hình và ngưỡng đã cố định)."""
    _, x_test, _, y_test = data
    return {
        name: subgroup_metrics(model, x_test, y_test, groups, threshold)
        for name, groups in customer_groups(x_test).items()
    }
