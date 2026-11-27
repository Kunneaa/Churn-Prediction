"""So sánh hai cấu hình có kiểm định thống kê: CV lặp + t-test hiệu chỉnh (Nadeau & Bengio, 2003)."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from modules.modeling import RANDOM_STATE, PipelineConfig, build_pipeline


def repeated_cv_scores(
    cfg: PipelineConfig, x: pd.DataFrame, y: pd.Series, n_splits: int = 3, n_repeats: int = 3
) -> np.ndarray:
    """ROC-AUC trên từng fold của RepeatedStratifiedKFold (cùng seed -> cùng fold cho mọi cấu hình)."""
    folds = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    return cross_val_score(build_pipeline(x, cfg), x, y, cv=folds, scoring="roc_auc", n_jobs=1)


def corrected_ttest(diff: np.ndarray, test_fraction: float) -> tuple[float, float]:
    """t-test ghép cặp hiệu chỉnh cho CV lặp: phương sai nhân (1/k + n_test/n_train).

    Các fold dùng chung dữ liệu train nên t-test thường đánh giá thấp phương sai (quá lạc quan).
    Trả về (t, p hai phía).
    """
    k = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return (0.0, 1.0) if np.mean(diff) == 0 else (float(np.sign(np.mean(diff)) * np.inf), 0.0)
    ratio = test_fraction / (1 - test_fraction)
    t = float(np.mean(diff) / np.sqrt((1 / k + ratio) * var))
    return t, float(2 * stats.t.sf(abs(t), df=k - 1))


def compare_configs(
    cfg_a: PipelineConfig,
    cfg_b: PipelineConfig,
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    n_repeats: int = 3,
) -> dict[str, float | str]:
    """So sánh ROC-AUC của A với B trên cùng các fold; p < 0.05 -> khác biệt có ý nghĩa."""
    a = repeated_cv_scores(cfg_a, x, y, n_splits, n_repeats)
    b = repeated_cv_scores(cfg_b, x, y, n_splits, n_repeats)
    t, p = corrected_ttest(a - b, test_fraction=1 / n_splits)
    return {
        "A": cfg_a.name,
        "B": cfg_b.name,
        "auc_A": float(a.mean()),
        "auc_B": float(b.mean()),
        "chênh A-B": float((a - b).mean()),
        "t": t,
        "p": p,
        "có ý nghĩa (p<0.05)": bool(p < 0.05),
    }
