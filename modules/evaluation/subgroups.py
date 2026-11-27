"""Phân tích lỗi theo nhóm khách hàng: mô hình tốt/kém ở nhóm nào."""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


def subgroup_metrics(
    fitted: Pipeline, x: pd.DataFrame, y: pd.Series, groups: pd.Series, threshold: float = 0.5
) -> pd.DataFrame:
    """Chỉ số theo từng nhóm (`groups` cùng index với x); ROC-AUC = NaN nếu nhóm chỉ có một lớp."""
    proba = pd.Series(fitted.predict_proba(x)[:, 1], index=x.index)
    pred = (proba >= threshold).astype(int)
    rows = []
    for name, idx in groups.groupby(groups, observed=True).groups.items():
        yt, pp, yp = y.loc[idx], proba.loc[idx], pred.loc[idx]
        rows.append(
            {
                "nhóm": str(name),
                "n": len(idx),
                "churn rate": yt.mean(),
                "accuracy": (yt == yp).mean(),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "roc_auc": roc_auc_score(yt, pp) if yt.nunique() == 2 else float("nan"),
            }
        )
    return pd.DataFrame(rows).round(4)
