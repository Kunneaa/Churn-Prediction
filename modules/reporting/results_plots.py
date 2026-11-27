"""Biểu đồ kết quả: so sánh cấu hình, PCA, ROC/confusion, chọn ngưỡng, mức quan trọng đặc trưng."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline

from .figures import save_figure


def plot_comparison(
    df: pd.DataFrame, metric: str = "cv_roc_auc", title: str = "", ax: Axes | None = None
) -> Axes:
    """Biểu đồ thanh ngang so sánh `metric` giữa các thí nghiệm (kèm độ lệch chuẩn nếu có)."""
    ax = ax or plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))[1]
    d = df.sort_values(metric)
    ax.barh(d["experiment"], d[metric], xerr=d.get(f"{metric}_std"), color="#4C72B0", alpha=0.85)
    ax.set_xlim(max(0.0, d[metric].min() - 0.05), min(1.0, d[metric].max() + 0.03))
    for i, v in enumerate(d[metric]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(title or metric)
    ax.figure.tight_layout()
    return ax


def plot_metric_panels(
    df: pd.DataFrame, metrics: list[str], titles: list[str], save: str | None = None
) -> Figure:
    """Nhiều biểu đồ so sánh cạnh nhau (vd. CV F1 và CV recall của các mô hình)."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 0.45 * len(df) + 1.5))
    for ax, metric, title in zip(np.atleast_1d(axes), metrics, titles, strict=True):
        plot_comparison(df, metric, title, ax=ax)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_pca_variance(explained_ratio: np.ndarray, save: str | None = None) -> Figure:
    """Phương sai tích luỹ theo số thành phần chính, đánh dấu mức 90/95/99%."""
    cum = np.cumsum(explained_ratio)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(1, len(cum) + 1), cum)
    for v in (0.90, 0.95, 0.99):
        # cùng quy tắc với sklearn PCA(n_components=v): side="right"; không vượt quá số thành phần
        k = min(int(np.searchsorted(cum, v, side="right") + 1), len(cum))
        ax.axhline(v, ls="--", lw=0.8, color="gray")
        ax.annotate(
            f"{v:.0%}: {k} thành phần",
            (k, v),
            fontsize=9,
            xytext=(5, -12),
            textcoords="offset points",
        )
    ax.set_xlabel("số thành phần chính")
    ax.set_ylabel("phương sai tích luỹ")
    ax.set_title(f"PCA trên {len(cum)} đặc trưng")
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_final_models(
    models: dict[str, Pipeline], x_test: pd.DataFrame, y_test: pd.Series, save: str | None = None
) -> Figure:
    """ROC curve của nhiều mô hình + confusion matrix (ngưỡng 0.5) của từng mô hình."""
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(5 * (len(models) + 1), 4.5))
    for name, m in models.items():
        RocCurveDisplay.from_estimator(m, x_test, y_test, name=name, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC curve (tập test)")
    for ax, (name, m) in zip(axes[1:], models.items(), strict=False):
        ConfusionMatrixDisplay.from_estimator(
            m, x_test, y_test, ax=ax, colorbar=False, display_labels=["No churn", "Churn"]
        )
        ax.set_title(name)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_threshold_curves(curves: dict[str, pd.DataFrame], save: str | None = None) -> Figure:
    """Balanced accuracy và F1 out-of-fold theo ngưỡng cho từng mô hình (bảng từ `tune_threshold`).

    Điểm tròn đánh dấu ngưỡng tối đa của mỗi đường.
    """
    metrics = {"balanced_accuracy": "Balanced accuracy", "f1": "F1"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, (col, label) in zip(axes, metrics.items(), strict=True):
        for name, c in curves.items():
            (line,) = ax.plot(c["threshold"], c[col], label=name)
            best = c.loc[c[col].idxmax()]
            ax.plot(best["threshold"], best[col], "o", color=line.get_color())
        ax.set_xlabel("ngưỡng")
        ax.set_ylabel(f"{label} (out-of-fold trên train)")
        ax.set_title(f"Ngưỡng tối đa {label}")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_importance(importance: pd.Series, title: str, save: str | None = None) -> Figure:
    """Thanh ngang mức quan trọng (vd. permutation importance), lớn nhất ở trên."""
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(importance) + 1))
    ax.barh(importance.index[::-1], importance.to_numpy()[::-1], color="#DD8452")
    ax.set_xlabel("giảm ROC-AUC khi xáo trộn cột")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_subgroups(tables: dict[str, pd.DataFrame], save: str | None = None) -> Figure:
    """ROC-AUC và recall theo nhóm cho từng cách chia (số mẫu ghi trên cột)."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4), squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items(), strict=True):
        pos = np.arange(len(t))
        ax.bar(pos - 0.2, t["roc_auc"], 0.4, label="ROC-AUC", color="#4C72B0")
        ax.bar(pos + 0.2, t["recall"], 0.4, label="recall", color="#DD8452")
        for i, n in enumerate(t["n"]):
            ax.text(i, 0.02, f"n={n}", ha="center", fontsize=8, rotation=90, color="white")
        ax.set_xticks(pos, t["nhóm"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(name, fontsize=10)
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Chất lượng mô hình theo nhóm khách hàng (tập test)", y=1.02)
    fig.tight_layout()
    save_figure(fig, save)
    return fig
