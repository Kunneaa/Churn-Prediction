"""Biểu đồ EDA: nhãn, giá trị thiếu, phân phối, ngoại lai, churn theo nhóm, tương quan."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from modules.datasets import TARGET

from .figures import save_figure


def plot_missing(df: pd.DataFrame, title: str, save: str | None = None) -> Figure:
    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(pct) + 1.2))
    ax.barh(pct.index[::-1], pct.to_numpy()[::-1], color="#C44E52")
    for i, v in enumerate(pct.to_numpy()[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% giá trị thiếu")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_target(y: pd.Series, save: str | None = None) -> Figure:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Phân phối nhãn churn")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def _grid(n: int, ncols: int, cell: tuple[float, float]) -> tuple[Figure, np.ndarray]:
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


def plot_numeric_distributions(
    df: pd.DataFrame, cols: list[str], save: str | None = None
) -> Figure:
    fig, axes = _grid(len(cols), 4, (4, 3))
    for ax, col in zip(axes, cols, strict=False):
        lo, hi = df[col].quantile([0.01, 0.99])
        sub = df[df[col].between(lo, hi)]
        sns.histplot(
            data=sub,
            x=col,
            hue=TARGET,
            stat="density",
            common_norm=False,
            bins=40,
            element="step",
            ax=ax,
        )
        ax.set_title(col, fontsize=10)
    fig.suptitle("Phân phối biến số theo churn (cắt 1%-99%)", y=1.0)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_boxplots(df: pd.DataFrame, cols: list[str], save: str | None = None) -> Figure:
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.5))
    for ax, c in zip(np.atleast_1d(axes), cols, strict=False):
        sns.boxplot(data=df, x=TARGET, y=c, ax=ax, fliersize=1)
        ax.set_title(c, fontsize=10)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_churn_rate_by_category(
    df: pd.DataFrame, cols: list[str], save: str | None = None
) -> Figure:
    fig, axes = _grid(len(cols), 3, (5, 3.2))
    target = df[TARGET].astype(int)
    for ax, c in zip(axes, cols, strict=False):
        rate = target.groupby(df[c].fillna("NaN")).agg(["mean", "size"]).sort_values("mean")
        ax.barh(rate.index.astype(str), rate["mean"], color="#55A868")
        ax.axvline(target.mean(), color="k", ls="--", lw=1)
        ax.set_title(f"{c} (n nhóm = {len(rate)})", fontsize=10)
        ax.set_xlabel("churn rate")
    fig.suptitle("Tỷ lệ churn theo biến phân loại (đường đứt = churn rate chung)", y=1.0)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_category_volume(table: pd.DataFrame, title: str, save: str | None = None) -> Figure:
    """Cột: số khách hàng (thang log); đường: churn rate — cho bảng từ `category_churn`."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(table.index.astype(str), table["n"], color="#8172B2")
    ax.set_ylabel("số khách hàng")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(table.index.astype(str), table["churn_rate"], "o-", color="#C44E52")
    ax2.set_ylabel("churn rate")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save)
    return fig


def plot_correlation(df: pd.DataFrame, top_k: int = 25, save: str | None = None) -> Figure:
    corr = df.select_dtypes("number").corr()
    if TARGET in corr.columns:
        keep = corr[TARGET].abs().sort_values(ascending=False).head(top_k).index
        corr = corr.loc[keep, keep]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.7},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"Ma trận tương quan ({len(corr)} biến số liên quan nhất tới churn)")
    fig.tight_layout()
    save_figure(fig, save)
    return fig
