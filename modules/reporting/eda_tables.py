"""Bảng EDA: tổng quan cột, giá trị thiếu (tường minh / ẩn / theo churn), ngoại lai, tương quan."""

import numpy as np
import pandas as pd

from modules.datasets import TARGET, hidden_missing_mask


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểu dữ liệu, số giá trị khác nhau, % thiếu của từng cột."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "example": df.iloc[0],
        }
    )


def hidden_missing_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Giá trị thiếu 'ẩn' (không phải NaN) trong dữ liệu thô và churn rate tương ứng."""
    mask = hidden_missing_mask(raw)
    churn = raw[TARGET].astype(int)
    rows = []
    for col in mask.columns[mask.any()]:
        m = mask[col]
        rows.append(
            {
                "cột": col,
                "giá trị": ", ".join(sorted(raw.loc[m, col].astype(str).unique())[:3]),
                "số dòng": int(m.sum()),
                "% dữ liệu": round(m.mean() * 100, 2),
                "churn rate khi thiếu (%)": round(churn[m].mean() * 100, 2),
                "churn rate còn lại (%)": round(churn[~m].mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("% dữ liệu", ascending=False)


def missing_vs_churn(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Churn rate khi một cột bị thiếu so với khi có giá trị -> giá trị thiếu có mang thông tin?"""
    churn = df[TARGET].astype(int)
    rows = []
    for c in df.columns[df.isna().any()]:
        m = df[c].isna()
        rows.append(
            {
                "cột": c,
                "% thiếu": round(m.mean() * 100, 2),
                "churn khi thiếu (%)": round(churn[m].mean() * 100, 1),
                "churn khi có (%)": round(churn[~m].mean() * 100, 1),
            }
        )
    out = pd.DataFrame(rows)
    out["chênh lệch"] = (out["churn khi thiếu (%)"] - out["churn khi có (%)"]).round(1)
    return out.sort_values("% thiếu", ascending=False).head(top)


def missing_by_months(raw: pd.DataFrame, column: str = "avg6mou", cap: int = 12) -> pd.DataFrame:
    """% thiếu của `column` theo số tháng sử dụng (thiếu có cấu trúc với khách hàng mới)."""
    pct = raw.groupby(raw["months"].clip(upper=cap))[column].apply(
        lambda s: round(s.isna().mean() * 100, 1)
    )
    return pct.rename("% thiếu").to_frame().T


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Số giá trị khác nhau và giá trị phổ biến nhất của mỗi cột phân loại."""
    cat = df.select_dtypes(exclude="number")
    return pd.DataFrame(
        {"n_unique": cat.nunique(), "giá trị phổ biến nhất": cat.mode().iloc[0]}
    ).sort_values("n_unique", ascending=False)


def category_churn(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Số khách hàng và churn rate theo từng giá trị của `column` (nhiều nhất trước)."""
    return (
        df.groupby(column)[TARGET]
        .agg(n="size", churn_rate="mean")
        .sort_values("n", ascending=False)
    )


def outlier_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Số ngoại lai theo quy tắc IQR (ngoài [Q1 - 1.5 IQR, Q3 + 1.5 IQR])."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        rows.append(
            {
                "cột": c,
                "min": s.min(),
                "Q1": q1,
                "median": s.median(),
                "Q3": q3,
                "max": s.max(),
                "skew": round(s.skew(), 2),
                "n_outliers": int(mask.sum()),
                "% outliers": round(mask.mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("% outliers", ascending=False)


def target_correlations(df: pd.DataFrame, top: int = 10) -> pd.Series:
    """Tương quan tuyến tính của các biến số với nhãn (|r| lớn nhất trước)."""
    corr = df.select_dtypes("number").corr()[TARGET].drop(TARGET)
    return corr.sort_values(key=abs, ascending=False).head(top).round(3)


def highly_correlated_pairs(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    corr = df.select_dtypes("number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().reset_index()
    pairs.columns = ["biến 1", "biến 2", "|corr|"]
    return pairs[pairs["|corr|"] >= threshold].sort_values("|corr|", ascending=False).round(3)
