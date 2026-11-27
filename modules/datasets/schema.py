"""Lược đồ dữ liệu thô: cột nhãn/ID, cột nhiều giá trị và mã giá trị thiếu "ẩn"."""

import pandas as pd

TARGET = "churn"
ID_COLUMN = "Customer_ID"
# Cột phân loại nhiều giá trị (credit class code: 54 giá trị, nhiều mã rất hiếm)
HIGH_CARDINALITY_COLUMNS: tuple[str, ...] = ("crclscod",)

# Mã unknown trong cột phân loại (theo data dictionary của bộ dữ liệu).
# "U" trong prizm_social_one nghĩa là Urban (thành thị) -> KHÔNG phải giá trị thiếu.
HIDDEN_MISSING_CODES: dict[str, tuple[str, ...]] = {
    "new_cell": ("U",),  # thuê bao mới: Y / N / U(nknown)
    "marital": ("U",),  # hôn nhân: M / S / A / B / U(nknown)
    "ethnic": ("U",),  # nhóm dân tộc (mã hoá): U = unknown
    "dualband": ("U",),  # máy 2 băng tần: Y / N / T / U(nknown)
    "hnd_webcap": ("UNKW",),  # khả năng web của máy: WCMB / WC / UNKW
}


def hidden_missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Mặt nạ True tại các ô là giá trị thiếu ẩn (mã unknown, eqpdays < 0)."""
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, codes in HIDDEN_MISSING_CODES.items():
        if col in df.columns:
            mask[col] = df[col].isin(codes)
    if "eqpdays" in df.columns:
        mask["eqpdays"] = df["eqpdays"] < 0  # số ngày dùng thiết bị không thể âm
    return mask
