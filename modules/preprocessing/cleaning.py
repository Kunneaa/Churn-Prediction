"""Làm sạch theo từng dòng và chia train/test (chưa fit bất kỳ phép biến đổi nào — DS-03)."""

import pandas as pd
from sklearn.model_selection import train_test_split

from modules.datasets import ID_COLUMN, TARGET, download_data, hidden_missing_mask, load_raw

Split = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def clean(df: pd.DataFrame, treat_hidden_missing: bool = True) -> pd.DataFrame:
    """Bỏ cột ID, ép nhãn về int, (tuỳ chọn) đổi giá trị thiếu ẩn thành NaN.

    Không học tham số từ dữ liệu nên không gây rò rỉ khi làm trước khi chia train/test.
    """
    out = df.drop(columns=[ID_COLUMN], errors="ignore").copy()
    out[TARGET] = out[TARGET].astype(int)
    if treat_hidden_missing:
        out = out.mask(hidden_missing_mask(out))
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Split:
    """Chia stratified train/test. Tập test chỉ dùng để đánh giá cuối cùng."""
    x, y = split_xy(df)
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)


def subsample_train(data: Split, n: int | None, random_state: int = 42) -> Split:
    """Lấy mẫu con phân tầng `n` dòng của TẬP TRAIN (tập test giữ nguyên); n None/lớn -> giữ nguyên."""
    x_train, x_test, y_train, y_test = data
    if n is None or n >= len(x_train):
        return data
    idx = (
        y_train.groupby(y_train, group_keys=False)
        .sample(frac=n / len(x_train), random_state=random_state)
        .index
    )
    return x_train.loc[idx], x_test, y_train.loc[idx], y_test


def load_dataset(
    treat_hidden_missing: bool = True,
    sample: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Split:
    """Tải -> làm sạch -> (tuỳ chọn lấy mẫu phân tầng) -> chia train/test."""
    download_data()
    df = clean(load_raw(), treat_hidden_missing=treat_hidden_missing)
    if sample is not None and sample < len(df):
        df = df.groupby(TARGET, group_keys=False).sample(
            frac=sample / len(df), random_state=random_state
        )
    return train_test(df, test_size=test_size, random_state=random_state)
