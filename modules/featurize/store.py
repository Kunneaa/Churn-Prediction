"""Lưu / nạp vector đặc trưng sau tiền xử lý dưới dạng .npy (thư mục features/ theo đề bài)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modules.shared import get_settings

FeatureArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def save_features(
    preprocessor: Pipeline,
    name: str,
    split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    meta: dict[str, object],
) -> Path:
    """Fit `preprocessor` (chưa fit) trên train rồi lưu X/y train-test ra `features/<name>/`.

    X_train lấy từ `fit_transform` chứ không phải `fit().transform()`: với TargetEncoder, chỉ
    `fit_transform` dùng cross-fitting; `transform` lại trên chính tập train sẽ rò rỉ nhãn.
    """
    x_train, x_test, y_train, y_test = split
    out_dir = get_settings().features_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train": preprocessor.fit_transform(x_train, y_train).astype(np.float32),
        "X_test": preprocessor.transform(x_test).astype(np.float32),
        "y_train": np.asarray(y_train, dtype=np.int8),
        "y_test": np.asarray(y_test, dtype=np.int8),
    }
    for key, value in arrays.items():
        np.save(out_dir / f"{key}.npy", value)
    try:
        names = [str(n) for n in preprocessor.get_feature_names_out()]
    except AttributeError:  # bước không hỗ trợ tên cột
        names = [f"f{i}" for i in range(arrays["X_train"].shape[1])]
    info = {
        **meta,
        "X_train_shape": list(arrays["X_train"].shape),
        "X_test_shape": list(arrays["X_test"].shape),
        "feature_names": names,
    }
    (out_dir / "meta.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return out_dir


def load_features(name: str) -> FeatureArrays:
    """Nạp (x_train, x_test, y_train, y_test) từ `features/<name>/`."""
    d = get_settings().features_dir / name
    x_train, x_test, y_train, y_test = (
        np.load(d / f"{k}.npy") for k in ("X_train", "X_test", "y_train", "y_test")
    )
    return x_train, x_test, y_train, y_test
