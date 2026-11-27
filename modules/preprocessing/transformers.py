"""Bộ tiền xử lý cấu hình được: impute -> encode -> (log) -> scale -> (PCA)."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from modules.datasets import HIGH_CARDINALITY_COLUMNS

from .config import PreprocessConfig

# Nhóm phân loại có < 20 mẫu gộp thành "infrequent" -> không lỗi khi gặp giá trị lạ ở test
_MIN_FREQUENCY = 20


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Thay mỗi giá trị phân loại bằng tần suất xuất hiện của nó trong tập train."""

    def fit(self, x: pd.DataFrame, y: object = None) -> "FrequencyEncoder":
        del y  # không dùng nhãn
        x = pd.DataFrame(x)
        self.columns_ = list(x.columns)
        self.maps_ = {c: x[c].value_counts(normalize=True) for c in self.columns_}
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x = pd.DataFrame(x, columns=self.columns_)
        cols = [x[c].map(self.maps_[c]).astype(float).fillna(0.0) for c in self.columns_]
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Tên `<cột gốc>_freq`; không truyền `input_features` thì lấy tên cột lúc fit.

        Trong Pipeline, bước đứng trước (SimpleImputer) trả mảng numpy nên `columns_` chỉ còn 0..n-1;
        tên cột gốc được Pipeline chuyền vào qua `input_features`.
        """
        names = self.columns_ if input_features is None else input_features
        if len(names) != len(self.columns_):
            raise ValueError(f"input_features cần {len(self.columns_)} tên, nhận {len(names)}")
        return np.array([f"{c}_freq" for c in names])


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """log1p giữ dấu: xử lý được cả giá trị âm (vd. change_mou)."""
    return np.sign(x) * np.log1p(np.abs(x))


def _num_steps(cfg: PreprocessConfig) -> list[tuple[str, object]]:
    if cfg.num_imputer == "knn":
        imputer = KNNImputer(n_neighbors=5, add_indicator=cfg.add_missing_indicator)
    else:
        imputer = SimpleImputer(strategy=cfg.num_imputer, add_indicator=cfg.add_missing_indicator)
    steps: list[tuple[str, object]] = [("impute", imputer)]
    if cfg.log_transform:
        steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    return steps


def _cat_imputer(cfg: PreprocessConfig) -> SimpleImputer:
    if cfg.cat_imputer == "constant":
        return SimpleImputer(strategy="constant", fill_value="Missing")
    return SimpleImputer(strategy=cfg.cat_imputer)


def _scaler(cfg: PreprocessConfig) -> object:
    if cfg.scaler == "standard":
        return StandardScaler()
    if cfg.scaler == "minmax":
        return MinMaxScaler(feature_range=cfg.minmax_range)
    if cfg.scaler == "none":
        return "passthrough"
    raise ValueError(f"scaler không hợp lệ: {cfg.scaler}")


def _cat_encoder(cfg: PreprocessConfig) -> object:
    if cfg.cat_encoder == "onehot":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=_MIN_FREQUENCY,
            drop="if_binary",
            sparse_output=False,
        )
    if cfg.cat_encoder == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"cat_encoder không hợp lệ: {cfg.cat_encoder}")


def _target_encoder() -> TargetEncoder:
    """TargetEncoder (cross-fitting nội bộ chống rò rỉ nhãn), tương thích sklearn cũ trên Colab."""
    if _sklearn_at_least(1, 9):
        return TargetEncoder(
            target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=42)
        )
    return TargetEncoder(target_type="binary", shuffle=True, random_state=42)


def _high_card_encoder(cfg: PreprocessConfig) -> object:
    if cfg.high_card_encoder == "onehot":
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=_MIN_FREQUENCY, sparse_output=False
        )
    if cfg.high_card_encoder == "frequency":
        return FrequencyEncoder()
    if cfg.high_card_encoder == "target":
        return _target_encoder()
    raise ValueError(f"high_card_encoder không hợp lệ: {cfg.high_card_encoder}")


class L1Selector(SelectFromModel):
    """SelectFromModel giữ đặc trưng có hệ số L1 khác 0; luôn giữ ít nhất 1 đặc trưng.

    Regularization quá mạnh có thể đưa mọi hệ số về 0 -> mô hình phía sau không còn cột nào.
    """

    def _get_support_mask(self) -> np.ndarray:
        mask = super()._get_support_mask()
        if not mask.any():  # đơn giản hoá: giữ đặc trưng có |hệ số| lớn nhất
            mask[np.argmax(np.abs(self.estimator_.coef_).ravel())] = True
        return mask


def _mutual_info(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return mutual_info_classif(x, y, random_state=42)


def _sklearn_at_least(major: int, minor: int) -> bool:
    return tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (major, minor)


def _l1_logreg(c: float) -> LogisticRegression:
    """Logistic Regression phạt L1; sklearn >= 1.8 bỏ `penalty`, dùng `l1_ratio=1` (Colab có thể cũ hơn).

    liblinear xáo thứ tự toạ độ bằng RNG: cố định seed để tập cột được chọn lặp lại giữa các lần chạy.
    """
    l1 = {"l1_ratio": 1.0} if _sklearn_at_least(1, 8) else {"penalty": "l1"}
    return LogisticRegression(
        solver="liblinear", C=c, class_weight="balanced", random_state=42, **l1
    )


def _selector(cfg: PreprocessConfig) -> object:
    if cfg.feature_selection == "kbest_f":
        return SelectKBest(f_classif, k=cfg.selection_k)
    if cfg.feature_selection == "kbest_mi":
        return SelectKBest(_mutual_info, k=cfg.selection_k)
    if cfg.feature_selection == "l1":
        return L1Selector(_l1_logreg(cfg.selection_c))
    raise ValueError(f"feature_selection không hợp lệ: {cfg.feature_selection}")


def build_preprocessor(columns_sample: pd.DataFrame, cfg: PreprocessConfig) -> Pipeline:
    """Pipeline impute -> encode -> (log) -> scale -> (chọn đặc trưng) -> (PCA) cho `columns_sample`.

    `columns_sample` chỉ dùng để xác định tên/kiểu cột, không fit trên nó.
    """
    high_card = [c for c in HIGH_CARDINALITY_COLUMNS if c in columns_sample.columns]
    cat_cols = [
        c for c in columns_sample.select_dtypes(exclude="number").columns if c not in high_card
    ]
    num_cols = columns_sample.select_dtypes(include="number").columns.tolist()
    transformers = [
        ("num", Pipeline(_num_steps(cfg)), num_cols),
        ("cat", Pipeline([("impute", _cat_imputer(cfg)), ("encode", _cat_encoder(cfg))]), cat_cols),
    ]
    if high_card and cfg.high_card_encoder != "drop":
        high_card_pipe = Pipeline(
            [("impute", _cat_imputer(cfg)), ("encode", _high_card_encoder(cfg))]
        )
        transformers.append(("high_card", high_card_pipe, high_card))
    steps: list[tuple[str, object]] = [
        (
            "columns",
            ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False),
        ),
        ("scale", _scaler(cfg)),
    ]
    if cfg.feature_selection is not None:
        steps.append(("select", _selector(cfg)))
    if cfg.pca_variance is not None:
        steps.append(("pca", PCA(n_components=cfg.pca_variance, random_state=42)))
    return Pipeline(steps)
