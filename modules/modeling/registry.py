"""Danh mục mô hình học máy truyền thống (+ MLP cho phần mở rộng)."""

from collections.abc import Callable
from typing import Any

# Nạp lightgbm TRƯỚC torch (torch_mlp import lười): trên macOS nạp torch trước lightgbm trong cùng
# tiến trình làm hai bản libomp xung đột -> treo/segfault. Import ở đây bảo đảm thứ tự cho mọi điểm vào.
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .config import RANDOM_STATE, PipelineConfig
from .torch_mlp import TorchMLPClassifier

Params = dict[str, Any]


def _logreg(p: Params, class_weight: str | None) -> ClassifierMixin:
    return LogisticRegression(max_iter=2000, class_weight=class_weight, **{"C": 1.0, **p})


def _linear_svm(p: Params, class_weight: str | None) -> ClassifierMixin:
    # SVM tuyến tính (hinge loss + L2) tối ưu bằng Averaged SGD: nhanh hơn LinearSVC/liblinear nhiều
    # lần với ~80k mẫu; SVC kernel RBF O(n^2) không khả thi trên Colab. tol=None -> chạy đúng max_iter.
    # SVM không cho xác suất -> CalibratedClassifierCV (Platt scaling) để tính ROC-AUC, chọn ngưỡng.
    params = {"alpha": 1e-3, "average": True, "max_iter": 20, "tol": None, **p}
    svm = SGDClassifier(
        loss="hinge", class_weight=class_weight, random_state=RANDOM_STATE, **params
    )
    return CalibratedClassifierCV(svm, cv=3, method="sigmoid")


def _random_forest(p: Params, class_weight: str | None) -> ClassifierMixin:
    params = {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 10, "n_jobs": -1, **p}
    return RandomForestClassifier(class_weight=class_weight, random_state=RANDOM_STATE, **params)


def _hist_gb(p: Params, class_weight: str | None) -> ClassifierMixin:
    params = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "early_stopping": True,
        **p,
    }
    return HistGradientBoostingClassifier(
        class_weight=class_weight, random_state=RANDOM_STATE, **params
    )


def _lightgbm(p: Params, class_weight: str | None) -> ClassifierMixin:
    # bagging theo dòng + lấy mẫu cột: hơn HistGradientBoosting ~0.003 ROC-AUC (CV 80k, 2026-11-22)
    params = {
        "n_estimators": 1500,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        **p,
    }
    return LGBMClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        **params,
    )


def _dummy(p: Params) -> ClassifierMixin:
    # mốc so sánh: xác suất = tỷ lệ churn trong train, không dùng đặc trưng (ROC-AUC = 0.5)
    return DummyClassifier(**{"strategy": "prior", **p})


def _knn(p: Params) -> ClassifierMixin:
    return KNeighborsClassifier(**{"n_neighbors": 25, "weights": "distance", "n_jobs": -1, **p})


def _naive_bayes(p: Params) -> ClassifierMixin:
    return GaussianNB(**p)


def _mlp(p: Params) -> ClassifierMixin:
    params = {
        "hidden_layer_sizes": (128, 64),
        "alpha": 1e-3,
        "early_stopping": True,
        "max_iter": 200,
        "learning_rate_init": 1e-3,
        **p,
    }
    return MLPClassifier(random_state=RANDOM_STATE, **params)


def _torch_mlp(p: Params) -> ClassifierMixin:
    params = {**p}
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return TorchMLPClassifier(random_state=RANDOM_STATE, **params)


# Mô hình hỗ trợ class_weight / không hỗ trợ
_WEIGHTED: dict[str, Callable[[Params, str | None], ClassifierMixin]] = {
    "logreg": _logreg,
    "linear_svm": _linear_svm,
    "random_forest": _random_forest,
    "hist_gb": _hist_gb,
    "lightgbm": _lightgbm,
}
_UNWEIGHTED: dict[str, Callable[[Params], ClassifierMixin]] = {
    "dummy": _dummy,  # mốc so sánh (baseline không học gì)
    "knn": _knn,
    "naive_bayes": _naive_bayes,
    "mlp": _mlp,  # phần mở rộng (deep learning cho dữ liệu bảng)
    "torch_mlp": _torch_mlp,  # phần mở rộng: MLP PyTorch (BatchNorm, dropout, early stopping)
}

MODEL_LABELS = {
    "dummy": "Baseline (Dummy)",
    "logreg": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "knn": "k-NN",
    "naive_bayes": "Gaussian Naive Bayes",
    "random_forest": "Random Forest",
    "hist_gb": "Gradient Boosting (Hist)",
    "lightgbm": "LightGBM",
    "mlp": "MLP (Neural Network)",
    "torch_mlp": "MLP PyTorch",
}


def build_model(cfg: PipelineConfig) -> ClassifierMixin:
    if cfg.model in _WEIGHTED:
        return _WEIGHTED[cfg.model](cfg.model_params, cfg.class_weight)
    if cfg.model in _UNWEIGHTED:
        return _UNWEIGHTED[cfg.model](cfg.model_params)
    raise ValueError(f"Mô hình không hỗ trợ: {cfg.model}. Chọn một trong {list(MODEL_LABELS)}")
