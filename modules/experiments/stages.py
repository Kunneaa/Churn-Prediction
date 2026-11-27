"""Các giai đoạn thí nghiệm 3.1–3.5: mỗi giai đoạn đổi MỘT bước pipeline, chọn bằng CV trên train.

Cấu hình được chọn của giai đoạn trước được mang sang giai đoạn sau. Chọn theo quy tắc một sai số chuẩn:
cấu hình ĐẦU TIÊN trong danh sách (xếp từ đơn giản/mặc định tới phức tạp) có CV ROC-AUC không kém cấu hình
cao nhất quá một sai số chuẩn — chênh lệch nhỏ hơn nhiễu giữa các fold không đủ để chọn phương án phức tạp hơn.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from modules.evaluation import run_grid
from modules.modeling import RANDOM_STATE, PipelineConfig, build_preprocessing
from modules.preprocessing import Split

# Mô hình cây không cần scaling / PCA / chọn đặc trưng (cây tự chọn đặc trưng khi tách nút)
TREE = {"scaler": "none", "pca_variance": None, "feature_selection": None}

IMPUTATION_VARIANTS: dict[str, dict[str, object]] = {
    "median + indicator": {},
    "mean + indicator": {"num_imputer": "mean"},
    "most_frequent + indicator": {"num_imputer": "most_frequent"},
    "median, no indicator": {"add_missing_indicator": False},
    "median + cat='Missing'": {"cat_imputer": "constant"},
}
# đơn giản hoá: KNNImputer có trong pipeline nhưng không so sánh — ~70% dòng có giá trị thiếu
# trên 80k mẫu khiến chi phí tính khoảng cách quá lớn cho Colab.
SCALING_VARIANTS: list[tuple[str, tuple[float, float], bool]] = [
    ("none", (0.0, 1.0), False),
    ("standard", (0.0, 1.0), False),
    ("minmax", (0.0, 1.0), False),
    ("minmax", (-1.0, 1.0), False),
    ("standard", (0.0, 1.0), True),
    ("minmax", (0.0, 1.0), True),
]
PCA_LEVELS: list[float | None] = [None, 0.90, 0.95, 0.99]
SELECTION_VARIANTS: dict[str, dict[str, object]] = {
    "không chọn": {"feature_selection": None},
    "ANOVA F, k=20": {"feature_selection": "kbest_f", "selection_k": 20},
    "ANOVA F, k=40": {"feature_selection": "kbest_f", "selection_k": 40},
    "ANOVA F, k=80": {"feature_selection": "kbest_f", "selection_k": 80},
    "mutual info, k=40": {"feature_selection": "kbest_mi", "selection_k": 40},
    "L1 (C=0.05)": {"feature_selection": "l1", "selection_c": 0.05},
    "L1 (C=0.01)": {"feature_selection": "l1", "selection_c": 0.01},
}


@dataclass(frozen=True)
class StageResult:
    table: pd.DataFrame
    config: PipelineConfig  # cấu hình tốt nhất sau giai đoạn


def _best_row(df: pd.DataFrame, cv: int, metric: str = "cv_roc_auc") -> pd.Series:
    """Dòng đầu tiên có `metric` ≥ (cao nhất − một sai số chuẩn của dòng cao nhất)."""
    top = df.loc[df[metric].idxmax()]
    se = top[f"{metric}_std"] / np.sqrt(cv)
    return df[df[metric] >= top[metric] - se].iloc[0]


def imputation_stage(
    base: PipelineConfig, data: Split, data_unknown: Split, cv: int
) -> StageResult:
    """3.1 Imputation + so sánh thêm: KHÔNG đổi giá trị thiếu ẩn thành NaN (`data_unknown`)."""
    cfgs = {n: base.with_(name=n, **ch) for n, ch in IMPUTATION_VARIANTS.items()}
    table = run_grid(cfgs.values(), data, cv=cv)
    best = cfgs[_best_row(table, cv).experiment]
    no_hidden = base.with_(name="không xử lý thiếu ẩn", treat_hidden_missing=False)
    table = pd.concat([table, run_grid([no_hidden], data_unknown, cv=cv)], ignore_index=True)
    return StageResult(table, best.with_(name="best"))


def encoding_stage(base: PipelineConfig, data: Split, cv: int) -> StageResult:
    """3.2 Encoding: cột phân loại thường × cột nhiều giá trị (crclscod)."""
    cfgs = [
        base.with_(name=f"cat={c} | crclscod={h}", cat_encoder=c, high_card_encoder=h)
        for c, h in product(["onehot", "ordinal"], ["frequency", "target", "onehot", "drop"])
    ]
    table = run_grid(cfgs, data, cv=cv)
    b = _best_row(table, cv)
    return StageResult(
        table, base.with_(cat_encoder=b.cat_encoder, high_card_encoder=b.high_card_encoder)
    )


def scaling_stage(base: PipelineConfig, data: Split, cv: int) -> StageResult:
    """3.3 Scaling (Standard / MinMax với feature_range) ± log1p, trên Logistic Regression và k-NN."""
    cfgs = [
        base.with_(
            name=f"{m} | " + ("log + " if lg else "") + s + (f"{r}" if s == "minmax" else ""),
            model=m,
            scaler=s,
            minmax_range=r,
            log_transform=lg,
        )
        for m in ("logreg", "knn")
        for s, r, lg in SCALING_VARIANTS
    ]
    table = run_grid(cfgs, data, cv=cv)
    lr = [c for c in cfgs if c.model == "logreg"]
    best = next(c for c in lr if c.name == _best_row(table[table.model == "logreg"], cv).experiment)
    return StageResult(
        table,
        base.with_(
            scaler=best.scaler, minmax_range=best.minmax_range, log_transform=best.log_transform
        ),
    )


def pca_stage(base: PipelineConfig, data: Split, cv: int) -> StageResult:
    """3.4 PCA giữ 90/95/99% phương sai so với không PCA (k-NN dùng StandardScaler)."""
    cfgs = [
        base.with_(
            name=f"{m} | PCA={p}",
            model=m,
            pca_variance=p,
            **({"scaler": "standard"} if m == "knn" else {}),
        )
        for m in ("logreg", "knn")
        for p in PCA_LEVELS
    ]
    x_train, _, y_train, _ = data
    # fit_final để ghi số đặc trưng sau PCA; không đưa tập test vào
    table = run_grid(cfgs, (x_train, None, y_train, None), cv=cv, fit_final=True)
    lr = table[table.model == "logreg"]
    best = next(c for c in cfgs if c.name == _best_row(lr, cv).experiment)
    return StageResult(table, base.with_(pca_variance=best.pca_variance))


def selection_stage(base: PipelineConfig, data: Split, cv: int) -> StageResult:
    """3.5 Chọn đặc trưng (lọc theo ANOVA F / mutual information, nhúng bằng L1) trên Logistic Regression."""
    cfgs = {n: base.with_(name=n, pca_variance=None, **ch) for n, ch in SELECTION_VARIANTS.items()}
    x_train, _, y_train, _ = data
    # fit_final để ghi số đặc trưng giữ lại; không đưa tập test vào
    table = run_grid(cfgs.values(), (x_train, None, y_train, None), cv=cv, fit_final=True)
    best = cfgs[_best_row(table, cv).experiment]
    return StageResult(
        table,
        base.with_(
            pca_variance=None if best.feature_selection else base.pca_variance,
            feature_selection=best.feature_selection,
            selection_k=best.selection_k,
            selection_c=best.selection_c,
        ),
    )


def pca_explained_variance(base: PipelineConfig, data: Split) -> np.ndarray:
    """Tỷ lệ phương sai từng thành phần PCA trên đặc trưng đã tiền xử lý (fit trên train).

    Luôn dùng StandardScaler (như biến thể `pca95` và các dòng k-NN + PCA) để hình khớp số thành phần.
    """
    x_train, _, y_train, _ = data
    cfg = base.with_(scaler="standard", pca_variance=None)
    z = build_preprocessing(x_train, cfg).fit_transform(x_train, y_train)
    return PCA(random_state=RANDOM_STATE).fit(z).explained_variance_ratio_


def model_stage(base: PipelineConfig, data: Split, cv: int) -> pd.DataFrame:
    """3.6 So sánh mô hình (kèm mốc Dummy) trên tiền xử lý đã chọn; thêm biến thể không class_weight."""
    cfgs = [
        base.with_(name="Baseline (Dummy)", model="dummy"),  # mốc: dự đoán theo tỷ lệ lớp
        base.with_(name="Logistic Regression", model="logreg"),
        base.with_(name="Linear SVM", model="linear_svm"),
        base.with_(name="k-NN (k=25)", model="knn", scaler="standard"),
        base.with_(name="Gaussian Naive Bayes", model="naive_bayes"),
        base.with_(name="Random Forest", model="random_forest", **TREE),
        base.with_(name="Gradient Boosting (Hist)", model="hist_gb", **TREE),
        base.with_(name="LightGBM", model="lightgbm", **TREE),
        base.with_(name="Logistic Regression (no class_weight)", model="logreg", class_weight=None),
        base.with_(
            name="Random Forest (no class_weight)", model="random_forest", class_weight=None, **TREE
        ),
        base.with_(
            name="Gradient Boosting (no class_weight)", model="hist_gb", class_weight=None, **TREE
        ),
    ]
    return run_grid(cfgs, data, cv=cv)
