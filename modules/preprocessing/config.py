"""Tham số của bộ tiền xử lý (imputation, encoding, biến đổi, scaling, PCA)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    num_imputer: str = "median"  # "mean" | "median" | "most_frequent" | "knn" (chậm với 80k mẫu)
    cat_imputer: str = "most_frequent"  # "most_frequent" | "constant" (điền "Missing")
    add_missing_indicator: bool = True  # thêm cột cờ is_missing cho cột số
    cat_encoder: str = "onehot"  # "onehot" | "ordinal"
    high_card_encoder: str = "frequency"  # "onehot" | "frequency" | "target" | "drop"
    log_transform: bool = False  # sign(x)*log1p(|x|) cho cột số (giảm độ lệch, ngoại lai)
    scaler: str = "standard"  # "standard" | "minmax" | "none"
    minmax_range: tuple[float, float] = (0.0, 1.0)
    # Chọn đặc trưng (sau scale, trước PCA): None | "kbest_f" (ANOVA F) | "kbest_mi" (mutual
    # information) | "l1" (SelectFromModel với Logistic Regression L1)
    feature_selection: str | None = None
    selection_k: int = 40  # số đặc trưng giữ lại cho kbest_*
    selection_c: float = 0.05  # độ mạnh regularization của L1 (nhỏ -> giữ ít đặc trưng hơn)
    pca_variance: float | None = None  # None | 0.90 | 0.95 | 0.99
