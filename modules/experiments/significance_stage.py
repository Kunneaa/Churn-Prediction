"""Kiểm định thống kê cho các so sánh quan trọng (CV lặp + t-test hiệu chỉnh)."""

import pandas as pd

from modules.evaluation import compare_configs
from modules.modeling import PipelineConfig
from modules.preprocessing import Split

from .settings import ExperimentSettings
from .stages import TREE


def significance_pairs(best: PipelineConfig) -> list[tuple[str, PipelineConfig, PipelineConfig]]:
    """Các cặp (câu hỏi, A, B) cho kết luận chính của báo cáo; A là cấu hình được chọn.

    B luôn là phương án khác hẳn cấu hình đã chọn (tránh so một cấu hình với chính nó).
    """
    lr = best.with_(name="cấu hình chọn (LR)", model="logreg")
    gb = best.with_(name="Gradient Boosting", model="hist_gb", **TREE)
    other_indicator = lr.with_(
        name="không cờ thiếu" if best.add_missing_indicator else "có cờ thiếu",
        add_missing_indicator=not best.add_missing_indicator,
    )
    other_log = lr.with_(
        name="không log" if best.log_transform else "có log",
        log_transform=not best.log_transform,
    )
    other_encoding = (
        lr.with_(name="ordinal + target", cat_encoder="ordinal", high_card_encoder="target")
        if (best.cat_encoder, best.high_card_encoder) == ("onehot", "frequency")
        else lr.with_(
            name="onehot + frequency", cat_encoder="onehot", high_card_encoder="frequency"
        )
    )
    other_pca = (
        lr.with_(name="không PCA", pca_variance=None)
        if (best.pca_variance, best.feature_selection) == (0.95, None)
        else lr.with_(name="PCA 95%", pca_variance=0.95, feature_selection=None)
    )
    selection = (
        lr.with_(name="không chọn đặc trưng", feature_selection=None)
        if best.feature_selection
        else lr.with_(name="ANOVA F, k=40", feature_selection="kbest_f", selection_k=40)
    )
    return [
        ("Cờ giá trị thiếu có giúp?", lr, other_indicator),
        ("Log-transform có giúp?", lr, other_log),
        ("Encoding có quan trọng?", lr, other_encoding),
        ("PCA 95% có hại?", lr, other_pca),
        ("Lựa chọn đặc trưng có tác động?", lr, selection),
        ("GB hơn RF?", gb, gb.with_(name="Random Forest", model="random_forest")),
        ("GB hơn LR?", gb, lr),
    ]


def significance_stage(
    best: PipelineConfig, data: Split, settings: ExperimentSettings
) -> pd.DataFrame:
    """Bảng kiểm định cho mọi cặp trong `significance_pairs` trên (mẫu con) train."""
    x_train, _, y_train, _ = data
    rows = [
        {"câu hỏi": q, **compare_configs(a, b, x_train, y_train, settings.cv, settings.cv_repeats)}
        for q, a, b in significance_pairs(best)
    ]
    return pd.DataFrame(rows).round(4)
