"""Giai đoạn 4–7: tinh chỉnh, đánh giá một lần trên test, MLP (mở rộng), lưu đặc trưng."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from modules.datasets import TARGET
from modules.evaluation import (
    best_threshold,
    evaluate_on_test,
    random_search,
    run_grid,
    tune_threshold,
)
from modules.featurize import save_features
from modules.modeling import MODEL_LABELS, PipelineConfig, build_pipeline, build_preprocessing
from modules.preprocessing import Split, subsample_train

from .settings import ExperimentSettings, TuningSettings
from .stages import TREE


@dataclass(frozen=True)
class FinalResult:
    table: pd.DataFrame  # chỉ số test ở ngưỡng 0.5 và các ngưỡng chọn out-of-fold trên train
    models: dict[str, Pipeline]
    curves: dict[str, pd.DataFrame]


def tuning_stage(
    base: PipelineConfig, data: Split, tuning: TuningSettings, seed: int
) -> tuple[dict[str, PipelineConfig], dict[str, pd.DataFrame]]:
    """RandomizedSearchCV cho LR / RF / GB trên mẫu con phân tầng của train."""
    xs, _, ys, _ = subsample_train(data, tuning.sample, seed)
    tuned, tables = {}, {}
    for key, space in tuning.spaces.items():
        cfg = base.with_(model=key, **(TREE if key != "logreg" else {}))
        best, table = random_search(cfg, space, xs, ys, n_iter=tuning.n_iter[key], cv=tuning.cv)
        tuned[key] = cfg.with_(name=f"{MODEL_LABELS[key]} (tuned)", model_params=best)
        tables[key] = table
    for (
        key
    ) in tuning.fixed:  # đơn giản hoá: tuning LightGBM tốn ~10 phút mà ra gần đúng tham số mặc định
        tuned[key] = base.with_(name=f"{MODEL_LABELS[key]} (mặc định)", model=key, **TREE)
    return tuned, tables


# Quy tắc ngưỡng: tên trong bảng kết quả -> tiêu chí tối đa trên xác suất out-of-fold của train.
# balanced_accuracy (≡ Youden J) là ngưỡng chính. F1 giữ lại để so sánh: dữ liệu ~50/50 nên đoán "churn"
# cho mọi khách đã có F1 ≈ 0.66 -> tối đa F1 đẩy ngưỡng xuống rất thấp (recall ≈ 1, accuracy giảm).
THRESHOLD_RULES = {"oof_balanced": "balanced_accuracy", "oof_f1": "f1"}


def baseline_config(base: PipelineConfig) -> PipelineConfig:
    """Mốc so sánh cho đánh giá cuối: Dummy dự đoán theo tỷ lệ lớp, không dùng đặc trưng."""
    return base.with_(name=MODEL_LABELS["dummy"], model="dummy")


def final_stage(
    configs: list[PipelineConfig], data: Split, settings: ExperimentSettings
) -> FinalResult:
    """Fit trên toàn bộ train, chọn ngưỡng out-of-fold trên (mẫu con) train, đo MỘT lần trên test.

    Mỗi mô hình có 3 dòng: ngưỡng 0.5 (`default`) và hai ngưỡng trong THRESHOLD_RULES.
    """
    x_train, x_test, y_train, y_test = data
    # đơn giản hoá: chọn ngưỡng trên mẫu con train (mặc định 30k) để vừa thời gian Colab. Đo 2026-11-22
    # (3 mô hình đã tinh chỉnh): ngưỡng lệch 0.01–0.02 và F1 OOF lệch < 0.004 so với dùng toàn bộ 80k.
    # Muốn khớp tuyệt đối: đặt `threshold_sample: null` trong configs/*.yaml.
    x_thr, _, y_thr, _ = subsample_train(data, settings.threshold_sample, settings.seed)
    rows, models, curves = [], {}, {}
    for cfg in configs:
        model = build_pipeline(x_train, cfg).fit(x_train, y_train)
        _, curve = tune_threshold(cfg, x_thr, y_thr, cv=settings.threshold_cv)
        models[cfg.name], curves[cfg.name] = model, curve
        rules = {"default": 0.5} | {r: best_threshold(curve, c) for r, c in THRESHOLD_RULES.items()}
        for rule, t in rules.items():
            rows.append(
                {
                    "model": cfg.name,
                    "threshold_rule": rule,
                    "threshold": t,
                    **evaluate_on_test(model, x_test, y_test, t),
                }
            )
    return FinalResult(pd.DataFrame(rows).round(4), models, curves)


def best_by_cv(tuned: dict[str, PipelineConfig], models: pd.DataFrame) -> str:
    """Tên (trong `tuned`) của họ mô hình có CV ROC-AUC cao nhất ở bảng so sánh mô hình 3.6.

    Mọi họ được đo trên CÙNG mẫu con và CÙNG các fold ở 3.6 (kể cả mô hình không tuning như LightGBM),
    nên so sánh công bằng hơn điểm tuning (mỗi họ một không gian tham số). Không dùng tập test.
    """
    labels = {MODEL_LABELS[k]: k for k in tuned}
    candidates = models[models["experiment"].isin(labels)]
    if candidates.empty:
        raise ValueError(f"bảng so sánh mô hình không có dòng nào cho {sorted(labels)}")
    return tuned[labels[candidates.loc[candidates["cv_roc_auc"].idxmax(), "experiment"]]].name


def importance(model: Pipeline, data: Split, n: int, seed: int) -> pd.Series:
    """Permutation importance (giảm ROC-AUC) trên mẫu `n` dòng của tập test, 15 cột cao nhất."""
    _, x_test, _, y_test = data
    xs = x_test.sample(min(n, len(x_test)), random_state=seed)
    pi = permutation_importance(
        model, xs, y_test.loc[xs.index], scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=1
    )
    return pd.Series(pi.importances_mean, index=xs.columns).sort_values(ascending=False).head(15)


_MLP_FAMILY = {"mlp": "MLP sklearn", "torch_mlp": "MLP PyTorch"}


def mlp_configs(base: PipelineConfig, settings: ExperimentSettings) -> list[PipelineConfig]:
    """Kiến trúc MLP từ configs/*.yaml (`model`: mlp | torch_mlp, còn lại là tham số mô hình).

    Dùng chung tiền xử lý tốt nhất nhưng luôn StandardScaler, không PCA, không chọn đặc trưng.
    """
    cfgs = []
    for entry in settings.mlp:
        params = {k: v for k, v in entry.items() if k != "model"}
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        model = entry.get("model", "mlp")
        desc = ", ".join(f"{k}={v}" for k, v in params.items() if k != "hidden_layer_sizes")
        cfgs.append(
            base.with_(
                name=f"{_MLP_FAMILY[model]} {params['hidden_layer_sizes']} {desc}",
                model=model,
                scaler="standard",
                pca_variance=None,
                feature_selection=None,
                model_params=params,
            )
        )
    return cfgs


def mlp_stage(
    base: PipelineConfig, data: Split, settings: ExperimentSettings
) -> tuple[pd.DataFrame, FinalResult]:
    """So sánh kiến trúc MLP bằng CV (mẫu con train); MLP tốt nhất của MỖI họ (sklearn, PyTorch)
    được đánh giá trên test như mục 5."""
    cfgs = mlp_configs(base, settings)
    stage_data = subsample_train(data, settings.stage_sample, settings.seed)
    table = run_grid(cfgs, stage_data, cv=settings.cv)
    best = []
    for model, label in _MLP_FAMILY.items():
        family = table[table.model == model]
        if len(family):
            name = family.loc[family.cv_roc_auc.idxmax(), "experiment"]
            best.append(next(c for c in cfgs if c.name == name).with_(name=f"{label} (best)"))
    return table, final_stage(best, data, settings)


def export_stage(base: PipelineConfig, data: Split) -> dict[str, Path]:
    """Lưu đặc trưng sau tiền xử lý (fit trên train) cho 3 biến thể ra features/<tên>/*.npy."""
    x_train = data[0]
    variants = {
        "best_linear": base,  # cho LR / SVM / MLP
        "best_tree": base.with_(**TREE),  # cho RF / GB
        "pca95": base.with_(scaler="standard", pca_variance=0.95),  # giảm chiều
    }
    paths = {}
    for name, cfg in variants.items():
        pre = build_preprocessing(x_train, cfg)  # save_features tự fit_transform trên train
        meta = {"config": cfg.with_(name=name).to_dict(), "target": TARGET}
        paths[name] = save_features(pre, name, data, meta)
    return paths
