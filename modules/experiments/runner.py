"""Chạy toàn bộ thí nghiệm bằng một lệnh (DS-05) và ghi bảng/hình vào reports/."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from modules.modeling import BASELINE
from modules.preprocessing import load_dataset, subsample_train
from modules.reporting import (
    plot_final_models,
    plot_importance,
    plot_metric_panels,
    plot_pca_variance,
    plot_subgroups,
    plot_threshold_curves,
)
from modules.shared import get_logger, get_settings

from .error_analysis import error_analysis
from .holdout_stages import (
    baseline_config,
    best_by_cv,
    export_stage,
    final_stage,
    importance,
    mlp_stage,
    tuning_stage,
)
from .settings import load_experiment_settings
from .significance_stage import significance_stage
from .stages import (
    encoding_stage,
    imputation_stage,
    model_stage,
    pca_explained_variance,
    pca_stage,
    scaling_stage,
    selection_stage,
)

log = get_logger(__name__)


def save_table(df: pd.DataFrame, name: str) -> None:
    """Ghi bảng kết quả ra reports/results/<name>.csv."""
    results_dir = get_settings().results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / f"{name}.csv", index=False)


def run_all(config_path: Path) -> dict[str, object]:
    """EDA đã có trong notebook; hàm này tái tạo mọi bảng kết quả, hình đánh giá và đặc trưng."""
    t0 = time.perf_counter()
    s = load_experiment_settings(config_path)
    split = {"test_size": s.test_size, "random_state": s.seed}
    data = load_dataset(treat_hidden_missing=True, **split)
    # thí nghiệm so sánh trên mẫu con train (cùng dòng cho cả hai biến thể làm sạch); test giữ nguyên
    stage_data = subsample_train(data, s.stage_sample, s.seed)
    unknown = load_dataset(treat_hidden_missing=False, **split)
    stage_unknown = subsample_train(unknown, s.stage_sample, s.seed)

    stage = imputation_stage(BASELINE, stage_data, stage_unknown, s.cv)
    save_table(stage.table, "exp1_imputation")
    for name, fn in (
        ("exp2_encoding", encoding_stage),
        ("exp3_scaling", scaling_stage),
        ("exp4_pca", pca_stage),
        ("exp4b_selection", selection_stage),
    ):
        stage = fn(stage.config, stage_data, s.cv)
        save_table(stage.table, name)
    best = stage.config
    plot_pca_variance(pca_explained_variance(best, stage_data), save="09_pca_variance")
    plt.close("all")  # CLI không hiển thị hình: đóng sau khi lưu để không giữ bộ nhớ
    models = model_stage(best, stage_data, s.cv)
    save_table(models, "exp5_models")
    save_table(significance_stage(best, stage_data, s), "exp5b_significance")
    plot_metric_panels(
        models,
        ["cv_f1", "cv_recall"],
        ["CV F1 (ngưỡng 0.5)", "CV Recall (ngưỡng 0.5)"],
        save="10_models_f1_recall",
    )
    plt.close("all")

    tuned, tables = tuning_stage(best, data, s.tuning, s.seed)
    for key, table in tables.items():
        save_table(table, f"exp6_tuning_{key}")
    final = final_stage([baseline_config(best), *tuned.values()], data, s)
    mlp_table, mlp_final = mlp_stage(best, data, s)
    save_table(mlp_table, "exp7_mlp")
    table = pd.concat([final.table, mlp_final.table], ignore_index=True)
    save_table(table, "final_test_results")

    _, x_test, _, y_test = data
    plot_final_models(final.models, x_test, y_test, save="11_final_roc_confusion")
    plot_threshold_curves(final.curves, save="12_threshold_tuning")
    plt.close("all")
    best_name = best_by_cv(tuned, models)  # chọn theo CV ở 3.6, không nhìn tập test
    imp = importance(final.models[best_name], data, s.importance_sample, s.seed)
    plot_importance(imp, f"Top 15 đặc trưng — {best_name}", save="13_permutation_importance")
    rows = final.table[
        (final.table.model == best_name) & (final.table.threshold_rule == "oof_balanced")
    ]
    groups = error_analysis(final.models[best_name], data, float(rows.threshold.iloc[0]))
    save_table(pd.concat(groups, names=["cách chia"]).reset_index(level=0), "exp8_subgroups")
    plot_subgroups(groups, save="14_subgroups")
    plt.close("all")
    features = export_stage(best, data)
    minutes = (time.perf_counter() - t0) / 60
    log.info("Hoàn tất trong %.1f phút; mô hình tốt nhất theo CV: %s", minutes, best_name)
    return {"best_config": best, "final": table, "features": features, "minutes": minutes}
