from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.experiments import (
    THRESHOLD_RULES,
    TREE,
    ExperimentSettings,
    TuningSettings,
    baseline_config,
    best_by_cv,
    customer_groups,
    encoding_stage,
    error_analysis,
    export_stage,
    final_stage,
    importance,
    imputation_stage,
    load_experiment_settings,
    mlp_configs,
    mlp_stage,
    model_stage,
    pca_explained_variance,
    pca_stage,
    run_all,
    scaling_stage,
    selection_stage,
    significance_pairs,
    significance_stage,
    tuning_stage,
)
from modules.modeling import BASELINE
from modules.preprocessing import load_dataset

FAST_YAML = (
    "cv: 2\ncv_repeats: 2\nstage_sample: 150\nthreshold_cv: 2\nthreshold_sample: 150\nimportance_sample: 40\n"
    "tuning: {sample: 120, cv: 2, n_iter: {logreg: 1}, spaces: {logreg: {model__C: [1.0]}}}\n"
    "mlp: [{hidden_layer_sizes: [8], alpha: 0.01}]\n"
)
FAST = ExperimentSettings(
    cv=2,
    cv_repeats=2,
    stage_sample=150,
    threshold_cv=2,
    threshold_sample=150,
    importance_sample=40,
    tuning=TuningSettings(
        sample=120, cv=2, n_iter={"logreg": 1}, spaces={"logreg": {"model__C": [0.1, 1.0]}}
    ),
    mlp=[
        {"hidden_layer_sizes": [8], "alpha": 0.01},
        {"model": "torch_mlp", "hidden_layer_sizes": [8], "max_epochs": 3},
    ],
)


def test_load_default_settings():
    s = load_experiment_settings(Path("configs/default.yaml"))
    assert s.cv >= 2 and set(s.tuning.spaces) == {"logreg", "random_forest", "hist_gb"}
    assert s.tuning.fixed == ["lightgbm"]
    assert all("hidden_layer_sizes" in a for a in s.mlp)


def test_preprocessing_stages_carry_best_config(split, settings_tmp):
    unknown = load_dataset(treat_hidden_missing=False, test_size=0.25, random_state=0)
    imp = imputation_stage(BASELINE, split, unknown, cv=2)
    assert imp.table.experiment.iloc[-1] == "không xử lý thiếu ẩn"
    enc = encoding_stage(imp.config, split, cv=2)
    assert len(enc.table) == 8
    sc = scaling_stage(enc.config, split, cv=2)
    assert sc.config.model == "logreg"
    pca = pca_stage(sc.config, split, cv=2)
    assert "test_roc_auc" not in pca.table.columns  # PCA chỉ dùng train
    sel = selection_stage(pca.config, split, cv=2)
    assert len(sel.table) == 7 and "test_roc_auc" not in sel.table.columns
    assert sel.table["n_features_after_preprocess"].min() <= 20
    models = model_stage(sel.config, split, cv=2)
    assert len(models) == 11 and models.experiment.iloc[0] == "Baseline (Dummy)"
    assert models.cv_roc_auc.iloc[0] == pytest.approx(0.5)


def test_tuning_final_mlp_and_export(split, settings_tmp):
    tuned, tables = tuning_stage(BASELINE, split, FAST.tuning, seed=0)
    assert set(tuned) == {"logreg"} and "mean_test_score" in tables["logreg"]
    final = final_stage([baseline_config(BASELINE), *tuned.values()], split, FAST)
    assert len(final.table) == 6 and set(final.table.threshold) >= {0.5}
    assert list(final.table.threshold_rule[:3]) == ["default", *THRESHOLD_RULES]
    dummy = final.table[final.table.model == "Baseline (Dummy)"]
    assert (dummy.roc_auc == 0.5).all()
    assert {"balanced_accuracy", "f1"} <= set(final.curves[tuned["logreg"].name].columns)
    imp = importance(next(iter(final.models.values())), split, n=40, seed=0)
    assert len(imp) <= 15
    table, mlp_final = mlp_stage(BASELINE, split, FAST)
    assert len(table) == 2
    assert set(mlp_final.table.model) == {"MLP sklearn (best)", "MLP PyTorch (best)"}
    assert mlp_configs(BASELINE, FAST)[0].model_params["hidden_layer_sizes"] == (8,)
    paths = export_stage(BASELINE, split)
    assert set(paths) == {"best_linear", "best_tree", "pca95"}
    assert (paths["best_tree"] / "X_train.npy").exists()
    assert TREE["scaler"] == "none"


@pytest.mark.slow
def test_run_all_end_to_end(settings_tmp, tmp_path):
    cfg = tmp_path / "fast.yaml"
    cfg.write_text(FAST_YAML)
    summary = run_all(cfg)
    assert {"best_config", "final", "features", "minutes"} <= set(summary)
    assert (settings_tmp.results_dir / "final_test_results.csv").exists()


@pytest.mark.slow
def test_run_all_regenerates_every_result_figure(settings_tmp, tmp_path):
    cfg = tmp_path / "fast.yaml"
    cfg.write_text(FAST_YAML)
    run_all(cfg)
    names = {p.stem for p in settings_tmp.figures_dir.glob("*.png")}
    assert {
        "09_pca_variance",
        "10_models_f1_recall",
        "11_final_roc_confusion",
        "12_threshold_tuning",
        "13_permutation_importance",
        "14_subgroups",
    } <= names
    assert (settings_tmp.results_dir / "exp8_subgroups.csv").exists()


def test_best_by_cv_uses_model_stage_cv_including_untuned_models():
    tuned = {
        "logreg": BASELINE.with_(name="LR (tuned)"),
        "lightgbm": BASELINE.with_(name="LightGBM (mặc định)"),
    }
    models = pd.DataFrame(
        {
            "experiment": ["Baseline (Dummy)", "Logistic Regression", "LightGBM", "Linear SVM"],
            "cv_roc_auc": [0.5, 0.64, 0.69, 0.99],  # Linear SVM không có trong tuned -> bỏ qua
            "test_roc_auc": [0.5, 0.99, 0.1, 0.1],  # cột test (nếu có) không ảnh hưởng
        }
    )
    assert best_by_cv(tuned, models) == "LightGBM (mặc định)"
    with pytest.raises(ValueError):
        best_by_cv(tuned, models.iloc[:1])


def test_selection_prefers_simplest_config_within_one_standard_error():
    from modules.experiments.stages import _best_row

    table = pd.DataFrame(
        {
            "experiment": ["đơn giản", "phức tạp", "kém"],
            "cv_roc_auc": [0.634, 0.635, 0.620],
            "cv_roc_auc_std": [0.002, 0.002, 0.002],
        }
    )
    assert _best_row(table, cv=3).experiment == "đơn giản"  # chênh 0.001 < SE = 0.002/√3
    table.loc[1, "cv_roc_auc"] = 0.640
    assert _best_row(table, cv=3).experiment == "phức tạp"


def test_significance_encoding_pair_never_compares_config_with_itself():
    for cat, high in (("onehot", "frequency"), ("ordinal", "target")):
        best = BASELINE.with_(cat_encoder=cat, high_card_encoder=high)
        _, a, b = next(p for p in significance_pairs(best) if p[0].startswith("Encoding"))
        assert (a.cat_encoder, a.high_card_encoder) != (b.cat_encoder, b.high_card_encoder)


# cấu hình đang được chọn ở 3.1–3.5: có cờ thiếu, có log, không PCA
SELECTED = BASELINE.with_(log_transform=True)


@pytest.mark.parametrize(
    ("changes", "question", "alternative"),
    [
        ({"add_missing_indicator": False}, "Cờ giá trị thiếu", "có cờ thiếu"),
        ({"log_transform": False}, "Log-transform", "có log"),
        ({"pca_variance": 0.95}, "PCA 95%", "không PCA"),
    ],
)
def test_significance_pairs_never_compare_config_with_itself(changes, question, alternative):
    pairs = significance_pairs(SELECTED.with_(**changes))
    assert [q for q, a, b in pairs if a.with_(name=b.name) == b] == []
    assert next(b.name for q, _, b in pairs if q.startswith(question)) == alternative


def test_significance_pairs_keep_reported_alternatives_for_selected_config():
    assert [b.name for _, _, b in significance_pairs(SELECTED)] == [
        "không cờ thiếu",
        "không log",
        "ordinal + target",
        "PCA 95%",
        "ANOVA F, k=40",
        "Random Forest",
        "cấu hình chọn (LR)",
    ]


def test_pca_explained_variance_always_uses_standard_scaler(split):
    minmax = pca_explained_variance(BASELINE.with_(scaler="minmax"), split)
    standard = pca_explained_variance(BASELINE.with_(scaler="standard"), split)
    np.testing.assert_allclose(minmax, standard)
    assert minmax.sum() == pytest.approx(1.0)


def test_settings_reject_space_without_n_iter(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("tuning: {n_iter: {logreg: 2}, spaces: {logreg: {model__C: [1]}, knn: {}}}\n")
    with pytest.raises(ValueError, match="knn"):
        load_experiment_settings(cfg)


def test_significance_stage_covers_all_pairs(split):
    table = significance_stage(BASELINE, split, FAST)
    assert len(table) == len(significance_pairs(BASELINE)) == 7
    assert {"câu hỏi", "chênh A-B", "p", "có ý nghĩa (p<0.05)"} <= set(table.columns)
    assert table["p"].between(0, 1).all()


def test_error_analysis_groups_cover_test_set(split, settings_tmp):
    x_tr, x_te, y_tr, _ = split
    from modules.modeling import build_pipeline

    model = build_pipeline(x_tr, BASELINE).fit(x_tr, y_tr)
    tables = error_analysis(model, split, threshold=0.4)
    assert set(tables) == set(customer_groups(x_te))
    assert all(t["n"].sum() <= len(x_te) for t in tables.values())
    assert tables["thời gian sử dụng (tháng)"]["n"].sum() == len(x_te)


def test_tuning_stage_adds_fixed_models_without_search(split):
    tuning = TuningSettings(sample=120, cv=2, n_iter={}, spaces={}, fixed=["lightgbm"])
    tuned, tables = tuning_stage(BASELINE, split, tuning, seed=0)
    assert tables == {} and tuned["lightgbm"].name == "LightGBM (mặc định)"
    assert tuned["lightgbm"].scaler == "none"
