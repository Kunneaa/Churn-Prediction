import numpy as np
import pandas as pd
import pytest

from modules.evaluation import (
    SCORING,
    THRESHOLD_CRITERIA,
    best_threshold,
    compare_configs,
    corrected_ttest,
    evaluate_on_test,
    random_search,
    repeated_cv_scores,
    run_experiment,
    run_grid,
    subgroup_metrics,
    summary_table,
    tune_threshold,
)
from modules.modeling import BASELINE, build_pipeline


def test_evaluate_on_test_returns_all_metrics_in_range(split):
    x_tr, x_te, y_tr, y_te = split
    model = build_pipeline(x_tr, BASELINE).fit(x_tr, y_tr)
    scores = evaluate_on_test(model, x_te, y_te, threshold=0.3)
    assert set(scores) == set(SCORING)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_run_experiment_cv_only_does_not_touch_test(split):
    row, fitted = run_experiment(BASELINE, split, cv=2, fit_final=False)
    assert fitted is None
    assert "cv_roc_auc" in row and not any(k.startswith("test_") for k in row)


def test_run_experiment_fit_final_reports_test(split):
    row, fitted = run_experiment(BASELINE, split, cv=2)
    assert fitted is not None and "test_f1" in row and row["n_features_after_preprocess"] > 0


def test_run_grid_saves_csv_and_summary(split, settings_tmp):
    cfgs = [BASELINE.with_(name="a"), BASELINE.with_(name="b", scaler="minmax")]
    df = run_grid(cfgs, split, cv=2, save_as="grid")
    assert list(df.experiment) == ["a", "b"]
    assert (settings_tmp.results_dir / "grid.csv").exists()
    table = summary_table(df, ["scaler"])
    assert "cv_roc_auc_std" not in table.columns and "scaler" in table.columns


def test_tune_threshold_picks_value_from_grid(split):
    x_tr, _, y_tr, _ = split
    thr, curve = tune_threshold(BASELINE, x_tr, y_tr, cv=2)
    assert 0.05 <= thr <= 0.95
    assert thr in set(curve.threshold) and isinstance(curve, pd.DataFrame)
    assert thr == best_threshold(curve, "f1")


def test_tune_threshold_balanced_accuracy_criterion(split):
    x_tr, _, y_tr, _ = split
    thr, curve = tune_threshold(BASELINE, x_tr, y_tr, cv=2, criterion="balanced_accuracy")
    assert set(THRESHOLD_CRITERIA) <= set(curve.columns)
    assert (
        curve.loc[curve.threshold == thr, "balanced_accuracy"].iloc[0]
        == curve.balanced_accuracy.max()
    )
    with pytest.raises(ValueError, match="criterion"):
        best_threshold(curve, "accuracy")


def test_random_search_strips_model_prefix(split):
    x_tr, _, y_tr, _ = split
    best, table = random_search(BASELINE, {"model__C": [0.1, 1.0]}, x_tr, y_tr, n_iter=2, cv=2)
    assert set(best) == {"C"} and len(table) == 2


def test_corrected_ttest_matches_formula_and_is_more_conservative():
    import numpy as np
    from scipy import stats

    diff = np.array([0.010, 0.012, 0.008, 0.011, 0.009, 0.010])
    t, p = corrected_ttest(diff, test_fraction=1 / 3)
    expected_t = diff.mean() / np.sqrt((1 / 6 + 0.5) * diff.var(ddof=1))
    assert t == pytest.approx(expected_t)
    naive_p = stats.ttest_1samp(diff, 0).pvalue
    assert p > naive_p  # hiệu chỉnh làm p lớn hơn t-test thường


def test_corrected_ttest_zero_variance():
    import numpy as np

    assert corrected_ttest(np.zeros(4), 1 / 3) == (0.0, 1.0)
    assert corrected_ttest(np.full(4, 0.01), 1 / 3)[1] == 0.0


def test_compare_configs_uses_same_folds(split):
    x_tr, _, y_tr, _ = split
    same = compare_configs(BASELINE, BASELINE.with_(name="copy"), x_tr, y_tr, 2, 2)
    assert same["chênh A-B"] == 0.0 and same["p"] == 1.0
    assert len(repeated_cv_scores(BASELINE, x_tr, y_tr, 2, 2)) == 4


def test_subgroup_metrics_per_group(split):
    x_tr, x_te, y_tr, y_te = split
    model = build_pipeline(x_tr, BASELINE).fit(x_tr, y_tr)
    groups = pd.Series(np.where(x_te["months"] > 30, "dài", "ngắn"), index=x_te.index)
    table = subgroup_metrics(model, x_te, y_te, groups, threshold=0.5)
    assert set(table["nhóm"]) == {"dài", "ngắn"}
    assert table["n"].sum() == len(x_te)
    assert table[["accuracy", "recall"]].stack().between(0, 1).all()
