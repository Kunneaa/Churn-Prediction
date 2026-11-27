import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from modules import reporting
from modules.modeling import BASELINE, build_pipeline
from modules.preprocessing import clean


def test_tables(raw):
    cleaned = clean(raw)
    assert len(reporting.overview(raw)) == raw.shape[1]
    hidden = reporting.hidden_missing_report(raw)
    assert {"marital", "eqpdays"} <= set(hidden["cột"])
    assert "chênh lệch" in reporting.missing_vs_churn(cleaned).columns
    assert reporting.missing_by_months(raw).shape[0] == 1
    assert "n_unique" in reporting.category_summary(raw).columns
    assert reporting.category_churn(raw, "area")["n"].sum() == raw["area"].notna().sum()
    assert len(reporting.outlier_table(raw, ["mou_Mean", "rev_Mean"])) == 2
    assert len(reporting.target_correlations(cleaned, top=3)) == 3
    assert {"biến 1", "biến 2"} <= set(reporting.highly_correlated_pairs(cleaned, 0.0).columns)


def test_eda_plots_save_png(raw, settings_tmp):
    cleaned = clean(raw)
    figs = [
        reporting.plot_target(raw["churn"], save="t"),
        reporting.plot_missing(raw, "m", save="m"),
        reporting.plot_numeric_distributions(raw, ["mou_Mean", "rev_Mean"], save="d"),
        reporting.plot_boxplots(raw, ["mou_Mean"], save="b"),
        reporting.plot_churn_rate_by_category(raw, ["area", "marital"], save="c"),
        reporting.plot_category_volume(reporting.category_churn(raw, "crclscod"), "v", save="v"),
        reporting.plot_correlation(cleaned, top_k=5, save="r"),
    ]
    assert all(isinstance(f, Figure) for f in figs)
    assert len(list(settings_tmp.figures_dir.glob("*.png"))) == len(figs)


def test_result_plots(split, settings_tmp):
    x_tr, x_te, y_tr, y_te = split
    model = build_pipeline(x_tr, BASELINE).fit(x_tr, y_tr)
    df = pd.DataFrame(
        {"experiment": ["a", "b"], "cv_roc_auc": [0.6, 0.7], "cv_roc_auc_std": [0.01, 0.02]}
    )
    assert reporting.plot_comparison(df).get_xlabel() == "cv_roc_auc"
    reporting.plot_pca_variance(np.array([0.5, 0.3, 0.15, 0.05]), save="pca")
    reporting.plot_metric_panels(df, ["cv_roc_auc", "cv_roc_auc"], ["a", "b"], save="panels")
    reporting.plot_final_models({"lr": model}, x_te, y_te, save="final")
    curve = pd.DataFrame(
        {"threshold": [0.3, 0.5], "f1": [0.6, 0.5], "balanced_accuracy": [0.55, 0.6]}
    )
    reporting.plot_threshold_curves({"lr": curve}, save="thr")
    reporting.plot_importance(pd.Series([0.02, 0.01], index=["a", "b"]), "imp", save="imp")
    sub = pd.DataFrame(
        {"nhóm": ["a", "b"], "n": [5, 7], "roc_auc": [0.6, 0.7], "recall": [0.5, 0.8]}
    )
    reporting.plot_subgroups({"chia": sub}, save="sub")
    reporting.save_figure(Figure(), None)  # name rỗng -> không lưu
    assert {p.stem for p in settings_tmp.figures_dir.glob("*.png")} == {
        "pca",
        "panels",
        "final",
        "thr",
        "imp",
        "sub",
    }


def test_pca_variance_annotation_matches_sklearn_rule():
    # phương sai tích luỹ [0.5, 0.95, 1.0]: PCA(0.95) của sklearn giữ 3 thành phần (side="right")
    fig = reporting.plot_pca_variance(np.array([0.5, 0.45, 0.05]))
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert "95%: 3 thành phần" in texts
    assert "99%: 3 thành phần" in texts  # không vượt quá số thành phần hiện có
