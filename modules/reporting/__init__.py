"""Bảng và biểu đồ cho EDA và báo cáo kết quả (lưu vào reports/figures/)."""

from .eda_plots import (
    plot_boxplots,
    plot_category_volume,
    plot_churn_rate_by_category,
    plot_correlation,
    plot_missing,
    plot_numeric_distributions,
    plot_target,
)
from .eda_tables import (
    category_churn,
    category_summary,
    hidden_missing_report,
    highly_correlated_pairs,
    missing_by_months,
    missing_vs_churn,
    outlier_table,
    overview,
    target_correlations,
)
from .figures import save_figure
from .results_plots import (
    plot_comparison,
    plot_final_models,
    plot_importance,
    plot_metric_panels,
    plot_pca_variance,
    plot_subgroups,
    plot_threshold_curves,
)

__all__ = [
    "category_churn",
    "category_summary",
    "hidden_missing_report",
    "highly_correlated_pairs",
    "missing_by_months",
    "missing_vs_churn",
    "outlier_table",
    "overview",
    "plot_boxplots",
    "plot_category_volume",
    "plot_churn_rate_by_category",
    "plot_comparison",
    "plot_correlation",
    "plot_final_models",
    "plot_importance",
    "plot_metric_panels",
    "plot_missing",
    "plot_numeric_distributions",
    "plot_pca_variance",
    "plot_subgroups",
    "plot_target",
    "plot_threshold_curves",
    "save_figure",
    "target_correlations",
]
