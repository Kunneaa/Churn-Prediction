import numpy as np
import pandas as pd
from sklearn.base import clone

from modules.featurize import FeatureEngineer, add_features, load_features, save_features
from modules.modeling import BASELINE, build_preprocessing

NEW_COLUMNS = {
    "rev_per_minute",
    "overage_share",
    "drop_block_rate",
    "complete_rate",
    "mou_change_pct",
    "mou_trend_3v6",
    "equipment_age_years",
    "rev_vs_lifetime_avg",
    "rev_change_pct",
    "rev_trend_3v6",
    "mou_vs_lifetime",
    "custcare_rate",
    "unanswered_rate",
    "recv_ratio",
    "overage_rev_share",
    "eqp_vs_tenure",
    "price_per_model",
    "credit_class",
}


def test_add_features_adds_columns_without_mutating_input(raw):
    before = raw.copy()
    out = add_features(raw)
    assert set(out.columns) >= NEW_COLUMNS
    pd.testing.assert_frame_equal(raw, before)
    assert out["credit_class"].eq(raw["crclscod"].str[0]).all()


def test_add_features_divides_safely_by_zero(raw):
    df = raw.head(3).copy()
    df["mou_Mean"] = 0.0
    assert add_features(df)["rev_per_minute"].isna().all()


def test_feature_engineer_names_match_transform(raw):
    fe = FeatureEngineer().fit(raw)
    assert list(fe.get_feature_names_out()) == list(fe.transform(raw).columns)


def test_save_and_load_features_roundtrip(split):
    x_tr, _, y_tr, _ = split
    out_dir = save_features(build_preprocessing(x_tr, BASELINE), "demo", split, {"note": "test"})
    assert (out_dir / "meta.json").exists()
    xa, xb, ya, yb = load_features("demo")
    assert xa.shape[0] == len(x_tr) and xa.dtype == np.float32
    assert len(ya) == len(y_tr) and len(xb) == len(yb)


def test_save_features_cross_fits_target_encoding(split):
    """X_train phải giống fit_transform (cross-fitting), không phải fit().transform() (rò rỉ nhãn)."""
    x_tr, _, y_tr, _ = split
    cfg = BASELINE.with_(high_card_encoder="target")
    expected = clone(build_preprocessing(x_tr, cfg)).fit_transform(x_tr, y_tr).astype(np.float32)
    leaked = build_preprocessing(x_tr, cfg).fit(x_tr, y_tr).transform(x_tr).astype(np.float32)
    assert not np.allclose(expected, leaked)  # hai cách cho kết quả khác nhau với TargetEncoder
    save_features(build_preprocessing(x_tr, cfg), "te", split, {})
    np.testing.assert_allclose(load_features("te")[0], expected)
