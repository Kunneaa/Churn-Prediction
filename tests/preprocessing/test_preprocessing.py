import numpy as np
import pandas as pd
import pytest

from modules.datasets import ID_COLUMN, TARGET
from modules.preprocessing import (
    FrequencyEncoder,
    L1Selector,
    PreprocessConfig,
    build_preprocessor,
    clean,
    load_dataset,
    signed_log1p,
    split_xy,
    subsample_train,
    train_test,
)


def test_clean_drops_id_and_converts_hidden_missing(raw):
    out = clean(raw)
    assert ID_COLUMN not in out.columns
    assert out[TARGET].dtype.kind == "i"
    assert not out["marital"].eq("U").any()
    assert out["marital"].isna().sum() == raw["marital"].eq("U").sum()
    assert out["prizm_social_one"].eq("U").any()  # Urban giữ nguyên


def test_clean_keeps_codes_when_disabled(raw):
    assert clean(raw, treat_hidden_missing=False)["marital"].eq("U").any()


def test_split_xy_and_stratified_train_test(raw):
    x, y = split_xy(clean(raw))
    assert TARGET not in x.columns and len(x) == len(y)
    _, x_te, y_tr, y_te = train_test(clean(raw), test_size=0.25, random_state=0)
    assert len(x_te) == pytest.approx(0.25 * len(raw), abs=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05


def test_load_dataset_sample(settings_tmp):
    x_tr, x_te, _, _ = load_dataset(sample=100, random_state=0)
    assert len(x_tr) + len(x_te) == pytest.approx(100, abs=2)


def test_signed_log1p_keeps_sign():
    out = signed_log1p(np.array([-np.e + 1, 0.0, np.e - 1]))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])


def test_frequency_encoder_maps_train_frequency_and_unknown_to_zero():
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "b", "c"]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "z"]}))
    np.testing.assert_allclose(out.ravel(), [0.5, 0.25, 0.0])
    assert list(enc.get_feature_names_out()) == ["c_freq"]


def test_frequency_feature_keeps_source_column_name_after_imputer(split):
    # trong pipeline, FrequencyEncoder đứng sau SimpleImputer nên nhận mảng numpy (mất tên cột)
    x_tr, _, y_tr, _ = split
    pre = build_preprocessor(x_tr, PreprocessConfig(high_card_encoder="frequency")).fit(x_tr, y_tr)
    names = list(pre.get_feature_names_out())
    assert "crclscod_freq" in names
    assert "0_freq" not in names


def test_frequency_encoder_names_numpy_input_from_input_features():
    enc = FrequencyEncoder().fit(np.array([["a"], ["a"], ["b"]]))
    assert list(enc.get_feature_names_out(["c"])) == ["c_freq"]
    with pytest.raises(ValueError, match="input_features"):
        enc.get_feature_names_out(["c", "d"])


@pytest.mark.parametrize(
    "cfg",
    [
        PreprocessConfig(),
        PreprocessConfig(cat_encoder="ordinal", high_card_encoder="target", log_transform=True),
        PreprocessConfig(high_card_encoder="onehot", scaler="minmax", minmax_range=(-1.0, 1.0)),
        PreprocessConfig(high_card_encoder="drop", scaler="none", cat_imputer="constant"),
        PreprocessConfig(num_imputer="knn", pca_variance=0.9, add_missing_indicator=False),
        PreprocessConfig(feature_selection="kbest_f", selection_k=5),
        PreprocessConfig(feature_selection="kbest_mi", selection_k=5, pca_variance=0.9),
        PreprocessConfig(feature_selection="l1", selection_c=0.5),
    ],
)
def test_build_preprocessor_outputs_finite_matrix(split, cfg):
    x_tr, x_te, y_tr, _ = split
    pre = build_preprocessor(x_tr, cfg).fit(x_tr, y_tr)
    out = pre.transform(x_te)
    assert out.shape[0] == len(x_te)
    assert np.isfinite(out).all()


def test_build_preprocessor_rejects_unknown_option(split):
    x_tr, _, y_tr, _ = split
    with pytest.raises(ValueError, match="scaler"):
        build_preprocessor(x_tr, PreprocessConfig(scaler="robust")).fit(x_tr, y_tr)


def test_subsample_train_keeps_test_and_stratifies(split):
    x_tr, x_te, y_tr, y_te = subsample_train(split, 90, random_state=0)
    assert len(x_tr) == pytest.approx(90, abs=2) and x_te is split[1] and y_te is split[3]
    assert abs(y_tr.mean() - split[2].mean()) < 0.05
    assert subsample_train(split, None) is split


def test_feature_selection_keeps_k_columns(split):
    x_tr, x_te, y_tr, _ = split
    pre = build_preprocessor(x_tr, PreprocessConfig(feature_selection="kbest_f", selection_k=5))
    assert pre.fit(x_tr, y_tr).transform(x_te).shape[1] == 5


def test_feature_selection_rejects_unknown_method(split):
    x_tr, _, y_tr, _ = split
    with pytest.raises(ValueError, match="feature_selection"):
        build_preprocessor(x_tr, PreprocessConfig(feature_selection="boruta")).fit(x_tr, y_tr)


def test_l1_selection_is_reproducible_across_fits(split):
    # liblinear xáo thứ tự toạ độ bằng RNG: không cố định seed thì mỗi lần fit ra hệ số (và tập cột) khác
    x_tr, _, y_tr, _ = split
    cfg = PreprocessConfig(feature_selection="l1", selection_c=0.5)
    coefs = [
        build_preprocessor(x_tr, cfg).fit(x_tr, y_tr).named_steps["select"].estimator_.coef_
        for _ in range(2)
    ]
    np.testing.assert_array_equal(coefs[0], coefs[1])


def test_l1_selector_keeps_at_least_one_feature():
    import sklearn
    from sklearn.linear_model import LogisticRegression

    # như _l1_logreg: sklearn >= 1.8 thay `penalty="l1"` bằng `l1_ratio=1`; Colab có thể cũ hơn
    new_api = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
    l1 = {"l1_ratio": 1.0} if new_api else {"penalty": "l1"}
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(60, 4)), rng.integers(0, 2, 60)
    sel = L1Selector(LogisticRegression(solver="liblinear", C=1e-6, **l1)).fit(x, y)
    assert sel.transform(x).shape[1] == 1
