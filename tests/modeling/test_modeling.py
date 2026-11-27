import numpy as np
import pytest

from modules.modeling import (
    BASELINE,
    MODEL_LABELS,
    PipelineConfig,
    build_model,
    build_pipeline,
    build_preprocessing,
)


def test_with_creates_modified_copy():
    cfg = BASELINE.with_(name="x", scaler="minmax")
    assert cfg.scaler == "minmax" and BASELINE.scaler == "standard"


def test_preprocess_config_mirrors_fields():
    cfg = PipelineConfig(num_imputer="mean", log_transform=True, pca_variance=0.95)
    pre = cfg.preprocess_config()
    assert (pre.num_imputer, pre.log_transform, pre.pca_variance) == ("mean", True, 0.95)


def test_to_dict_is_flat_and_serialisable():
    d = BASELINE.with_(model_params={"C": 1}).to_dict()
    assert isinstance(d["model_params"], str) and isinstance(d["minmax_range"], str)


@pytest.mark.parametrize("name", list(MODEL_LABELS))
def test_build_model_for_every_registered_name(name):
    assert hasattr(build_model(BASELINE.with_(model=name)), "fit")


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError, match="không hỗ trợ"):
        build_model(BASELINE.with_(model="xgboost"))


def test_build_pipeline_fits_and_predicts_probabilities(split):
    x_tr, x_te, y_tr, _ = split
    model = build_pipeline(x_tr, BASELINE).fit(x_tr, y_tr)
    proba = model.predict_proba(x_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert build_preprocessing(x_tr, BASELINE).fit_transform(x_tr, y_tr).shape[0] == len(x_tr)
    assert np.isfinite(proba).all()


def test_torch_mlp_fits_predicts_and_is_deterministic():
    from sklearn.base import clone

    from modules.modeling import TorchMLPClassifier

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 6)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    # 3000 mẫu / batch 512 -> ~6 bước mỗi epoch
    model = TorchMLPClassifier(hidden_layer_sizes=(16,), lr=5e-3, max_epochs=40, random_state=1)
    proba = model.fit(x, y).predict_proba(x)
    assert proba.shape == (3000, 2) and np.allclose(proba.sum(axis=1), 1)
    assert (model.predict(x) == y).mean() > 0.8  # học được quan hệ tuyến tính đơn giản
    assert 1 <= model.n_epochs_ <= 40
    again = clone(model).fit(x, y).predict_proba(x)  # clone sklearn + cùng seed -> cùng kết quả
    np.testing.assert_allclose(proba, again, rtol=1e-5)


def test_torch_mlp_in_pipeline(split):
    x_tr, x_te, y_tr, _ = split
    cfg = BASELINE.with_(
        model="torch_mlp", model_params={"hidden_layer_sizes": [8], "max_epochs": 3}
    )
    proba = build_pipeline(x_tr, cfg).fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_dummy_baseline_ignores_features(split):
    from sklearn.metrics import roc_auc_score

    x_tr, x_te, y_tr, y_te = split
    proba = build_pipeline(x_tr, BASELINE.with_(model="dummy")).fit(x_tr, y_tr).predict_proba(x_te)
    assert np.allclose(proba[:, 1], y_tr.mean())  # xác suất = tỷ lệ churn của train
    assert roc_auc_score(y_te, proba[:, 1]) == 0.5


def test_torch_mlp_survives_nan_validation_loss():
    from modules.modeling import TorchMLPClassifier

    x = np.full((200, 3), np.nan, dtype=np.float32)  # loss luôn NaN -> không có trọng số "tốt nhất"
    y = np.tile([0, 1], 100)
    model = TorchMLPClassifier(hidden_layer_sizes=(4,), max_epochs=2).fit(x, y)
    assert model.predict_proba(x).shape == (200, 2)
