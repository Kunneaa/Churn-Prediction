"""Golden cho machine-learning: metric + băm đặc trưng trên mẫu 20k, seed 42.

Chạy từ gốc repo (cần data/raw — dữ liệu riêng; chưa có thì load_dataset tự tải qua mạng):
    uv run python -B tests/golden/golden.py 2>/dev/null | diff tests/golden/golden.txt -
Không in gì = khớp. Log đi stderr, chỉ so stdout; không ghi file nào vào repo (-B: không bytecode).
Mốc `golden.txt` là bản tóm tắt: shape, metric CV/test 10 chữ số, 16 ký tự đầu sha256 ma trận đặc trưng
(làm tròn 8 chữ số) của 9 cấu hình + ngưỡng F1 OOF — không chứa dữ liệu thô.
Đổi hành vi pipeline có chủ đích -> thay `golden.txt` (mốc cũ nằm trong lịch sử git) + ghi Log trong vault.
"""

import hashlib
import warnings

import numpy as np

from modules.evaluation import run_experiment, tune_threshold
from modules.modeling import BASELINE
from modules.preprocessing import load_dataset

warnings.filterwarnings("ignore")

data = load_dataset(sample=20000)
x_train, x_test, y_train, y_test = data
print("shape", x_train.shape, x_test.shape, int(y_train.sum()), int(y_test.sum()))
tree = {"scaler": "none"}
cfgs = [
    BASELINE.with_(name="baseline"),
    BASELINE.with_(
        name="target_log_minmax",
        high_card_encoder="target",
        log_transform=True,
        scaler="minmax",
        minmax_range=(-1.0, 1.0),
        cat_imputer="constant",
    ),
    BASELINE.with_(
        name="ordinal_pca95_noind",
        cat_encoder="ordinal",
        pca_variance=0.95,
        add_missing_indicator=False,
        num_imputer="mean",
    ),
    BASELINE.with_(name="svm", model="linear_svm"),
    BASELINE.with_(name="knn_nb", model="knn", treat_hidden_missing=True),
    BASELINE.with_(name="nb", model="naive_bayes"),
    BASELINE.with_(name="rf", model="random_forest", **tree),
    BASELINE.with_(name="hgb_nocw", model="hist_gb", class_weight=None, **tree),
    BASELINE.with_(name="mlp", model="mlp", model_params={"hidden_layer_sizes": (32,)}),
]
for cfg in cfgs:
    row, fitted = run_experiment(cfg, data, cv=3)
    keys = sorted(k for k in row if k.startswith(("cv_", "test_")) and not k.endswith("time_s"))
    print(
        cfg.name, row["n_features_after_preprocess"], " ".join(f"{k}={row[k]:.10f}" for k in keys)
    )
    z = np.asarray(fitted[:-1].transform(x_train), dtype=np.float64)
    print(cfg.name, "Xsha", hashlib.sha256(np.round(z, 8).tobytes()).hexdigest()[:16], z.shape)
thr, curve = tune_threshold(BASELINE, x_train, y_train, cv=3)
print("threshold", thr, f"{curve.f1.max():.10f}")
