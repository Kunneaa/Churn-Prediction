"""Fixture dùng chung: bộ dữ liệu tổng hợp nhỏ có đủ cột mà pipeline cần, thư mục tạm cho settings."""

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from modules.shared import get_settings

mpl.use("Agg")

N_ROWS = 240


def make_raw(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Dữ liệu thô giả lập cùng lược đồ Telecom Customer Churn (gồm NaN và mã thiếu ẩn)."""
    rng = np.random.default_rng(seed)
    churn = rng.integers(0, 2, n)
    mou = rng.gamma(2.0, 200.0, n) * (1 + 0.3 * churn)
    df = pd.DataFrame(
        {
            "rev_Mean": mou * 0.1 + rng.normal(0, 5, n),
            "mou_Mean": mou,
            "ovrmou_Mean": rng.gamma(1.0, 20.0, n),
            "drop_blk_Mean": rng.gamma(1.0, 5.0, n),
            "attempt_Mean": rng.gamma(2.0, 50.0, n),
            "complete_Mean": rng.gamma(2.0, 40.0, n),
            "change_mou": rng.normal(0, 100, n),
            "avg3mou": rng.gamma(2.0, 200.0, n),
            "avg6mou": rng.gamma(2.0, 200.0, n),
            "avgrev": rng.gamma(2.0, 30.0, n),
            "change_rev": rng.normal(0, 10, n),
            "avg3rev": rng.gamma(2.0, 30.0, n),
            "avg6rev": rng.gamma(2.0, 30.0, n),
            "avgmou": rng.gamma(2.0, 200.0, n),
            "custcare_Mean": rng.gamma(1.0, 2.0, n),
            "unan_vce_Mean": rng.gamma(1.0, 20.0, n),
            "plcd_vce_Mean": rng.gamma(2.0, 50.0, n),
            "recv_vce_Mean": rng.gamma(2.0, 40.0, n),
            "ovrrev_Mean": rng.gamma(1.0, 5.0, n),
            "hnd_price": rng.choice([10.0, 60.0, 150.0], n),
            "models": rng.integers(1, 4, n).astype(float),
            "eqpdays": rng.integers(-3, 1500, n).astype(float),
            "months": rng.integers(6, 60, n),
            "income": rng.integers(1, 10, n).astype(float),
            "crclscod": rng.choice(["A", "AA", "B", "BA", "C", "Z5"], n),
            "area": rng.choice(["NYC", "LA", "MIDWEST"], n),
            "prizm_social_one": rng.choice(["S", "U", "C", "T"], n),
            "new_cell": rng.choice(["Y", "N", "U"], n),
            "marital": rng.choice(["M", "S", "U"], n),
            "ethnic": rng.choice(["N", "H", "U"], n),
            "dualband": rng.choice(["Y", "N", "U"], n),
            "hnd_webcap": rng.choice(["WCMB", "WC", "UNKW"], n),
            "churn": churn,
            "Customer_ID": np.arange(1_000_000, 1_000_000 + n),
        }
    )
    for col in ("income", "avg6mou", "hnd_webcap", "area"):
        df.loc[rng.random(n) < 0.1, col] = np.nan
    return df


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def settings_tmp(tmp_path, monkeypatch, raw):
    """Settings trỏ vào thư mục tạm; CSV thô giả lập đã nằm sẵn nên không tải mạng."""
    for key, sub in (
        ("DATA_DIR", "data"),
        ("FEATURES_DIR", "features"),
        ("REPORTS_DIR", "reports"),
    ):
        monkeypatch.setenv(f"CHURN_{key}", str(tmp_path / sub))
    get_settings.cache_clear()
    settings = get_settings()
    settings.data_dir.mkdir(parents=True)
    raw.to_csv(settings.data_dir / settings.raw_csv_name, index=False)
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def split(settings_tmp):
    from modules.preprocessing import load_dataset

    del settings_tmp
    return load_dataset(test_size=0.25, random_state=0)
