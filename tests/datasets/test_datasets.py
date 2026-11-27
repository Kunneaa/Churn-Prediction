import http.client
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from modules.datasets import (
    HIDDEN_MISSING_CODES,
    TARGET,
    download_data,
    hidden_missing_mask,
    load_raw,
    raw_csv_path,
)


def test_hidden_missing_mask_flags_unknown_codes_only(raw):
    mask = hidden_missing_mask(raw)
    for col, codes in HIDDEN_MISSING_CODES.items():
        assert mask[col].equals(raw[col].isin(codes))
    assert not mask["prizm_social_one"].any()  # "U" = Urban, không phải giá trị thiếu
    assert mask["eqpdays"].equals(raw["eqpdays"] < 0)


def test_hidden_missing_mask_handles_missing_columns():
    mask = hidden_missing_mask(pd.DataFrame({"x": [1, 2]}))
    assert not mask.to_numpy().any()


def test_raw_csv_path_and_load_raw(settings_tmp, raw):
    assert raw_csv_path() == settings_tmp.data_dir / settings_tmp.raw_csv_name
    loaded = load_raw()
    assert loaded.shape == raw.shape
    assert TARGET in loaded.columns


def test_download_data_skips_when_file_exists(settings_tmp):
    assert download_data() == raw_csv_path()


def test_download_falls_back_to_curl_on_incomplete_read(settings_tmp, monkeypatch, raw):
    raw_csv_path().unlink()  # buộc phải tải lại

    def broken_urlopen(*args, **kwargs):
        raise http.client.IncompleteRead(b"partial")

    def fake_curl(cmd, check):
        zip_path = Path(cmd[cmd.index("-o") + 1])
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(settings_tmp.raw_csv_name, raw.to_csv(index=False))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(urllib.request, "urlopen", broken_urlopen)
    monkeypatch.setattr(subprocess, "run", fake_curl)
    assert download_data() == raw_csv_path()
    assert load_raw().shape == raw.shape
