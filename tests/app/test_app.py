import importlib
import logging
import sys

import pandas as pd

from modules import app


def test_configure_runtime_sets_package_log_level():
    app.configure_runtime("DEBUG")
    assert logging.getLogger("modules").level == logging.DEBUG


def test_main_parses_config_and_runs(monkeypatch, tmp_path):
    called = {}

    def fake_run_all(path):
        called["path"] = path
        return {"final": pd.DataFrame({"model": ["m"], "roc_auc": [0.7]})}

    # `modules.app.main` là hàm (export trong __init__), nên lấy submodule qua importlib
    monkeypatch.setattr(importlib.import_module("modules.app.main"), "run_all", fake_run_all)
    monkeypatch.setattr(sys, "argv", ["churn", "--config", str(tmp_path / "x.yaml")])
    app.main()
    assert called["path"] == tmp_path / "x.yaml"
