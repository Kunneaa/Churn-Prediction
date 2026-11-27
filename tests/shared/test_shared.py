from pathlib import Path

from modules.shared import Settings, get_logger, get_settings, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_paths_anchor_to_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # chạy từ thư mục khác gốc repo (vd. notebooks/)
    s = Settings()
    assert s.data_dir == PROJECT_ROOT / "data" / "raw"
    assert s.experiment_config == PROJECT_ROOT / "configs" / "default.yaml"
    assert s.experiment_config.exists()


def test_absolute_override_is_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_DATA_DIR", str(tmp_path / "d"))
    assert Settings().data_dir == tmp_path / "d"
    assert Settings().results_dir == Settings().reports_dir / "results"


def test_get_settings_cached_and_logger():
    assert get_settings() is get_settings()
    setup_logging("INFO")
    assert get_logger("modules.x").name == "modules.x"
