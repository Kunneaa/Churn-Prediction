"""Ghép pipeline đầy đủ: feature engineering -> tiền xử lý -> mô hình."""

import pandas as pd
from sklearn.pipeline import Pipeline

from modules.featurize import FeatureEngineer, add_features
from modules.preprocessing import build_preprocessor

from .config import PipelineConfig
from .registry import build_model


def build_preprocessing(x_sample: pd.DataFrame, cfg: PipelineConfig) -> Pipeline:
    """Feature engineering + impute/encode/scale/PCA. `x_sample` chỉ để lấy tên/kiểu cột."""
    columns = add_features(x_sample.head(50))
    steps = build_preprocessor(columns, cfg.preprocess_config()).steps
    return Pipeline([("engineer", FeatureEngineer()), *steps])


def build_pipeline(x_sample: pd.DataFrame, cfg: PipelineConfig) -> Pipeline:
    """Pipeline fit trên train: mọi thống kê (imputer, encoder, scaler, PCA) đều học từ train."""
    return Pipeline(
        [("preprocess", build_preprocessing(x_sample, cfg)), ("model", build_model(cfg))]
    )
