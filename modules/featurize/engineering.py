"""Feature engineering theo từng dòng (stateless -> an toàn khi đặt trong cross-validation)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b > 0)


def add_features(x: pd.DataFrame) -> pd.DataFrame:
    """Thêm tỷ lệ / xu hướng sử dụng, cước, chất lượng dịch vụ và hạng tín dụng gộp từ cột gốc."""
    out = x.copy()
    out["rev_per_minute"] = _safe_div(out["rev_Mean"], out["mou_Mean"])  # cước thực tế / phút
    out["overage_share"] = _safe_div(out["ovrmou_Mean"], out["mou_Mean"])  # tỷ lệ phút vượt gói
    out["drop_block_rate"] = _safe_div(out["drop_blk_Mean"], out["attempt_Mean"])  # rớt/chặn
    out["complete_rate"] = _safe_div(out["complete_Mean"], out["attempt_Mean"])
    out["mou_change_pct"] = _safe_div(out["change_mou"], out["mou_Mean"])  # % đổi phút gọi
    out["mou_trend_3v6"] = _safe_div(out["avg3mou"], out["avg6mou"])  # xu hướng gần đây
    out["equipment_age_years"] = out["eqpdays"] / 365.0
    out["rev_vs_lifetime_avg"] = _safe_div(out["rev_Mean"], out["avgrev"])  # cước / trung bình
    # Xu hướng & chất lượng dịch vụ (thêm 2026-11-22: CV ROC-AUC GB +0.005 trên 80k train)
    out["rev_change_pct"] = _safe_div(out["change_rev"], out["rev_Mean"])  # % đổi cước
    out["rev_trend_3v6"] = _safe_div(out["avg3rev"], out["avg6rev"])  # xu hướng cước gần đây
    out["mou_vs_lifetime"] = _safe_div(out["mou_Mean"], out["avgmou"])  # phút gọi / trung bình đời
    out["custcare_rate"] = _safe_div(out["custcare_Mean"], out["attempt_Mean"])  # gọi CSKH / cuộc
    out["unanswered_rate"] = _safe_div(out["unan_vce_Mean"], out["plcd_vce_Mean"])
    out["recv_ratio"] = _safe_div(out["recv_vce_Mean"], out["plcd_vce_Mean"])  # gọi đến / gọi đi
    out["overage_rev_share"] = _safe_div(out["ovrrev_Mean"], out["rev_Mean"])  # cước vượt gói
    out["eqp_vs_tenure"] = _safe_div(out["eqpdays"], out["months"] * 30.4)  # máy cũ so với tuổi TB
    out["price_per_model"] = _safe_div(out["hnd_price"], out["models"])
    # Hạng tín dụng: chữ cái đầu của crclscod (54 mã -> ~10 nhóm)
    if "crclscod" in out.columns:
        out["credit_class"] = out["crclscod"].str[0]
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Bọc add_features() thành transformer (có tên cột đầu ra) để đặt trong Pipeline."""

    def fit(self, x: pd.DataFrame, y: object = None) -> "FeatureEngineer":
        del y  # không dùng nhãn
        self.feature_names_out_ = np.array(add_features(pd.DataFrame(x).head(5)).columns)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return add_features(pd.DataFrame(x))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        del input_features
        return self.feature_names_out_
