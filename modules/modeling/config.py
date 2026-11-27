"""PipelineConfig: toàn bộ lựa chọn của một thí nghiệm (làm sạch, tiền xử lý, mô hình)."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from modules.preprocessing import PreprocessConfig

RANDOM_STATE = 42


@dataclass(frozen=True)
class PipelineConfig:
    """Mọi lựa chọn của pipeline. Mỗi thí nghiệm = một PipelineConfig."""

    name: str = "baseline"
    # 1. Làm sạch: coi các mã "U"/"UNKW" (unknown) và eqpdays < 0 là giá trị thiếu
    treat_hidden_missing: bool = True
    # 2. Imputation
    num_imputer: str = "median"
    cat_imputer: str = "most_frequent"
    add_missing_indicator: bool = True
    # 3. Encoding
    cat_encoder: str = "onehot"
    high_card_encoder: str = "frequency"
    # 4. Biến đổi & scaling
    log_transform: bool = False
    scaler: str = "standard"
    minmax_range: tuple[float, float] = (0.0, 1.0)
    # 5. Chọn đặc trưng & giảm chiều
    feature_selection: str | None = None
    selection_k: int = 40
    selection_c: float = 0.05
    pca_variance: float | None = None
    # 6. Mô hình (tên hợp lệ: MODEL_LABELS trong modules/modeling/registry.py)
    model: str = "logreg"
    model_params: dict[str, Any] = field(default_factory=dict)
    class_weight: str | None = "balanced"

    def with_(self, **changes: Any) -> "PipelineConfig":
        """Tạo cấu hình mới từ cấu hình hiện tại, chỉ đổi một vài tham số."""
        return replace(self, **changes)

    def preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            num_imputer=self.num_imputer,
            cat_imputer=self.cat_imputer,
            add_missing_indicator=self.add_missing_indicator,
            cat_encoder=self.cat_encoder,
            high_card_encoder=self.high_card_encoder,
            log_transform=self.log_transform,
            scaler=self.scaler,
            minmax_range=self.minmax_range,
            feature_selection=self.feature_selection,
            selection_k=self.selection_k,
            selection_c=self.selection_c,
            pca_variance=self.pca_variance,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["minmax_range"] = str(self.minmax_range)
        d["model_params"] = str(self.model_params)
        return d


BASELINE = PipelineConfig()
