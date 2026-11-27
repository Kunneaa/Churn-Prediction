"""MLP bằng PyTorch bọc thành estimator sklearn (phần mở rộng học sâu cho dữ liệu bảng).

Kiến trúc: [Linear -> BatchNorm -> ReLU -> Dropout] × n -> Linear(1); tối ưu AdamW với BCE,
dừng sớm theo loss trên tập validation tách từ dữ liệu fit, khôi phục trọng số tốt nhất.
torch được import lười trong fit/predict để các mô hình khác không phụ thuộc torch.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

_BATCH_SIZE = 512
_PATIENCE = 5  # số epoch liên tiếp loss validation không giảm thì dừng


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Phân loại nhị phân bằng mạng nơ-ron nhiều tầng (PyTorch, CPU)."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.random_state = random_state

    def _network(self, n_features: int) -> Any:
        from torch import nn

        layers: list[nn.Module] = []
        width = n_features
        for h in self.hidden_layer_sizes:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)

    def _train_epoch(self, net: Any, opt: Any, x: Any, y: Any, gen: Any) -> None:
        import torch

        net.train()
        order = torch.randperm(len(x), generator=gen)
        for start in range(0, len(x), _BATCH_SIZE):
            idx = order[start : start + _BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm cần ≥ 2 mẫu
                continue
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(x[idx]).squeeze(1), y[idx]
            )
            loss.backward()
            opt.step()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPClassifier":
        import torch

        # 1 luồng: tránh treo do xung đột OpenMP với lightgbm trên macOS (tái hiện 2026-11-22), và
        # cho kết quả tất định; mạng nhỏ nên gần như không chậm hơn.
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        x_fit, x_val, y_fit, y_val = train_test_split(
            x, y, test_size=0.1, stratify=y, random_state=self.random_state
        )
        xt, yt = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
        xv, yv = torch.from_numpy(x_val), torch.from_numpy(y_val)
        net = self._network(x.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.random_state)
        best_loss, best_state, bad = np.inf, None, 0
        self.n_epochs_ = 0
        for _ in range(self.max_epochs):
            self._train_epoch(net, opt, xt, yt, gen)
            self.n_epochs_ += 1
            net.eval()
            with torch.no_grad():
                val = torch.nn.functional.binary_cross_entropy_with_logits(net(xv).squeeze(1), yv)
            if val.item() < best_loss - 1e-4:
                best_loss, bad = val.item(), 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= _PATIENCE:
                    break
        if best_state is not None:  # loss validation luôn NaN -> giữ trọng số cuối thay vì lỗi
            net.load_state_dict(best_state)
        self.net_, self.best_val_loss_ = net, best_loss
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.from_numpy(np.asarray(x, dtype=np.float32))).squeeze(1)
        p = torch.sigmoid(logits).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
