"""Lưu hình vào reports/figures/."""

from matplotlib.figure import Figure

from modules.shared import get_settings


def save_figure(fig: Figure, name: str | None) -> None:
    """Lưu `fig` thành `reports/figures/<name>.png` (bỏ qua nếu name rỗng)."""
    if not name:
        return
    figures_dir = get_settings().figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=120, bbox_inches="tight")
