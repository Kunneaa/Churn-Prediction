"""Logging chuẩn cho toàn dự án. Dùng thay cho print().

- `setup_logging(level)` gọi ĐÚNG MỘT LẦN ở app/ (main() hoặc create_app()).
- Mọi nơi khác chỉ `log = get_logger(__name__)`.
- Log của dự án theo `level`; thư viện ngoài (httpx, urllib3...) giữ WARNING để khỏi nhiễu.
"""

import logging

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_PKG = __name__.split(".")[0]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(format=_FORMAT, level=logging.WARNING)
    logging.getLogger(_PKG).setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
