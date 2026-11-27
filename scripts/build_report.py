"""Dựng báo cáo PDF nộp bài từ `reports/report.md` (Markdown -> HTML -> PDF bằng Chrome headless).

Chạy (từ gốc repo, sau khi notebook / `uv run churn` đã sinh hình ở reports/figures/):
    uv run --with markdown-it-py python scripts/build_report.py
Ra: reports/report.pdf (nộp kèm) và reports/report.html (bản trung gian, không vào git).
Cần Google Chrome hoặc Chromium; đặt đường dẫn khác bằng biến môi trường CHROME.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "report.md"
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
)
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Times New Roman", serif; font-size: 11.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt;
     break-after: avoid; }
h3 { font-size: 12.5pt; margin-top: 12pt; break-after: avoid; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 10pt; break-inside: avoid; }
th, td { border: 1px solid #888; padding: 2pt 6pt; }
th { background: #eee; }
img { max-width: 100%; max-height: 120mm; display: block; margin: 6pt auto; break-inside: avoid; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.5pt; }
pre { background: #f5f5f5; padding: 6pt; font-size: 9pt; white-space: pre-wrap; }
blockquote { color: #444; border-left: 3px solid #bbb; margin-left: 0; padding-left: 8pt; }
.page-break { break-after: page; }
"""


def _chrome() -> str:
    for c in (os.environ.get("CHROME"), *CHROME_CANDIDATES):
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    sys.exit("Không tìm thấy Chrome/Chromium — đặt biến môi trường CHROME")


def main() -> None:
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(SOURCE.read_text())
    html_path = SOURCE.with_suffix(".html")
    html_path.write_text(
        f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>{CSS}</style>'
        f"</head><body>{body}</body></html>",
        encoding="utf-8",
    )
    pdf_path = SOURCE.with_suffix(".pdf")
    subprocess.run(
        [
            _chrome(),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    sys.stdout.write(f"{pdf_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
