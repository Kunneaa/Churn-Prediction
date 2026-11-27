# machine-learning — hướng dẫn cho AI

BTL Học máy HK261 Bài 1: pipeline ML truyền thống cấu hình được, dự đoán churn (Telecom Customer Churn 100k × 100), front-end Colab.

- Chuẩn **DevVault 1.0** (`.devvault.toml`) · Profile **Data Science & AI** · `AGENTS.md` → symlink file này · đọc trước vault `20-Projects/machine-learning/machine-learning — MOC.md`
- Graph: `graphify extract . --code-only --out ~/.cache/devgraph/machine-learning` sau khi đổi code, rồi `graphify query|affected`

## Cấu trúc (ADR-0001: không `src/`)
`modules/{app,shared,<module>}`, `tests/<module>/`, `notebooks/01-churn-pipeline.ipynb`, `features/`, `reports/`. DAG: experiments → evaluation · reporting → modeling → featurize · preprocessing → datasets. Chạy: `uv run churn --config configs/default.yaml`.

## Gate và quy trình
`gate` xanh trước commit; luật bước Phát triển (tiêu chí nghiệm thu, R-54, note module khi API đổi), Bảng định tuyến, Chia việc: vault `10-Standards/Project Lifecycle.md`. Chạm pipeline → `[golden] run`; đổi notebook/pipeline → thêm Run all bằng nbconvert. Không dùng `/simplify`, `/code-review`, `/init`, `engineering:*`. Dữ liệu riêng: `data/` (`.agents/roster.yaml`).

## Luật riêng
- Đề bài (`docs/*.pdf`, chỉ máy local) là ràng buộc cứng: Run all trên Colab chạy hết; không mount Drive; dữ liệu từ link công khai; notebook tự cài thư viện.
- Không nhìn test khi chọn cấu hình: mọi lựa chọn theo CV/OOF; test chỉ ở `final_stage`.
- Bước học tham số nằm trong `sklearn.Pipeline` (DS-03); ngoài pipeline chỉ biến đổi theo dòng.
- Colab 2 CPU: thí nghiệm nặng dùng `tuning.sample` hoặc CV 3-fold; không KNNImputer / SVC kernel mặc định.
- `data/raw/` chỉ đọc; số liệu báo cáo lấy từ `reports/results/*.csv`, không sửa tay. Đổi hành vi có chủ đích → thay `tests/golden/golden.txt` + Log.

## Khi viết code
Đúng module, không thư mục cấp cao mới; tối giản (R-33); import chéo qua `from modules.x import …`, cạnh có trong DAG.
