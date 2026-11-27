# Dự đoán khách hàng rời bỏ (Customer Churn) — Pipeline học máy truyền thống

Bài tập lớn môn **Học máy** — **Bài 1: Học máy với dữ liệu dạng bảng (Tabular Data)**.

| | |
|---|---|
| Môn học | Học máy (Machine Learning) — Mã môn học: `<MÃ MÔN>` |
| Học kỳ | Học kỳ I, năm học 2026–2027 |
| Đơn vị | Khoa Khoa học và Kỹ thuật Máy tính, Trường ĐH Bách Khoa – ĐHQG-HCM |
| Giảng viên hướng dẫn | `<Họ tên GVHD>` — `<email>` |
| Báo cáo PDF | [`reports/report.pdf`](reports/report.pdf) (nguồn: `reports/report.md`) |
| Mã nguồn | https://github.com/Kunneaa/Churn-Prediction |
| Colab notebook | [Mở trên Colab](https://colab.research.google.com/github/Kunneaa/Churn-Prediction/blob/main/notebooks/01-churn-pipeline.ipynb) |

### Thành viên nhóm

| Họ và tên | MSSV | Email | Đóng góp |
|---|---|---|---|
| `<Thành viên 1>` | `<MSSV>` | `<email>` | `<%>` |
| `<Thành viên 2>` | `<MSSV>` | `<email>` | `<%>` |
| `<Thành viên 3>` | `<MSSV>` | `<email>` | `<%>` |

---

## 1. Mục tiêu

Xây dựng **pipeline học máy truyền thống cấu hình được** cho bài toán phân loại nhị phân: dự đoán khách hàng
viễn thông có rời bỏ nhà mạng (`churn = 1`) hay không, gồm đầy đủ các bước:

1. **EDA** — thống kê mô tả, giá trị thiếu (tường minh / ẩn / có cấu trúc), ngoại lai, phân phối, tương quan.
2. **Tiền xử lý** — làm sạch, imputation, encoding (kể cả cột nhiều giá trị), log-transform, scaling, chia train/test.
3. **Trích xuất & lựa chọn đặc trưng** — feature engineering; chọn đặc trưng (ANOVA F, mutual information, L1);
   PCA (giữ 90% / 95% / 99% phương sai).
4. **Huấn luyện** — mốc Dummy, Logistic Regression, Linear SVM, k-NN, Naive Bayes, Random Forest, Gradient Boosting, LightGBM.
5. **Đánh giá** — Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC; so sánh nhiều cấu hình, kiểm định thống kê,
   phân tích lỗi theo nhóm khách hàng.
6. **Mở rộng (điểm cộng)** — mạng nơ-ron MLP (sklearn và PyTorch) so sánh với pipeline truyền thống.

## 2. Dữ liệu

| | |
|---|---|
| Tên | Telecom Customer Churn |
| Nguồn gốc | Teradata Center for Customer Relationship Management, Duke University |
| Trang dataset | https://www.kaggle.com/datasets/abhinav89/telecom-customer |
| Link tải (công khai, không cần đăng nhập) | https://www.kaggle.com/api/v1/datasets/download/abhinav89/telecom-customer |
| Kích thước | 100.000 khách hàng × 100 cột (78 số, 21 phân loại, 1 nhãn) |
| Đặc trưng `.npy` | không vào git (khoảng 50–95 MB mỗi biến thể); sinh lại bằng notebook hoặc `uv run churn`, đóng vào file zip nộp bài |
| Nhãn | `churn` (0/1) — 49,6% churn |
| Giá trị thiếu | 43 cột có `NaN` (từ 1 dòng đến 49,4% số dòng), cùng các mã unknown `"U"`/`"UNKW"` và `eqpdays < 0` |
| Cột phân loại | 21 cột; `crclscod` có 54 giá trị (high cardinality) |

Notebook **tự tải và giải nén** dữ liệu vào `data/raw/` — không mount Google Drive hay dịch vụ cloud cá nhân.

## 3. Hướng dẫn chạy

### Google Colab (front-end nộp bài)
1. Mở `notebooks/01-churn-pipeline.ipynb` trên Colab.
   Link trực tiếp: [colab.research.google.com/github/Kunneaa…](https://colab.research.google.com/github/Kunneaa/Churn-Prediction/blob/main/notebooks/01-churn-pipeline.ipynb)
2. Cell đầu tiên tự `git clone` repo này (`REPO_URL`) để lấy package `modules/` rồi `pip install -r requirements.txt`
   (chỉ cài thêm `pydantic-settings`, phần còn lại Colab có sẵn).
3. Chọn **Runtime → Run all**. Không mount Google Drive; dữ liệu tự tải từ link công khai ở mục 2.

### Máy cá nhân (uv)
```bash
uv sync                                        # Python 3.12, phiên bản khoá trong uv.lock
uv run churn --config configs/default.yaml     # tái tạo bảng kết quả, hình 09–14 và features/ (hình EDA 01–08 sinh từ notebook)
uv run jupyter notebook notebooks/01-churn-pipeline.ipynb   # hoặc chạy notebook
```
Không dùng uv: `pip install -r requirements.txt jupyter`, rồi chạy notebook.

### Kiểm tra chất lượng
```bash
uv run pytest                                  # 88 test trên dữ liệu tổng hợp (~1 phút)
uv run ruff check . && uv run ruff format --check .
uv run lint-imports                            # luật kiến trúc (DAG module)
uv run python scripts/check_structure.py       # luật cấu trúc
```
Hook pre-commit chạy các kiểm tra trên mỗi commit (`uv run pre-commit install`).

## 4. Cấu trúc thư mục

```text
.
├── notebooks/
│   └── 01-churn-pipeline.ipynb  # front-end: EDA → thí nghiệm → đánh giá → lưu đặc trưng → MLP (chỉ gọi modules/)
├── modules/                     # package Python (đề bài bắt buộc tên thư mục — ADR-0001)
│   ├── app/                     # điểm vào: CLI `churn`, configure_runtime() cho notebook
│   ├── datasets/                # tải dữ liệu public, đọc CSV thô, lược đồ + mã thiếu ẩn
│   ├── preprocessing/           # làm sạch, chia train/test, imputer/encoder/log/scaler/chọn đặc trưng/PCA cấu hình được
│   ├── featurize/               # feature engineering + lưu/nạp đặc trưng .npy
│   ├── modeling/                # PipelineConfig, danh mục mô hình, ghép pipeline
│   ├── evaluation/              # cross-validation, chỉ số, chọn ngưỡng, RandomizedSearchCV, kiểm định, chỉ số theo nhóm
│   ├── reporting/               # bảng & biểu đồ EDA / kết quả
│   ├── experiments/             # các giai đoạn thí nghiệm 3.1 → 7, run_all
│   └── shared/                  # cấu hình (.env, tiền tố CHURN_), logging
├── configs/default.yaml         # tham số thí nghiệm: số fold, không gian tuning, kiến trúc MLP
├── features/                    # đặc trưng đã trích xuất (.npy sinh ra khi chạy, meta.json)
├── reports/
│   ├── figures/                 # biểu đồ sinh từ notebook / CLI
│   └── results/                 # bảng kết quả các thí nghiệm (.csv)
├── tests/<module>/              # pytest, phản chiếu modules/
├── scripts/                     # check_structure.py (luật cấu trúc), build_report.py (report.md -> PDF)
├── docs/                        # đề bài (chỉ ở máy local, không đưa lên git)
├── pyproject.toml · uv.lock · requirements.txt · .env.example · CLAUDE.md
└── README.md
```
Phụ thuộc giữa các module được máy kiểm tra (`lint-imports`):
`app → experiments → evaluation | reporting → modeling → featurize | preprocessing → datasets → shared`.

## 5. Pipeline cấu hình được

Mỗi thí nghiệm là một `PipelineConfig` (`modules/modeling/config.py`):

| Bước | Tham số | Lựa chọn |
|---|---|---|
| Làm sạch | `treat_hidden_missing` | `True` / `False` |
| Imputation | `num_imputer`, `cat_imputer`, `add_missing_indicator` | `median` / `mean` / `most_frequent` / `knn`; `most_frequent` / `constant`; có / không cờ thiếu |
| Encoding | `cat_encoder`, `high_card_encoder` | `onehot` / `ordinal`; `frequency` / `target` / `onehot` / `drop` |
| Biến đổi | `log_transform` | `sign(x)·log1p(\|x\|)` hoặc không |
| Scaling | `scaler`, `minmax_range` | `standard` / `minmax` (tuỳ chỉnh `feature_range`) / `none` |
| Chọn đặc trưng | `feature_selection`, `selection_k`, `selection_c` | `None` / `kbest_f` / `kbest_mi` / `l1` |
| Giảm chiều | `pca_variance` | `None` / `0.90` / `0.95` / `0.99` |
| Mô hình | `model`, `model_params`, `class_weight` | `dummy` (mốc), `logreg`, `linear_svm`, `knn`, `naive_bayes`, `random_forest`, `hist_gb`, `lightgbm`, `mlp`, `torch_mlp` |

```python
from modules.evaluation import run_experiment
from modules.modeling import BASELINE
from modules.preprocessing import load_dataset

data = load_dataset()  # (x_train, x_test, y_train, y_test)
cfg = BASELINE.with_(name="rf_pca95", model="random_forest", scaler="none", pca_variance=0.95)
row, fitted = run_experiment(cfg, data, cv=3)  # CV trên train + đo trên test
```

Mọi bước học tham số nằm trong một `sklearn.Pipeline`, nên khi cross-validation chúng chỉ được fit trên phần train
của từng fold (**không rò rỉ dữ liệu**). Cấu hình được chọn bằng CV trên tập train; tập test (20%) chỉ dùng một lần
để báo cáo kết quả cuối.

## 6. Kết quả chính

Kết quả trên **tập test** (20.000 khách hàng, giữ riêng, chỉ dùng một lần). Ngưỡng được chọn bằng xác suất out-of-fold
trên train (tối đa balanced accuracy ≡ Youden J). Bảng đầy đủ ở `reports/results/`, hình ở `reports/figures/`.

| Mô hình | ROC-AUC | PR-AUC | Accuracy (0.5) | F1 (0.5) | Ngưỡng OOF | Accuracy / Recall ở ngưỡng đó |
|---|---|---|---|---|---|---|
| **LightGBM** (chọn theo CV) | **0.705** | **0.697** | **0.645** | **0.646** | 0.47 | 0.644 / 0.718 |
| Gradient Boosting (Hist, tuned) | 0.701 | 0.692 | 0.642 | 0.643 | 0.49 | 0.642 / 0.677 |
| Random Forest (tuned) | 0.686 | 0.677 | 0.628 | 0.634 | 0.51 | 0.628 / 0.614 |
| MLP PyTorch — mở rộng học sâu | 0.669 | 0.658 | 0.621 | 0.611 | 0.47 | 0.619 / 0.667 |
| MLP sklearn — mở rộng học sâu | 0.658 | 0.649 | 0.612 | 0.598 | 0.50 | 0.612 / 0.582 |
| Logistic Regression (tuned) | 0.651 | 0.638 | 0.610 | 0.603 | 0.49 | 0.609 / 0.626 |
| Mốc Dummy (đoán theo tỷ lệ lớp) | 0.500 | 0.496 | 0.504 | 0.000 | — | — |

Những phát hiện chính (chọn bằng CV 3-fold trên train theo quy tắc một sai số chuẩn; kiểm định bằng CV lặp + t-test hiệu
chỉnh Nadeau–Bengio):
- **Cờ giá trị thiếu** (p = 0.0003) và **log-transform trước khi scale** (p = 0.001) cải thiện có ý nghĩa thống kê; cách
  **encoding không tạo khác biệt** (p = 0.80) nên giữ one-hot + frequency (đơn giản, đúng bản chất biến định danh).
- **PCA làm giảm** ROC-AUC (0.643 → 0.628 khi giữ 90% phương sai, p = 0.005 ở mức 95%); **lọc đặc trưng theo ANOVA F** cũng
  làm giảm, còn **L1** giữ 158/231 đặc trưng gần như không mất gì ⇒ cấu hình cuối không giảm chiều.
- **Mô hình cây tăng cường** hơn **Random Forest** (p = 0.003) **và Logistic Regression** (+0.045, p < 0.0001); LightGBM
  và Gradient Boosting tương đương trên CV (chênh 0.0006). Hai MLP đứng giữa, MLP PyTorch (BatchNorm + dropout) tốt hơn
  MLP sklearn.
- **Chọn ngưỡng:** với 49.6% churn, đoán "churn" cho mọi khách đã có F1 = 0.663, nên tối đa F1 đẩy ngưỡng xuống 0.23–0.38
  (recall ≈ 0.9, accuracy tụt về 53–61%). Báo cáo dùng ngưỡng tối đa balanced accuracy (0.47–0.51); bảng có đủ cả ba quy tắc.
- Đặc trưng quan trọng nhất: `mou_Mean`, `change_mou`, `eqpdays`, `months`. Mô hình yếu nhất ở khách 13–24 tháng và khách
  có thiết bị trên 2 năm (`reports/results/exp8_subgroups.csv`).

Thời gian chạy toàn bộ notebook: 17–22 phút trên máy 10 nhân (sklearn 1.9.1, LightGBM 4.7, torch 2.15). Trên Colab (2 CPU) có
thể lâu hơn vài lần; giảm `stage_sample` / `tuning.sample` trong `configs/default.yaml` nếu cần.
