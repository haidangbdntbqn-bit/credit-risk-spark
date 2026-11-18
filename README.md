# Dự đoán rủi ro vỡ nợ tín dụng từ dữ liệu đa nguồn với Apache Spark

Đồ án học phần Nhập môn Big Data · GVHD: TS. Ngô Dương Hà
Nhóm: Nguyễn Hải Đăng (2001230172), Nguyễn Bình An (2001230003)

| Thành phần | Công nghệ |
|---|---|
| Lưu trữ phân tán | HDFS (Hadoop 3, một máy) · định dạng Parquet |
| Xử lý | Apache Spark 3.5: Spark SQL, DataFrame, MLlib (local mode) |
| Mô hình | Logistic Regression (baseline), Random Forest, Gradient-Boosted Trees |
| Trực quan hóa | Jupyter Notebook (matplotlib) và dashboard Streamlit (`app.py`) |
| Dữ liệu | Home Credit Default Risk — https://www.kaggle.com/competitions/home-credit-default-risk/data |

## Cấu trúc thư mục

```
credit-risk-spark/
├── credit_risk_spark.ipynb   # toàn bộ pipeline Spark
├── app.py                    # dashboard Streamlit
├── scripts/hdfs_setup.sh     # cài, chạy HDFS và đưa CSV lên HDFS
├── requirements.txt
├── data/                     # 7 file CSV tải từ Kaggle (không nộp kèm vì 2,5 GB)
└── results/                  # notebook tự tạo: bảng số liệu, hình, mô hình, dữ liệu dashboard
```

## 1. Cài Java 17

```bash
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
```

Thêm dòng `export` vào `~/.zshrc` để khỏi gõ lại. Kiểm tra: `$JAVA_HOME/bin/java -version` in ra `17`.

## 2. Môi trường Python

```bash
cd credit-risk-spark
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải 7 file CSV từ Kaggle (bấm Join Competition, rồi Download All), giải nén vào `data/`:
`application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
`POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`.

## 3. Cài và chạy HDFS

```bash
bash scripts/hdfs_setup.sh
```

Script cài Hadoop bằng Homebrew, cấu hình HDFS một máy (`hdfs://localhost:9000`, hệ số sao chép 1),
khởi động NameNode, DataNode và đưa 7 file CSV lên `/credit-risk/raw`.
Xem dữ liệu trên HDFS tại **http://localhost:9870** → Utilities → Browse the file system.

Những lần sau chỉ cần bật lại HDFS:

```bash
hdfs --daemon start namenode && hdfs --daemon start datanode
```

Chưa cài được HDFS? Đặt `STORAGE = "local"` ở ô cấu hình của notebook; notebook sẽ đọc `data/` và ghi `parquet/` trên máy.

## 4. Chạy notebook

```bash
jupyter notebook credit_risk_spark.ipynb
```

- Chạy thử: đặt `SAMPLE_FRACTION = 0.2`, chọn Run → Run All Cells.
- Chạy thật: đặt `SAMPLE_FRACTION = 1.0`, chọn Kernel → Restart & Run All (khoảng 20–40 phút trên MacBook Air M4).
- Trong lúc chạy mở **http://localhost:4040** (Spark UI) để xem job, stage, task.

## 5. Mở dashboard

```bash
streamlit run app.py
```

Dashboard có 4 tab: tổng quan dữ liệu (CSV và Parquet, thời gian Spark và pandas), khám phá dữ liệu,
đánh giá mô hình (đường ROC, kéo ngưỡng duyệt vay để xem Precision, Recall, ma trận nhầm lẫn) và
chấm điểm một hồ sơ bằng mô hình Spark đã lưu.

## 6. Kết quả notebook sinh ra

| File | Nội dung |
|---|---|
| `results/ket_qua.json` | Toàn bộ số liệu: số dòng, chất lượng dữ liệu, tỷ lệ vỡ nợ, thời gian, chỉ số mô hình |
| `results/bang_4_1_thong_ke_du_lieu.csv` | Số dòng, dung lượng CSV và Parquet, số khối HDFS |
| `results/bang_chat_luong_*.csv` | Trùng lặp, giá trị thiếu, ngoại lệ |
| `results/bang_4_3_ket_qua_mo_hinh.csv` | AUC-ROC, AUC-PR, Precision, Recall, F1, thời gian huấn luyện |
| `results/bang_4_4_so_sanh_thoi_gian.csv` | pandas và Spark |
| `results/bang_4_5_do_quan_trong_dac_trung.csv` | 20 đặc trưng quan trọng nhất |
| `results/figures/*.png` | Hình cho báo cáo |
| `results/dashboard/*.csv` | Dữ liệu cho `app.py` |

## Lỗi thường gặp

| Lỗi | Cách sửa |
|---|---|
| `JAVA_GATEWAY_EXITED` | Chưa đặt `JAVA_HOME` (bước 1) |
| `Connection refused localhost:9000` | HDFS chưa chạy: `hdfs --daemon start namenode && hdfs --daemon start datanode` |
| `OutOfMemoryError` | Giảm `SAMPLE_FRACTION` hoặc đổi `spark.driver.memory` thành `6g` |
| GBT chạy quá lâu | Giảm `maxIter` của GBT từ 60 xuống 30 |
| Dashboard báo thiếu `ket_qua.json` | Chạy hết notebook trước |
