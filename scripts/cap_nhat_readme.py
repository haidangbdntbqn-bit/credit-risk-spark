"""Điền bảng kết quả vào README.md từ các file trong results/ (chạy sau khi chạy hết notebook).

Chạy trong thư mục credit-risk-spark:   python scripts/cap_nhat_readme.py
"""
import csv
import re
from pathlib import Path

RES = Path("results")
README = Path("README.md")


def doc_csv(ten):
    p = RES / ten
    if not p.exists():
        raise SystemExit(f"Thiếu {p}. Hãy chạy hết notebook credit_risk_spark.ipynb trước.")
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def so(x, chu_so=4):
    try:
        return f"{float(x):.{chu_so}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "–"


mo_hinh = doc_csv("bang_4_3_ket_qua_mo_hinh.csv")
thoi_gian = doc_csv("bang_4_4_so_sanh_thoi_gian.csv")
du_lieu = doc_csv("bang_4_1_thong_ke_du_lieu.csv")

tot_nhat = max(mo_hinh, key=lambda r: float(r["AUC_ROC"]))
tong_dong = sum(int(float(r["so_dong"])) for r in du_lieu)
tong_csv = sum(float(r["csv_MB"]) for r in du_lieu)
tong_pq = sum(float(r["parquet_MB"]) for r in du_lieu)

dong = [
    f"**Mô hình tốt nhất: {tot_nhat['Mo_hinh']}, AUC-ROC {so(tot_nhat['AUC_ROC'])}** trên tập kiểm tra.",
    "",
    "| Mô hình | AUC-ROC | AUC-PR | Precision | Recall | F1 | Huấn luyện (giây) |",
    "|---|---|---|---|---|---|---|",
]
for r in mo_hinh:
    dong.append(
        f"| {r['Mo_hinh']} | {so(r['AUC_ROC'])} | {so(r['AUC_PR'])} | {so(r.get('Precision'), 3)} "
        f"| {so(r.get('Recall'), 3)} | {so(r.get('F1'), 3)} | {so(r.get('Thoi_gian_huan_luyen_s'), 1)} |"
    )
dong += [
    "",
    f"Dữ liệu: **{tong_dong:,} dòng** từ {len(du_lieu)} bảng; CSV {tong_csv:,.0f} MB → Parquet {tong_pq:,.0f} MB "
    f"(giảm {100 * (1 - tong_pq / tong_csv):.0f}%).".replace(",", "."),
    "",
    "| Đọc và tổng hợp bảng installments_payments | Thời gian (giây) |",
    "|---|---|",
]
for r in thoi_gian:
    dong.append(f"| {r['Cach_xu_ly']} | {so(r['Thoi_gian_s'], 1)} |")

khoi = "<!-- KET_QUA_START -->\n" + "\n".join(dong) + "\n<!-- KET_QUA_END -->"
noi_dung = README.read_text(encoding="utf-8")
moi, n = re.subn(r"<!-- KET_QUA_START -->.*?<!-- KET_QUA_END -->", lambda _: khoi, noi_dung, flags=re.S)
if n != 1:
    raise SystemExit("Không tìm thấy cặp <!-- KET_QUA_START --> ... <!-- KET_QUA_END --> trong README.md")
README.write_text(moi, encoding="utf-8")
print("Đã cập nhật README.md")
print(dong[0])
