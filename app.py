"""Dashboard rủi ro tín dụng — đọc kết quả notebook và chấm điểm hồ sơ bằng mô hình Spark đã lưu.

Chạy:  streamlit run app.py
Cần chạy notebook credit_risk_spark.ipynb trước để có thư mục results/.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RES = Path("results")
DASH = RES / "dashboard"
BLUE, RED, GREY = "#2072B7", "#E81F26", "#9AA5B1"

st.set_page_config(page_title="Rủi ro tín dụng · Apache Spark", layout="wide")
st.title("Dashboard dự đoán rủi ro vỡ nợ tín dụng")
st.caption("Dữ liệu Home Credit · xử lý bằng Apache Spark (HDFS, Spark SQL, MLlib) · Nhóm: Nguyễn Hải Đăng, Nguyễn Bình An · GVHD: TS. Ngô Dương Hà")

if not (RES / "ket_qua.json").exists():
    st.error("Chưa có results/ket_qua.json. Hãy chạy hết notebook credit_risk_spark.ipynb trước.")
    st.stop()


@st.cache_data
def load():
    summary = json.loads((RES / "ket_qua.json").read_text(encoding="utf-8"))
    read = lambda p: pd.read_csv(p) if p.exists() else None
    return {
        "summary": summary,
        "stats": read(RES / "bang_4_1_thong_ke_du_lieu.csv"),
        "metrics": read(RES / "bang_4_3_ket_qua_mo_hinh.csv"),
        "time": read(RES / "bang_4_4_so_sanh_thoi_gian.csv"),
        "fi": read(RES / "bang_4_5_do_quan_trong_dac_trung.csv"),
        "target": read(DASH / "eda_target.csv"),
        "edu": read(DASH / "eda_hoc_van.csv"),
        "inc": read(DASH / "eda_thu_nhap.csv"),
        "age": read(DASH / "eda_tuoi.csv"),
        "ext": read(DASH / "eda_ext_source.csv"),
        "miss": read(DASH / "gia_tri_thieu.csv"),
        "pred": read(DASH / "du_doan_tap_kiem_tra.csv"),
    }


D = load()
S = D["summary"]
MODELS = [m["Mo_hinh"] for m in S["ket_qua_mo_hinh"]]
BEST = S["mo_hinh_tot_nhat"]


def risk_level(p):
    return "Cao" if p >= 0.7 else ("Trung bình" if p >= 0.4 else "Thấp")


def bar(df, x, y, title, color=BLUE, horizontal=True, fmt=".1f"):
    if horizontal:
        enc = alt.Chart(df).mark_bar(color=color).encode(
            y=alt.Y(f"{x}:N", sort="-x", title=None), x=alt.X(f"{y}:Q", title=title),
            tooltip=[x, alt.Tooltip(y, format=fmt)])
    else:
        enc = alt.Chart(df).mark_bar(color=color).encode(
            x=alt.X(f"{x}:N", title=None, sort=None), y=alt.Y(f"{y}:Q", title=title),
            tooltip=[x, alt.Tooltip(y, format=fmt)])
    return enc.properties(height=max(220, 28 * len(df)) if horizontal else 300)


tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan dữ liệu", "Khám phá dữ liệu", "Đánh giá mô hình", "Chấm điểm hồ sơ"])

# ---------------- 1. Tổng quan
with tab1:
    st_ = D["stats"]
    c = st.columns(4)
    c[0].metric("Số bảng dữ liệu", len(st_) if st_ is not None else 7)
    c[1].metric("Tổng số dòng", f"{int(st_.so_dong.sum()):,}" if st_ is not None else "-")
    c[2].metric("Hồ sơ dùng để huấn luyện", f"{S['so_ho_so']:,}")
    c[3].metric("Tỷ lệ vỡ nợ", f"{100 * S['ti_le_vo_no']:.2f}%")
    if st_ is not None:
        c = st.columns(3)
        csv_mb, pq_mb = st_.csv_MB.sum(), st_.parquet_MB.sum()
        c[0].metric("Dung lượng CSV", f"{csv_mb / 1024:.2f} GB")
        c[1].metric("Dung lượng Parquet", f"{pq_mb / 1024:.2f} GB", f"-{100 * (1 - pq_mb / csv_mb):.0f}%", delta_color="inverse")
        c[2].metric("Nơi lưu trữ", S.get("luu_tru", "local").upper())
        st.subheader("Các bảng sau khi nạp bằng Spark")
        st.dataframe(st_, use_container_width=True, hide_index=True)
        long = st_.melt(id_vars="bang", value_vars=["csv_MB", "parquet_MB"], var_name="dinh_dang", value_name="MB")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y("bang:N", title=None), x=alt.X("MB:Q", title="Dung lượng (MB)"),
            color=alt.Color("dinh_dang:N", scale=alt.Scale(range=[GREY, BLUE]), title="Định dạng"),
            yOffset="dinh_dang:N", tooltip=["bang", "dinh_dang", "MB"]).properties(height=380),
            use_container_width=True)
    if D["time"] is not None:
        st.subheader("Thời gian đọc và tổng hợp 13,6 triệu dòng trả góp")
        st.altair_chart(bar(D["time"], "Cach_xu_ly", "Thoi_gian_s", "Thời gian (giây)"), use_container_width=True)

# ---------------- 2. Khám phá
with tab2:
    c1, c2 = st.columns(2)
    for col, key, label, cat in [(c1, "edu", "Tỷ lệ vỡ nợ theo trình độ học vấn", "NAME_EDUCATION_TYPE"),
                                 (c2, "inc", "Tỷ lệ vỡ nợ theo nguồn thu nhập", "NAME_INCOME_TYPE")]:
        df = D[key]
        if df is not None:
            df = df.assign(ti_le=100 * df.ti_le_vo_no)
            col.markdown(f"**{label}**")
            col.altair_chart(bar(df, cat, "ti_le", "Tỷ lệ vỡ nợ (%)"), use_container_width=True)
    c1, c2 = st.columns(2)
    if D["age"] is not None:
        c1.markdown("**Tỷ lệ vỡ nợ theo nhóm tuổi**")
        c1.altair_chart(bar(D["age"].assign(ti_le=100 * D["age"].ti_le_vo_no), "nhom_tuoi", "ti_le",
                            "Tỷ lệ vỡ nợ (%)", horizontal=False), use_container_width=True)
    if D["ext"] is not None:
        e = D["ext"].assign(diem=D["ext"].nhom / 10, ti_le=100 * D["ext"].ti_le_vo_no)
        c2.markdown("**Tỷ lệ vỡ nợ theo điểm tín dụng bên ngoài (EXT_SOURCE)**")
        c2.altair_chart(alt.Chart(e).mark_line(point=True, color=RED).encode(
            x=alt.X("diem:Q", title="Điểm EXT_SOURCE trung bình"), y=alt.Y("ti_le:Q", title="Tỷ lệ vỡ nợ (%)"),
            tooltip=["diem", alt.Tooltip("ti_le", format=".1f"), "so_ho_so"]).properties(height=300),
            use_container_width=True)
    if D["miss"] is not None:
        n = st.slider("Số cột có tỷ lệ thiếu cao nhất", 5, 40, 15)
        st.markdown("**Giá trị thiếu trong bảng hồ sơ**")
        st.altair_chart(bar(D["miss"].head(n), "cot", "ti_le_thieu_%", "Tỷ lệ thiếu (%)"), use_container_width=True)

# ---------------- 3. Mô hình
with tab3:
    st.subheader("Kết quả trên tập kiểm tra (ngưỡng 0,5)")
    st.dataframe(D["metrics"], use_container_width=True, hide_index=True)
    pred = D["pred"]
    if pred is not None:
        from sklearn.metrics import roc_curve
        rows = []
        for m in MODELS:
            fpr, tpr, _ = roc_curve(pred.TARGET, pred[m])
            idx = np.linspace(0, len(fpr) - 1, min(400, len(fpr))).astype(int)
            rows.append(pd.DataFrame({"FPR": fpr[idx], "TPR": tpr[idx], "Mô hình": m}))
        roc = pd.concat(rows)
        c1, c2 = st.columns(2)
        c1.markdown("**Đường ROC**")
        c1.altair_chart(alt.Chart(roc).mark_line().encode(
            x="FPR:Q", y="TPR:Q", color=alt.Color("Mô hình:N", scale=alt.Scale(range=[GREY, BLUE, RED]))
        ).properties(height=340), use_container_width=True)
        if D["fi"] is not None:
            c2.markdown("**20 đặc trưng quan trọng nhất**")
            c2.altair_chart(bar(D["fi"], "dac_trung", "do_quan_trong", "Độ quan trọng", fmt=".4f"),
                            use_container_width=True)

        st.subheader("Chọn ngưỡng duyệt vay")
        m = st.selectbox("Mô hình", MODELS, index=MODELS.index(BEST))
        thr = st.slider("Ngưỡng xác suất vỡ nợ để từ chối hồ sơ", 0.05, 0.95, 0.5, 0.05)
        y, yhat = pred.TARGET.values, (pred[m].values >= thr).astype(int)
        tp, fp = int(((y == 1) & (yhat == 1)).sum()), int(((y == 0) & (yhat == 1)).sum())
        fn, tn = int(((y == 1) & (yhat == 0)).sum()), int(((y == 0) & (yhat == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        c = st.columns(4)
        c[0].metric("Recall (bắt được người vỡ nợ)", f"{100 * rec:.1f}%")
        c[1].metric("Precision (cảnh báo đúng)", f"{100 * prec:.1f}%")
        c[2].metric("Hồ sơ bị từ chối", f"{100 * (tp + fp) / len(y):.1f}%")
        c[3].metric("Người vỡ nợ bị bỏ sót", f"{fn:,}")
        st.table(pd.DataFrame([[tp, fn], [fp, tn]], index=["Thực tế: vỡ nợ", "Thực tế: không"],
                              columns=["Dự đoán: vỡ nợ", "Dự đoán: không"]))

# ---------------- 4. Chấm điểm
with tab4:
    if pred is not None:
        p = pred[BEST]
        lv = p.apply(risk_level).value_counts().reindex(["Thấp", "Trung bình", "Cao"]).fillna(0).astype(int)
        st.subheader(f"Phân bố mức rủi ro trên tập kiểm tra — {BEST}")
        c = st.columns(3)
        for k, (name, v) in enumerate(lv.items()):
            c[k].metric(f"Rủi ro {name}", f"{v:,} hồ sơ", f"{100 * v / len(p):.1f}%", delta_color="off")

    st.subheader("Chấm điểm một hồ sơ bằng mô hình Spark đã lưu")
    st.caption("Ngưỡng: Thấp < 40% · Trung bình 40–70% · Cao ≥ 70% (ngưỡng minh họa)")

    @st.cache_resource(show_spinner="Đang khởi động Spark và nạp mô hình...")
    def spark_objects():
        from pyspark.sql import SparkSession
        from pyspark.ml import PipelineModel
        from pyspark.ml.classification import (GBTClassificationModel, LogisticRegressionModel,
                                               RandomForestClassificationModel)
        spark = (SparkSession.builder.appName("CreditRisk-Dashboard").master("local[*]")
                 .config("spark.driver.memory", "4g").config("spark.sql.shuffle.partitions", "8")
                 .config("spark.ui.port", "4050").getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
        loader = {"GBT": GBTClassificationModel, "Random Forest": RandomForestClassificationModel,
                  "Logistic Regression": LogisticRegressionModel}[BEST]
        prep = PipelineModel.load(S["duong_dan_tien_xu_ly"])
        model = loader.load(S["duong_dan_mo_hinh"])
        feats = spark.read.parquet(S["duong_dan_dac_trung"])
        return spark, prep, model, feats

    default_id = int(pred.sample(1, random_state=1).SK_ID_CURR.iloc[0]) if pred is not None else 100002
    c1, c2 = st.columns([1, 2])
    if "sk" not in st.session_state:
        st.session_state.sk = default_id
    if pred is not None and c1.button("Lấy ngẫu nhiên một hồ sơ trong tập kiểm tra"):
        st.session_state.sk = int(pred.sample(1).SK_ID_CURR.iloc[0])
    sk_id = c1.number_input("Mã hồ sơ SK_ID_CURR", step=1, key="sk")
    if c1.button("Chấm điểm", type="primary"):
        from pyspark.sql import functions as F
        from pyspark.ml.functions import vector_to_array
        spark, prep, model, feats = spark_objects()
        row = feats.filter(F.col("SK_ID_CURR") == int(sk_id))
        if row.count() == 0:
            st.warning("Không tìm thấy hồ sơ này trong bảng đặc trưng.")
        else:
            cat = [c for c, t in feats.dtypes if t == "string"]
            num = [c for c, t in feats.dtypes if t != "string" and c not in ("SK_ID_CURR", "TARGET")]
            row = row.select("SK_ID_CURR", F.col("TARGET").cast("double").alias("TARGET"),
                             *[F.col(c).cast("double").alias(c) for c in num], *cat).withColumn("weight", F.lit(1.0))
            out = model.transform(prep.transform(row))
            prob = float(out.select(vector_to_array("probability")[1]).first()[0])
            info = row.select("TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS", "CREDIT_INCOME_RATIO",
                              "EXT_SOURCE_MEAN", "BUR_COUNT", "PREV_REFUSED_RATIO", "INS_LATE_RATIO",
                              "INS_DPD_MAX").toPandas().T
            level = risk_level(prob)
            color = {"Thấp": BLUE, "Trung bình": "#B7791F", "Cao": RED}[level]
            c2.markdown(f"<div style='padding:18px;border-radius:8px;background:{color};color:white'>"
                        f"<div style='font-size:18px'>Xác suất vỡ nợ</div>"
                        f"<div style='font-size:44px;font-weight:700'>{100 * prob:.1f}%</div>"
                        f"<div style='font-size:20px'>Mức rủi ro: <b>{level}</b></div></div>", unsafe_allow_html=True)
            info.columns = ["Giá trị"]
            info.index = ["Thực tế (TARGET)", "Thu nhập", "Khoản vay", "Tuổi", "Khoản vay / thu nhập",
                          "Điểm EXT_SOURCE TB", "Số khoản vay ở tổ chức khác", "Tỷ lệ hồ sơ cũ bị từ chối",
                          "Tỷ lệ kỳ trả góp trễ", "Số ngày trễ tối đa"]
            c2.dataframe(info.round(3), use_container_width=True)
