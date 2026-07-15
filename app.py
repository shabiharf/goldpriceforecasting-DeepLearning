"""
Gold Price Forecasting Dashboard — Halaman Utama (v3.1)
Tugas Akhir: Shabiha Rahma Fauziah (1206220017)
Program Studi Sains Data, Telkom University Surabaya

Update v3.1:
- Best model auto-detect dari evaluation_metrics.csv (sekarang: Bi-LSTM, MAPE 1.54%)
- Window size auto-detect dari model input shape (window=1)
- Badge "Updated Daily" untuk data via GitHub Actions (bukan lagi "Offline")
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from utils import (
    load_models_and_scalers,
    load_test_predictions,
    load_evaluation_metrics,
    get_best_model_name,
    get_market_data,
    predict_next_day,
    predict_n_days_iterative,
    format_idr,
)

st.set_page_config(
    page_title="Gold Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1419; }
    section[data-testid="stSidebar"] { background-color: #151a21; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 500; }
    div[data-testid="stMetric"] {
        background-color: #1a2028;
        padding: 16px;
        border-radius: 12px;
        border: 0.5px solid #2a3038;
    }
    h1, h2, h3 { color: #e8e6e0 !important; }
    .live-badge {
        display: inline-block;
        background: #1d9e75;
        color: white;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
    }
    .offline-badge {
        display: inline-block;
        background: #2c7a9e;
        color: white;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 500;
    }
    div[data-testid="stExpander"] {
        background-color: #1a2028;
        border: 0.5px solid #7a6418;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    div[data-testid="stExpander"] summary {
        color: #f5c441 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

best_model_name = get_best_model_name()

with st.sidebar:
    st.markdown("## 📈 Gold Forecast")
    st.markdown("---")
    st.markdown("### Filter")

    model_options = []
    metrics_for_dropdown = load_evaluation_metrics()
    if metrics_for_dropdown is not None:
        sorted_models = metrics_for_dropdown.sort_values("MAPE (%)")["Model"].tolist()
        for m in sorted_models:
            label = f"{m} ⭐ (best)" if m == best_model_name else m
            model_options.append(label)
    else:
        model_options = [f"{best_model_name} ⭐ (best)", "GRU", "LSTM"]

    model_pilihan = st.selectbox("Model", model_options, index=0)
    horizon = st.selectbox("Horizon prediksi", ["1 hari", "7 hari", "30 hari"], index=1)

with st.spinner("Memuat data pasar..."):
    df, source_label = get_market_data()

if df is None:
    st.error("❌ Tidak ada data tersedia. Pastikan `data/historical_data.csv` ada, "
             "atau koneksi internet aktif untuk Yahoo Finance.")
    st.stop()

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# Dashboard Peramalan Harga Emas")
    st.markdown(f"*Data terakhir: {df.index[-1].strftime('%d %b %Y')} · Source: {source_label}*")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if "live" in source_label.lower():
        st.markdown('<span class="live-badge">● Live</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="offline-badge">● Updated Daily</span>', unsafe_allow_html=True)

with st.expander("⚠️ Catatan Penting tentang Akurasi Prediksi Live", expanded=False):
    st.markdown(f"""
    Model pada dashboard ini dilatih menggunakan data historis **Januari 2015 – Desember 2025**
    dengan metodologi **hybrid window selection** (PACF + validasi empiris).
    Hasil mini grid search menentukan **window size optimal = 1 hari**, sehingga model
    melakukan prediksi t+1 berdasarkan harga t.

    Metrik evaluasi terbaik (test set 2023–2025):
    - **{best_model_name}** — model dengan MAPE terendah di antara 3 arsitektur

    Data historis di-refresh otomatis setiap hari via GitHub Actions. Jika Yahoo Finance
    tidak dapat diakses langsung dari server dashboard, sistem menggunakan data
    hasil refresh terakhir sebagai fallback.

    Untuk prediksi setelah Desember 2025:
    - Prediksi bersifat **ekstrapolasi** dari distribusi data training
    - Akurasi aktual dapat berbeda dari MAPE yang dilaporkan, tergantung seberapa jauh
      kondisi pasar menyimpang dari distribusi training (*out-of-distribution*)
    - Prediksi multi-hari (>1 hari) menggunakan *iterative forecasting* — error
      akan terakumulasi semakin jauh horizon

    Dashboard ini disajikan sebagai **proof-of-concept** dari hasil penelitian.
    """)

models, scaler, scaler_target, window_size = load_models_and_scalers()
metrics_df = load_evaluation_metrics()
test_preds = load_test_predictions()

if models is None:
    st.info("💡 **Mode Demo**: File model belum ter-load. "
            "Pastikan folder `models/` berisi 3 file `.h5` + 2 file `.pkl`.")
    window_size = 1

active_model_key = model_pilihan.replace(" ⭐ (best)", "").strip()
active_model = models[active_model_key] if models is not None else None

harga_terakhir = df["Gold_IDR_gram"].iloc[-1]
harga_kemarin = df["Gold_IDR_gram"].iloc[-2]
change_today = (harga_terakhir - harga_kemarin) / harga_kemarin * 100

pred_besok = predict_next_day(active_model, scaler, scaler_target, df, window_size)
change_pred = (pred_besok - harga_terakhir) / harga_terakhir * 100

usd_idr = df["USDIDR"].iloc[-1]
usd_idr_prev = df["USDIDR"].iloc[-2]
usd_change = (usd_idr - usd_idr_prev) / usd_idr_prev * 100

if metrics_df is not None:
    active_mape = metrics_df[metrics_df["Model"] == active_model_key]["MAPE (%)"].values
    mape_value = active_mape[0] if len(active_mape) > 0 else 1.54
else:
    mape_value = 1.54

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Harga Emas Terakhir", format_idr(harga_terakhir),
              delta=f"{change_today:+.2f}% dari sebelumnya")
with col2:
    st.metric(f"Prediksi Berikutnya ({active_model_key})", format_idr(pred_besok),
              delta=f"{change_pred:+.2f}% estimasi")
with col3:
    st.metric(f"MAPE Model {active_model_key}", f"{mape_value:.2f}%",
              delta="Best model" if active_model_key == best_model_name else "",
              delta_color="off")
with col4:
    st.metric("USD/IDR", f"{usd_idr:,.0f}", delta=f"{usd_change:+.2f}%", delta_color="inverse")

st.markdown("---")
st.markdown("### 📉 Aktual vs Prediksi — Test Set")

fig = go.Figure()
if test_preds is not None:
    fig.add_trace(go.Scatter(x=test_preds["Date"], y=test_preds["Actual"],
                              name="Aktual", line=dict(color="#f5c441", width=2)))
    fig.add_trace(go.Scatter(x=test_preds["Date"], y=test_preds["LSTM"],
                              name="LSTM", line=dict(color="#378add", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=test_preds["Date"], y=test_preds["GRU"],
                              name="GRU", line=dict(color="#1d9e75", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=test_preds["Date"], y=test_preds["BiLSTM"],
                              name="Bi-LSTM", line=dict(color="#e24b4a", width=1.5, dash="dash")))
else:
    st.warning("File `test_predictions.csv` tidak ditemukan di folder `data/`.")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1a2028",
    plot_bgcolor="#1a2028",
    height=380,
    margin=dict(l=40, r=40, t=20, b=40),
    xaxis=dict(gridcolor="#2a3038", title="Tanggal"),
    yaxis=dict(gridcolor="#2a3038", title="Harga (IDR/gram)", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🏆 Perbandingan Model")
    if metrics_df is not None:
        metrics_sorted = metrics_df.sort_values("MAPE (%)")
        models_list = metrics_sorted["Model"].tolist()
        mape_list = metrics_sorted["MAPE (%)"].tolist()
        colors_map = {"LSTM": "#378add", "GRU": "#1d9e75", "Bi-LSTM": "#e24b4a"}
        colors_list = [colors_map.get(m, "#888780") for m in models_list]
    else:
        models_list = ["Bi-LSTM", "LSTM", "GRU"]
        mape_list = [1.54, 2.36, 2.45]
        colors_list = ["#e24b4a", "#378add", "#1d9e75"]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        y=models_list, x=mape_list, orientation="h",
        marker=dict(color=colors_list),
        text=[f"{v:.2f}%" for v in mape_list], textposition="outside",
    ))
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=280,
        margin=dict(l=20, r=40, t=20, b=40),
        xaxis=dict(title="MAPE (%)", gridcolor="#2a3038", range=[0, max(mape_list) * 1.3]),
        yaxis=dict(gridcolor="#2a3038"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    n_horizon = {"1 hari": 1, "7 hari": 7, "30 hari": 30}.get(horizon, 7)
    st.markdown(f"### 🔮 Prediksi {n_horizon} Hari ke Depan ({active_model_key})")

    forecast = predict_n_days_iterative(active_model, scaler, scaler_target, df, window_size, n_horizon)
    future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=n_horizon)

    fig_forecast = go.Figure()
    hist_tail = df["Gold_IDR_gram"].iloc[-14:]
    fig_forecast.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        name="Historis", line=dict(color="#888780", width=1.5),
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_dates, y=forecast,
        name="Forecast", mode="lines+markers",
        line=dict(color="#f5c441", width=2),
        marker=dict(size=8, color="#f5c441", line=dict(color="#1a2028", width=1)),
    ))
    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=280,
        margin=dict(l=20, r=40, t=20, b=40),
        xaxis=dict(gridcolor="#2a3038"),
        yaxis=dict(gridcolor="#2a3038", tickformat=",.0f"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.caption(f"💡 Semakin jauh horizon, semakin buram akurasi (iterative forecasting). Window size: {window_size} hari.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888780; font-size: 13px;'>"
    "Tugas Akhir Dashboard · Shabiha Rahma Fauziah (1206220017) · "
    "Sains Data — Telkom University Surabaya"
    "</div>",
    unsafe_allow_html=True,
)
