"""
Halaman Prediksi — Forecasting interaktif dengan custom horizon dan model.
v3: window size auto-detect, best model dinamis.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import (
    load_models_and_scalers,
    load_evaluation_metrics,
    get_best_model_name,
    get_market_data,
    predict_n_days_iterative,
    format_idr,
)

st.set_page_config(page_title="Prediksi · Gold Forecast", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1419; }
    section[data-testid="stSidebar"] { background-color: #151a21; }
    h1, h2, h3 { color: #e8e6e0 !important; }
    div[data-testid="stMetric"] {
        background-color: #1a2028;
        padding: 16px;
        border-radius: 12px;
        border: 0.5px solid #2a3038;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔮 Prediksi Harga Emas")
st.markdown("*Forecasting interaktif dengan pilihan model dan horizon kustom*")
st.markdown("---")

metrics_df = load_evaluation_metrics()
best_model_name = get_best_model_name()

col_model, col_horizon, col_start = st.columns(3)
with col_model:
    if metrics_df is not None:
        sorted_models = metrics_df.sort_values("MAPE (%)")
        model_labels = []
        for _, row in sorted_models.iterrows():
            label = f"{row['Model']} ⭐ (best, MAPE {row['MAPE (%)']:.2f}%)" \
                    if row['Model'] == best_model_name \
                    else f"{row['Model']} (MAPE {row['MAPE (%)']:.2f}%)"
            model_labels.append(label)
    else:
        model_labels = [f"{best_model_name} ⭐ (best)", "GRU", "Bi-LSTM"]

    model_pilihan = st.selectbox(
        "Pilih Model", model_labels, index=0,
        help=f"{best_model_name} direkomendasikan karena akurasi terbaik pada test set",
    )

with col_horizon:
    n_days = st.slider(
        "Horizon Prediksi (hari)",
        min_value=1, max_value=60, value=7, step=1,
        help="Semakin jauh, semakin besar akumulasi error",
    )

with col_start:
    st.markdown("**Warning Tingkat**")
    if n_days <= 3:
        st.success("✅ Short-term — akurasi baik")
    elif n_days <= 14:
        st.warning("⚠️ Medium-term — error mulai terakumulasi")
    else:
        st.error("❌ Long-term — error signifikan")

with st.spinner("Loading data dan model..."):
    df, source_label = get_market_data()
    models, scaler, scaler_target, window_size = load_models_and_scalers()

if df is None:
    st.error("Data tidak tersedia.")
    st.stop()

active_key = model_pilihan.split(" ")[0]  # "LSTM ⭐..." → "LSTM"
if "Bi-LSTM" in model_pilihan:
    active_key = "Bi-LSTM"

active_model = models[active_key] if models is not None else None

if window_size is None:
    window_size = 1

if active_model is None:
    st.warning("Model belum ter-load. Hasil prediksi pakai mode demo.")

with st.spinner(f"Menjalankan {active_key} untuk {n_days} hari ke depan..."):
    forecast = predict_n_days_iterative(active_model, scaler, scaler_target, df, window_size, n_days)

future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=n_days)
harga_terakhir = df["Gold_IDR_gram"].iloc[-1]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Harga Terakhir Observed",
              format_idr(harga_terakhir),
              delta=df.index[-1].strftime("%d %b %Y"))
with col2:
    pred_end = forecast[-1]
    change_total = (pred_end - harga_terakhir) / harga_terakhir * 100
    st.metric(f"Prediksi Hari ke-{n_days}",
              format_idr(pred_end),
              delta=f"{change_total:+.2f}% dari terakhir")
with col3:
    pred_avg = forecast.mean()
    st.metric(f"Rata-rata Prediksi {n_days} hari",
              format_idr(pred_avg),
              delta=f"Range: {format_idr(forecast.min())}–{format_idr(forecast.max())}",
              delta_color="off")

st.markdown("---")
st.markdown("### 📈 Visualisasi Forecast")

fig = go.Figure()

hist_tail = df["Gold_IDR_gram"].iloc[-30:]
fig.add_trace(go.Scatter(
    x=hist_tail.index, y=hist_tail.values,
    name="Historis (30 hari terakhir)",
    line=dict(color="#888780", width=1.5),
))

fig.add_trace(go.Scatter(
    x=future_dates, y=forecast,
    name=f"Forecast {active_key}",
    mode="lines+markers",
    line=dict(color="#f5c441", width=2),
    marker=dict(size=7, color="#f5c441", line=dict(color="#1a2028", width=1)),
))

fig.add_trace(go.Scatter(
    x=[df.index[-1], future_dates[0]],
    y=[harga_terakhir, forecast[0]],
    mode="lines",
    line=dict(color="#f5c441", width=2, dash="dot"),
    showlegend=False,
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#1a2028",
    plot_bgcolor="#1a2028",
    height=420,
    margin=dict(l=40, r=40, t=20, b=40),
    xaxis=dict(gridcolor="#2a3038", title="Tanggal"),
    yaxis=dict(gridcolor="#2a3038", title="Harga Emas (IDR/gram)", tickformat=",.0f"),
    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 📋 Tabel Prediksi")

pred_df = pd.DataFrame({
    "Tanggal": future_dates.strftime("%A, %d %b %Y"),
    "Prediksi (IDR/gram)": [f"Rp {v:,.0f}" for v in forecast],
    "Change dari hari sebelumnya": [
        f"{((forecast[i] - (forecast[i-1] if i > 0 else harga_terakhir)) / (forecast[i-1] if i > 0 else harga_terakhir) * 100):+.2f}%"
        for i in range(n_days)
    ],
})
pred_df.index = range(1, len(pred_df) + 1)
pred_df.index.name = "Hari ke-"

st.dataframe(pred_df, use_container_width=True, height=min(400, 35 * (n_days + 1)))

csv_df = pd.DataFrame({
    "Hari_ke": range(1, n_days + 1),
    "Tanggal": future_dates.strftime("%Y-%m-%d"),
    "Prediksi_IDR_per_gram": forecast.round(2),
    "Model": active_key,
    "Window_Size": window_size,
    "Data_Source": source_label,
})
csv = csv_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download tabel prediksi (CSV)",
    data=csv,
    file_name=f"prediksi_emas_{active_key}_{n_days}hari_{df.index[-1].strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    f"💡 Window size: **{window_size} hari** (hasil hybrid PACF + empirical grid search). "
    f"Iterative forecasting: output hari ke-N dipakai sebagai input untuk prediksi hari ke-N+1. "
    f"Data source: {source_label}."
)
