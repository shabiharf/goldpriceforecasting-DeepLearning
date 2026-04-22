"""
Halaman Data Explorer — Exploratory Data Analysis
Time series, correlation, autocorrelation, statistik deskriptif.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_historical_data, get_market_data

st.set_page_config(page_title="Data Explorer · Gold Forecast", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f1419; }
    section[data-testid="stSidebar"] { background-color: #151a21; }
    h1, h2, h3 { color: #e8e6e0 !important; }
    div[data-testid="stMetric"] {
        background-color: #1a2028;
        padding: 12px;
        border-radius: 10px;
        border: 0.5px solid #2a3038;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Data Explorer")
st.markdown("*Analisis eksplorasi data: harga emas Indonesia, USD/IDR, IHSG*")
st.markdown("---")

with st.spinner("Loading data..."):
    df_live, source_label = get_market_data()
    df_historical = load_historical_data()

data_source = st.radio(
    "Pilih rentang data:",
    ["Training data (2015-2025)", "Live data (2015-sekarang)"],
    horizontal=True,
    index=0,
    help="Training = data yang dipakai model. Live = termasuk data terbaru dari Yahoo Finance."
)

if data_source == "Training data (2015-2025)" and df_historical is not None:
    df = df_historical
    source_note = f"📁 Data dari `data/historical_data.csv` — {len(df)} baris"
else:
    df = df_live
    source_note = f"🔴 Data live dari {source_label} — {len(df)} baris"

st.caption(source_note)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Jumlah Hari", f"{len(df):,}")
with col2:
    st.metric("Periode Awal", df.index[0].strftime("%d %b %Y"))
with col3:
    st.metric("Periode Akhir", df.index[-1].strftime("%d %b %Y"))
with col4:
    days_span = (df.index[-1] - df.index[0]).days
    st.metric("Rentang Hari", f"{days_span:,} hari")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time Series",
    "🔗 Korelasi",
    "📉 Autokorelasi (ACF/PACF)",
    "📋 Statistik Deskriptif",
])

with tab1:
    st.markdown("### Time Series Ketiga Variabel")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Harga Emas Indonesia (IDR/gram)",
            "Nilai Tukar USD/IDR",
            "IHSG (Indeks Harga Saham Gabungan)",
        ),
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Gold_IDR_gram"],
        name="Emas", line=dict(color="#f5c441", width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["USDIDR"],
        name="USD/IDR", line=dict(color="#4a90e2", width=1),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["IHSG"],
        name="IHSG", line=dict(color="#2ecc71", width=1),
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=700,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    for r in range(1, 4):
        fig.update_xaxes(gridcolor="#2a3038", row=r, col=1)
        fig.update_yaxes(gridcolor="#2a3038", row=r, col=1, tickformat=",.0f")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Observasi:**
    - **Emas** menunjukkan tren naik kuat mulai 2020, mencapai puncak tertinggi di akhir periode
    - **USD/IDR** fluktuatif dengan kecenderungan depresiasi rupiah
    - **IHSG** menunjukkan volatility tinggi dengan drop signifikan di 2020 (pandemi)
    """)

with tab2:
    st.markdown("### Matriks Korelasi Pearson")

    corr_matrix = df.corr(method="pearson")

    col_corr, col_info = st.columns([2, 1])

    with col_corr:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=["Harga Emas", "USD/IDR", "IHSG"],
            y=["Harga Emas", "USD/IDR", "IHSG"],
            colorscale=[
                [0, "#c0392b"],
                [0.5, "#2c3e50"],
                [1, "#27ae60"],
            ],
            zmin=-1, zmax=1,
            text=corr_matrix.round(4).values,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            showscale=True,
        ))
        fig_heatmap.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a2028",
            plot_bgcolor="#1a2028",
            height=400,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_info:
        st.markdown("**Interpretasi:**")
        r_gold_usd = corr_matrix.loc["Gold_IDR_gram", "USDIDR"]
        r_gold_ihsg = corr_matrix.loc["Gold_IDR_gram", "IHSG"]
        r_usd_ihsg = corr_matrix.loc["USDIDR", "IHSG"]

        st.markdown(f"""
        - **Emas ↔ USD/IDR**: r = **{r_gold_usd:.4f}** (sangat kuat positif)
          Ketika rupiah melemah, harga emas IDR meningkat.

        - **Emas ↔ IHSG**: r = **{r_gold_ihsg:.4f}** (kuat positif)
          Emas dan IHSG cenderung bergerak searah dalam jangka panjang.

        - **USD/IDR ↔ IHSG**: r = **{r_usd_ihsg:.4f}** (kuat positif)
          Weakening rupiah beriringan dengan penguatan IHSG (inflation effect).
        """)

    st.markdown("---")
    st.markdown("### Scatter Plot")
    scatter_choice = st.selectbox(
        "Plot hubungan:",
        ["Emas vs USD/IDR", "Emas vs IHSG", "USD/IDR vs IHSG"]
    )

    pairs = {
        "Emas vs USD/IDR": ("Gold_IDR_gram", "USDIDR", "Harga Emas (IDR/gram)", "USD/IDR"),
        "Emas vs IHSG": ("Gold_IDR_gram", "IHSG", "Harga Emas (IDR/gram)", "IHSG"),
        "USD/IDR vs IHSG": ("USDIDR", "IHSG", "USD/IDR", "IHSG"),
    }
    xcol, ycol, xlabel, ylabel = pairs[scatter_choice]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df[xcol], y=df[ycol],
        mode="markers",
        marker=dict(size=4, color=df.index.year, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Tahun")),
        name=scatter_choice,
    ))
    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=400,
        xaxis=dict(title=xlabel, gridcolor="#2a3038"),
        yaxis=dict(title=ylabel, gridcolor="#2a3038"),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("### Autocorrelation Function (ACF) & Partial (PACF)")
    st.caption("Analisis ketergantungan harga emas pada lag waktu — informasi struktur temporal untuk justifikasi window size.")

    from statsmodels.tsa.stattools import acf, pacf

    max_lag = st.slider("Jumlah Lag", min_value=20, max_value=100, value=60, step=10)

    gold_series = df["Gold_IDR_gram"].dropna()

    acf_vals = acf(gold_series, nlags=max_lag)
    pacf_vals = pacf(gold_series, nlags=max_lag)

    n = len(gold_series)
    ci = 1.96 / np.sqrt(n)

    fig_acf = make_subplots(
        rows=2, cols=1,
        subplot_titles=("ACF — Autocorrelation Function", "PACF — Partial Autocorrelation Function"),
        vertical_spacing=0.15,
    )

    for i, val in enumerate(acf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[i, i], y=[0, val],
            mode="lines", line=dict(color="#4a90e2", width=2),
            showlegend=False,
        ), row=1, col=1)
    fig_acf.add_hline(y=ci, line=dict(color="#e74c3c", dash="dash"), row=1, col=1)
    fig_acf.add_hline(y=-ci, line=dict(color="#e74c3c", dash="dash"), row=1, col=1)
    fig_acf.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=1)

    for i, val in enumerate(pacf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[i, i], y=[0, val],
            mode="lines", line=dict(color="#2ecc71", width=2),
            showlegend=False,
        ), row=2, col=1)
    fig_acf.add_hline(y=ci, line=dict(color="#e74c3c", dash="dash"), row=2, col=1)
    fig_acf.add_hline(y=-ci, line=dict(color="#e74c3c", dash="dash"), row=2, col=1)
    fig_acf.add_hline(y=0, line=dict(color="white", width=0.5), row=2, col=1)

    fig_acf.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig_acf.update_xaxes(gridcolor="#2a3038", title="Lag (hari)")
    fig_acf.update_yaxes(gridcolor="#2a3038")

    st.plotly_chart(fig_acf, use_container_width=True)

    st.markdown(f"""
    **Insight:**
    - **ACF** menunjukkan korelasi tinggi dan turun lambat → data **non-stasioner** (trend kuat)
    - **PACF** memiliki spike tajam di lag-1 → menunjukkan karakteristik **AR(1)** yang kuat
    - Nilai ACF pada lag 1: **{acf_vals[1]:.4f}**, lag 7: **{acf_vals[7]:.4f}**, lag 30: **{acf_vals[30]:.4f}**, lag 60: **{acf_vals[60]:.4f}**
    - Window size 60 hari yang dipakai model mencakup informasi temporal yang masih signifikan
    """)

with tab4:
    st.markdown("### Statistik Deskriptif")

    desc = df.describe().round(2)
    desc.index = ["Jumlah", "Mean", "Std Dev", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
    desc.columns = ["Harga Emas (IDR/gram)", "USD/IDR", "IHSG"]

    st.dataframe(desc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Distribusi Harga Emas")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["Gold_IDR_gram"],
        nbinsx=50,
        marker=dict(color="#f5c441", line=dict(color="#1a2028", width=1)),
    ))
    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1a2028",
        plot_bgcolor="#1a2028",
        height=350,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(title="Harga Emas (IDR/gram)", gridcolor="#2a3038", tickformat=",.0f"),
        yaxis=dict(title="Frekuensi", gridcolor="#2a3038"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Mean", f"Rp {df['Gold_IDR_gram'].mean():,.0f}")
    with col_b:
        skew = df["Gold_IDR_gram"].skew()
        st.metric("Skewness", f"{skew:.3f}",
                  "→ skewed kanan" if skew > 0 else "→ skewed kiri")
    with col_c:
        kurt = df["Gold_IDR_gram"].kurtosis()
        st.metric("Kurtosis", f"{kurt:.3f}",
                  "→ ekor tebal" if kurt > 0 else "→ ekor tipis",
                  delta_color="off")
