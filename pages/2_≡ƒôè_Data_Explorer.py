"""
Halaman Data Explorer v3 — EDA dengan tambahan ADF Test dan Log Return.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_historical_data, get_market_data, ADF_RESULTS

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
    .stat-badge-yes {
        display: inline-block;
        background: #1d9e75;
        color: white;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 10px;
    }
    .stat-badge-no {
        display: inline-block;
        background: #c0392b;
        color: white;
        font-size: 12px;
        padding: 3px 10px;
        border-radius: 10px;
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🔗 Korelasi",
    "🧪 Uji Stasioneritas (ADF)",
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

    fig.add_trace(go.Scatter(x=df.index, y=df["Gold_IDR_gram"],
                              name="Emas", line=dict(color="#f5c441", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["USDIDR"],
                              name="USD/IDR", line=dict(color="#4a90e2", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["IHSG"],
                              name="IHSG", line=dict(color="#2ecc71", width=1)), row=3, col=1)

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

with tab2:
    st.markdown("### Matriks Korelasi Pearson")
    corr_matrix = df.corr(method="pearson")

    col_corr, col_info = st.columns([2, 1])
    with col_corr:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=["Harga Emas", "USD/IDR", "IHSG"],
            y=["Harga Emas", "USD/IDR", "IHSG"],
            colorscale=[[0, "#c0392b"], [0.5, "#2c3e50"], [1, "#27ae60"]],
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
        st.markdown(f"""
        - **Emas ↔ USD/IDR**: r = **{r_gold_usd:.4f}** (sangat kuat positif)
        - **Emas ↔ IHSG**: r = **{r_gold_ihsg:.4f}** (kuat positif)
        """)

with tab3:
    st.markdown("### Augmented Dickey-Fuller (ADF) Test")
    st.markdown("""
    **Hipotesis:**
    - H₀: Data **non-stasioner** (memiliki unit root)
    - H₁: Data **stasioner**

    Keputusan: tolak H₀ jika **p-value < 0.05** atau ADF statistic < critical value (5%).
    """)

    st.markdown("#### 1. Data Mentah (sebelum transformasi)")
    raw = ADF_RESULTS["raw"]

    cols_raw = st.columns(3)
    for col, (var_name, result) in zip(cols_raw, raw.items()):
        with col:
            badge = '<span class="stat-badge-yes">✅ STASIONER</span>' if result["stationary"] \
                    else '<span class="stat-badge-no">❌ NON-STASIONER</span>'
            st.markdown(f"""
            <div style="background:#1a2028;border:0.5px solid #2a3038;border-radius:10px;padding:14px;">
                <strong>{var_name}</strong><br>
                <span style="color:#888;font-size:13px;">ADF stat: <code>{result['adf_stat']:.4f}</code></span><br>
                <span style="color:#888;font-size:13px;">p-value: <code>{result['p_value']:.6f}</code></span><br><br>
                {badge}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### 2. Setelah Log Return Transformation")
    st.markdown("Log return: $r_t = \\ln(P_t / P_{t-1})$ — transformasi standar untuk stasionerisasi data finansial.")

    log_ret = ADF_RESULTS["log_return"]
    cols_log = st.columns(3)
    for col, (var_name, result) in zip(cols_log, log_ret.items()):
        with col:
            badge = '<span class="stat-badge-yes">✅ STASIONER</span>' if result["stationary"] \
                    else '<span class="stat-badge-no">❌ NON-STASIONER</span>'
            st.markdown(f"""
            <div style="background:#1a2028;border:0.5px solid #1d9e75;border-radius:10px;padding:14px;">
                <strong>{var_name}</strong><br>
                <span style="color:#888;font-size:13px;">ADF stat: <code>{result['adf_stat']:.4f}</code></span><br>
                <span style="color:#888;font-size:13px;">p-value: <code>{result['p_value']:.6f}</code></span><br><br>
                {badge}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### 3. Ringkasan & Implikasi")
    summary_df = pd.DataFrame({
        "Variabel": ["Harga Emas", "USD/IDR", "IHSG"],
        "Data Mentah": ["❌ Non-stasioner"] * 3,
        "Log Return": ["✅ Stasioner"] * 3,
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.info("""
    **Insight Metodologi:**
    - Data mentah ketiga variabel **non-stasioner** — memiliki tren panjang yang dominan
    - Setelah transformasi log return, semua variabel **stasioner** (p < 0.001)
    - Stasioneritas penting untuk validitas analisis ACF/PACF — analisis dilakukan pada **log return**
    - Critical Values: 1% = -3.4327, 5% = -2.8626, 10% = -2.5673
    """)

with tab4:
    st.markdown("### Autocorrelation Function (ACF) & Partial (PACF)")
    st.caption("Analisis ketergantungan harga emas pada lag waktu — informasi struktur temporal untuk justifikasi window size.")

    from statsmodels.tsa.stattools import acf, pacf

    series_choice = st.radio(
        "Analisis pada:",
        ["Log Return (stasioner)", "Data Mentah"],
        horizontal=True, index=0,
        help="Log return direkomendasikan karena stasioner."
    )

    if series_choice == "Log Return (stasioner)":
        target_series = np.log(df["Gold_IDR_gram"] / df["Gold_IDR_gram"].shift(1)).dropna()
    else:
        target_series = df["Gold_IDR_gram"].dropna()

    max_lag = st.slider("Jumlah Lag", min_value=10, max_value=60, value=40, step=5)

    acf_vals = acf(target_series, nlags=max_lag)
    pacf_vals = pacf(target_series, nlags=max_lag)

    n = len(target_series)
    ci = 1.96 / np.sqrt(n)

    fig_acf = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"ACF — {series_choice}", f"PACF — {series_choice}"),
        vertical_spacing=0.15,
    )

    for i, val in enumerate(acf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[i, i], y=[0, val],
            mode="lines", line=dict(color="#4a90e2", width=2), showlegend=False,
        ), row=1, col=1)
    fig_acf.add_hline(y=ci, line=dict(color="#e74c3c", dash="dash"), row=1, col=1)
    fig_acf.add_hline(y=-ci, line=dict(color="#e74c3c", dash="dash"), row=1, col=1)
    fig_acf.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=1)

    for i, val in enumerate(pacf_vals):
        fig_acf.add_trace(go.Scatter(
            x=[i, i], y=[0, val],
            mode="lines", line=dict(color="#2ecc71", width=2), showlegend=False,
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

    if series_choice == "Log Return (stasioner)":
        st.success(f"""
        **Insight — Justifikasi Window Size:**
        - PACF pada log return menunjukkan **spike signifikan hanya di lag-1**, lag berikutnya berada dalam confidence interval
        - Ini mengindikasikan **karakteristik AR(1)** kuat pada return harga emas
        - Berdasarkan rekomendasi Workneh & Jha (2025), kandidat window diambil dari lag PACF yang signifikan
        - Validasi empiris (mini grid search) memilih **window=1** sebagai optimal
        - Konsisten dengan teori: setelah memperhitungkan lag-1, kontribusi lag lain tidak signifikan
        """)
    else:
        st.info(f"""
        ACF data mentah turun lambat → indikasi **non-stasioneritas** (sudah dikonfirmasi via ADF test).
        Untuk identifikasi struktur temporal yang valid, gunakan analisis pada log return.
        """)

with tab5:
    st.markdown("### Statistik Deskriptif")
    desc = df.describe().round(2)
    desc.index = ["Jumlah", "Mean", "Std Dev", "Min", "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
    desc.columns = ["Harga Emas (IDR/gram)", "USD/IDR", "IHSG"]
    st.dataframe(desc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Distribusi Harga Emas")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df["Gold_IDR_gram"], nbinsx=50,
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
