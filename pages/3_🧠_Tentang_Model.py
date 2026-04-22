"""
Halaman Tentang Model — Dokumentasi arsitektur, metrik, dan metodologi penelitian.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_evaluation_metrics

st.set_page_config(page_title="Tentang Model · Gold Forecast", page_icon="🧠", layout="wide")

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
    .model-card {
        background-color: #1a2028;
        border: 0.5px solid #2a3038;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .winner-card {
        border: 1.5px solid #f5c441;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧠 Tentang Model")
st.markdown("*Arsitektur, metodologi, dan hasil evaluasi dari penelitian thesis.*")
st.markdown("---")

st.markdown("## 📖 Deskripsi Penelitian")
st.markdown("""
Penelitian ini membandingkan tiga model deep learning untuk peramalan harga emas
Indonesia dalam satuan IDR/gram, dengan menggunakan data multivariat yang mencakup:

- **Harga emas global** (USD/troy ounce dari Yahoo Finance, ticker `GC=F`)
- **Nilai tukar USD/IDR** (ticker `IDR=X`)
- **IHSG** (Indeks Harga Saham Gabungan, ticker `^JKSE`)

**Konversi target:** Harga emas global dikonversi ke IDR/gram dengan formula:
""")

st.latex(r"\text{Harga}_\text{IDR/gram} = \frac{\text{Harga}_\text{USD/oz}}{31.1035} \times \text{Kurs USD/IDR}")

st.markdown("---")
st.markdown("## 🏆 Hasil Evaluasi Model")

metrics_df = load_evaluation_metrics()

if metrics_df is not None:
    col1, col2, col3 = st.columns(3)
    best_row = metrics_df.loc[metrics_df["MAPE (%)"].idxmin()]

    for idx, (col, row_name) in enumerate(zip([col1, col2, col3], ["GRU", "LSTM", "Bi-LSTM"])):
        row = metrics_df[metrics_df["Model"] == row_name].iloc[0]
        is_winner = row["Model"] == best_row["Model"]
        badge = "⭐ TERBAIK" if is_winner else ""
        with col:
            st.markdown(f"""
            <div class="model-card {'winner-card' if is_winner else ''}">
                <h3 style="margin: 0 0 4px;">{row['Model']} {badge}</h3>
                <p style="color: #888; margin: 0 0 12px; font-size: 13px;">Deep Learning Model</p>
                <p><strong>MAPE:</strong> {row['MAPE (%)']:.2f}%</p>
                <p><strong>RMSE:</strong> {row['RMSE']:,.0f}</p>
                <p><strong>MAE:</strong> {row['MAE']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Tabel Lengkap")
    display_metrics = metrics_df.copy()
    display_metrics["RMSE"] = display_metrics["RMSE"].apply(lambda x: f"{x:,.2f}")
    display_metrics["MAE"] = display_metrics["MAE"].apply(lambda x: f"{x:,.2f}")
    display_metrics["MAPE (%)"] = display_metrics["MAPE (%)"].apply(lambda x: f"{x:.4f}%")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    st.success(f"🏆 **Model terbaik: {best_row['Model']}** dengan MAPE {best_row['MAPE (%)']:.2f}%")
else:
    st.warning("File `evaluation_metrics.csv` tidak ditemukan.")

st.markdown("---")
st.markdown("## 🏗️ Arsitektur Model")

arch_tab1, arch_tab2, arch_tab3 = st.tabs(["GRU (Best)", "LSTM", "Bi-LSTM"])

with arch_tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
        **GRU (Gated Recurrent Unit)**

        Model terbaik dengan MAPE **2.41%**. GRU adalah varian RNN yang menggunakan
        dua gate (update & reset) untuk menangani dependensi jangka panjang dengan
        parameter lebih sedikit dibanding LSTM.

        **Hyperparameter (hasil Grid Search):**
        - Units: **128**
        - Batch size: **64**
        - Learning rate: **0.005**
        - Dropout: **0.2**
        - Optimizer: **Adam**
        - Loss: **MSE**
        """)
    with col_b:
        st.code("""
Sequential([
    Input(shape=(60, 3)),
    GRU(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# Total params: 59,393 (232 KB)
        """, language="python")

with arch_tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
        **LSTM (Long Short-Term Memory)**

        Model klasik dengan MAPE **6.98%**. LSTM menggunakan tiga gate
        (input, forget, output) untuk mengontrol aliran informasi.

        **Hyperparameter (hasil Grid Search):**
        - Units: **128**
        - Batch size: **32**
        - Learning rate: **0.005**
        - Dropout: **0.2**
        """)
    with col_b:
        st.code("""
Sequential([
    Input(shape=(60, 3)),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# Total params: 75,905 (297 KB)
        """, language="python")

with arch_tab3:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("""
        **Bi-LSTM (Bidirectional LSTM)**

        Model dengan MAPE **4.59%**. Bi-LSTM memproses data dari dua arah
        (forward dan backward), menangkap konteks temporal yang lebih kaya.

        **Hyperparameter (hasil Grid Search):**
        - Units: **128** (per arah)
        - Batch size: **32**
        - Learning rate: **0.001**
        - Dropout: **0.2**
        """)
    with col_b:
        st.code("""
Sequential([
    Input(shape=(60, 3)),
    Bidirectional(LSTM(128)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# Total params: 43,137 (169 KB)
        """, language="python")

st.markdown("---")
st.markdown("## 🔬 Metodologi")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    ### Preprocessing
    - **Rentang data:** 2 Jan 2015 – 30 Des 2025
    - **Total observasi:** 2,790 hari trading
    - **Missing value:** Forward-fill (max 1 hari), lalu drop
    - **Normalisasi:** Min-Max scaling [0, 1]

    ### Sliding Window
    - **Window size:** 60 hari
    - **Total sequences:** 2,730
    - **Input shape:** (60, 3) — 60 hari × 3 fitur

    ### Train/Test Split
    - **Rasio:** 80:20 (kronologis, tanpa shuffling)
    - **Training:** 2,184 sampel
    - **Test:** 546 sampel (23 Okt 2023 – 30 Des 2025)
    """)

with col_m2:
    st.markdown("""
    ### Hyperparameter Tuning
    - **Metode:** Grid Search
    - **Kombinasi:** 8 per model
    - **Parameters tested:**
        - `units`: [64, 128]
        - `batch_size`: [32, 64]
        - `learning_rate`: [0.001, 0.005]
        - `dropout`: [0.2]
    - **Early stopping:** patience 20 epoch
    - **Max epoch:** 200 (final training)

    ### Evaluasi
    - **RMSE** (Root Mean Squared Error)
    - **MAE** (Mean Absolute Error)
    - **MAPE** (Mean Absolute Percentage Error) ← primary metric
    """)

st.markdown("---")
st.markdown("## ⚠️ Keterbatasan & Future Work")

with st.expander("Lihat keterbatasan penelitian", expanded=True):
    st.markdown("""
    1. **Rentang data terbatas** — Model dilatih pada data 2015-2025. Untuk periode
       setelahnya, prediksi bersifat ekstrapolasi dan dapat mengalami *distribution shift*.

    2. **Iterative forecasting untuk multi-hari** — Prediksi lebih dari 1 hari ke depan
       dihitung secara rekursif, menyebabkan error terakumulasi seiring horizon.

    3. **Fitur eksternal terbatas** — Hanya menggunakan 3 fitur (emas USD, kurs, IHSG).
       Fitur lain seperti VIX, oil price, geopolitical index belum dipertimbangkan.

    4. **Single model per prediksi** — Tidak menggunakan ensemble atau model stacking.

    ### Rekomendasi Future Work
    - Implementasi retraining berkala (mis. setiap 6 bulan) dengan data terbaru
    - Pengembangan multi-output model untuk prediksi multi-step langsung
    - Penambahan fitur makroekonomi dan sentimen pasar
    - Ensemble method: kombinasi GRU + LSTM + Bi-LSTM
    """)

st.markdown("---")
st.markdown("## 👤 Tentang Peneliti")

col_p1, col_p2 = st.columns([2, 3])
with col_p1:
    st.markdown("""
    **Shabiha Rahma Fauziah**
    NIM: 1206220017

    Program Studi Sains Data
    Fakultas Informatika
    Universitas Telkom
    """)

with col_p2:
    st.markdown("""
    **Judul Thesis:**
    *Peramalan Harga Emas Indonesia Menggunakan LSTM, GRU, dan Bi-LSTM*

    **Stack:**
    Python · TensorFlow · Keras · Streamlit · Plotly · Yahoo Finance API
    """)
