"""
Halaman Tentang Model v3 — Dokumentasi arsitektur, hybrid window approach, ADF test.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from utils import load_evaluation_metrics, get_best_model_name, load_models_and_scalers

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
Indonesia dalam satuan IDR/gram, dengan menggunakan data multivariat:

- **Harga emas global** (USD/troy ounce dari Yahoo Finance, ticker `GC=F`)
- **Nilai tukar USD/IDR** (ticker `IDR=X`)
- **IHSG** (Indeks Harga Saham Gabungan, ticker `^JKSE`)

**Konversi target:**
""")
st.latex(r"\text{Harga}_\text{IDR/gram} = \frac{\text{Harga}_\text{USD/oz}}{31.1035} \times \text{Kurs USD/IDR}")

st.markdown("---")
st.markdown("## 🏆 Hasil Evaluasi Model")

metrics_df = load_evaluation_metrics()
best_model_name = get_best_model_name()

if metrics_df is not None:
    sorted_metrics = metrics_df.sort_values("MAPE (%)").reset_index(drop=True)
    cols = st.columns(3)

    for col, (_, row) in zip(cols, sorted_metrics.iterrows()):
        is_winner = row["Model"] == best_model_name
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
    display = sorted_metrics.copy()
    display["RMSE"] = display["RMSE"].apply(lambda x: f"{x:,.2f}")
    display["MAE"] = display["MAE"].apply(lambda x: f"{x:,.2f}")
    display["MAPE (%)"] = display["MAPE (%)"].apply(lambda x: f"{x:.4f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)

    best_row = sorted_metrics.iloc[0]
    st.success(f"🏆 **Model terbaik: {best_row['Model']}** dengan MAPE {best_row['MAPE (%)']:.2f}%")
else:
    st.warning("File `evaluation_metrics.csv` tidak ditemukan.")

st.markdown("---")
st.markdown("## 🏗️ Arsitektur Model")

_, _, _, window_size = load_models_and_scalers()
if window_size is None:
    window_size = 1

st.info(f"📌 **Window size: {window_size} hari** — ditentukan via hybrid approach (PACF + validasi empiris)")

arch_tab1, arch_tab2, arch_tab3 = st.tabs(["LSTM", "Bi-LSTM", "GRU"])

with arch_tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        **LSTM (Long Short-Term Memory)** {"⭐ BEST" if best_model_name == "LSTM" else ""}

        LSTM dengan tiga gate (input, forget, output) untuk mengontrol aliran informasi.
        Pada v3 dengan window=1, LSTM bertindak mirip dengan AR(1) non-linear yang
        memanfaatkan informasi multivariat (gold USD, USD/IDR, IHSG) hari kemarin
        untuk prediksi hari ini.

        **Hyperparameter:**
        - Units: 128
        - Dropout: 0.2
        - Optimizer: Adam
        - Loss: MSE
        """)
    with col_b:
        st.code(f"""
Sequential([
    Input(shape=({window_size}, 3)),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
        """, language="python")

with arch_tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        **Bi-LSTM (Bidirectional LSTM)** {"⭐ BEST" if best_model_name == "Bi-LSTM" else ""}

        Bi-LSTM memproses data dari dua arah (forward dan backward). Pada window pendek,
        kekuatan bidirectional kurang ter-eksploitasi tetapi tetap memberikan
        representasi feature yang lebih kaya.

        **Hyperparameter:**
        - Units: 128 (per arah)
        - Dropout: 0.2
        """)
    with col_b:
        st.code(f"""
Sequential([
    Input(shape=({window_size}, 3)),
    Bidirectional(LSTM(128)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
        """, language="python")

with arch_tab3:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"""
        **GRU (Gated Recurrent Unit)** {"⭐ BEST" if best_model_name == "GRU" else ""}

        GRU varian RNN dengan dua gate (update & reset), parameter lebih sedikit
        dibanding LSTM. Pada eksperimen v3, GRU terbukti underperform dibanding LSTM
        di window pendek — kemungkinan karena gate structure-nya kurang ekspresif
        untuk task ini.

        **Hyperparameter:**
        - Units: 128
        - Dropout: 0.2
        """)
    with col_b:
        st.code(f"""
Sequential([
    Input(shape=({window_size}, 3)),
    GRU(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])
        """, language="python")

st.markdown("---")
st.markdown("## 🔬 Metodologi")

st.markdown("### Hybrid Window Selection (PACF + Empirical)")
st.markdown("""
Window size tidak ditetapkan secara arbitrer atau fixed. Penelitian ini mengadopsi
**hybrid approach** berdasarkan rekomendasi Workneh & Jha (2025) dan Leites et al. (2024):

1. **Uji Stasioneritas (ADF Test)** pada data mentah dan log return
2. **Identifikasi kandidat window** dari spike PACF signifikan pada data stasioner (log return)
3. **Validasi empiris** via mini grid search menggunakan proxy GRU sederhana

Hasil: PACF log return menunjukkan **spike signifikan hanya di lag-1**, dikonfirmasi
oleh mini grid search yang memilih **window = 1** sebagai optimal.
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown(f"""
    ### Preprocessing
    - **Rentang data:** 2 Jan 2015 – 30 Des 2025
    - **Total observasi:** 2,790 hari trading
    - **Missing value:** Forward-fill (max 1 hari), lalu drop
    - **Normalisasi:** Min-Max scaling [0, 1]
    - **Stasionerisasi:** Log return untuk analisis ACF/PACF

    ### Sliding Window
    - **Window size:** {window_size} hari (hybrid PACF + empirical)
    - **Input shape:** ({window_size}, 3) — {window_size} hari × 3 fitur

    ### Train/Test Split
    - **Rasio:** 80:20 (kronologis, tanpa shuffling)
    """)

with col_m2:
    st.markdown("""
    ### Hyperparameter Tuning
    - **Metode:** Grid Search
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
    st.markdown(f"""
    1. **Window pendek (= {window_size} hari)** — Konsekuensi langsung dari PACF analysis pada log return.
       Meskipun secara teknis valid dan didukung mini grid search, kekuatan unique RNN
       (long-term memory) tidak teroptimalkan penuh. Model bertindak mirip dengan AR(1) non-linear.

    2. **Rentang data terbatas** — Model dilatih pada data 2015-2025. Untuk periode
       setelahnya, prediksi bersifat ekstrapolasi dan dapat mengalami *distribution shift*.

    3. **Iterative forecasting untuk multi-hari** — Prediksi lebih dari 1 hari ke depan
       dihitung secara rekursif, menyebabkan error terakumulasi seiring horizon.

    4. **Fitur eksternal terbatas** — Hanya menggunakan 3 fitur. Fitur lain seperti VIX,
       oil price, geopolitical index belum dipertimbangkan.

    ### Rekomendasi Future Work
    - Eksperimen dengan **window lebih panjang** untuk memaksimalkan keunggulan RNN
    - Implementasi retraining berkala dengan data terbaru
    - Pengembangan multi-output model untuk prediksi multi-step langsung
    - Penambahan fitur makroekonomi dan sentimen pasar
    - Ensemble method: kombinasi LSTM + GRU + Bi-LSTM
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
    Python · TensorFlow · Keras · Streamlit · Plotly · Yahoo Finance API · statsmodels
    """)
