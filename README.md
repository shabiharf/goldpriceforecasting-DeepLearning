# 📈 Gold Price Forecasting Dashboard

Interactive dashboard untuk peramalan harga emas Indonesia menggunakan deep learning models (LSTM, GRU, Bi-LSTM). Dibangun sebagai bagian dari Tugas Akhir program studi Sains Data, Telkom University Surabaya.

**🔴 Live Demo:** https://goldpriceforecasting-deeplearning-fzqpqonkzjmuv5d9a6ywak.streamlit.app/

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 🎯 Overview

Dashboard ini mengimplementasikan dan membandingkan tiga model deep learning untuk meramalkan harga emas Indonesia (IDR/gram) berdasarkan data multivariat:

- **Harga emas global** (USD/oz, `GC=F`)
- **Nilai tukar** USD/IDR
- **IHSG** (Indeks Harga Saham Gabungan)

### Hasil Evaluasi (Test Set 2023-2025)

| Model | MAPE | RMSE | MAE |
|-------|:----:|:----:|:---:|
| **LSTM** ⭐ | **0.94%** | 21,679 | 14,407 |
| Bi-LSTM | 1.67% | 35,038 | 26,817 |
| GRU | 2.53% | 49,780 | 40,340 |

Model **LSTM** menjadi model terbaik dengan MAPE terendah.

---

## ✨ Fitur Dashboard

- 🏠 **Dashboard Utama** — Overview live dengan metric cards, chart aktual vs prediksi, dan forecast N hari ke depan
- 🔮 **Prediksi** — Forecasting interaktif dengan pilihan model (LSTM/GRU/Bi-LSTM) dan horizon kustom (1-60 hari), dilengkapi export CSV
- 📊 **Data Explorer** — EDA lengkap: time series, heatmap korelasi, uji stasioneritas (ADF), ACF/PACF, statistik deskriptif
- 🧠 **Tentang Model** — Dokumentasi arsitektur, hyperparameter hasil grid search, dan metodologi hybrid window selection

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11 (rekomendasi) atau 3.10
- Git (untuk clone)

### Langkah Setup

```bash
# Clone repository
git clone https://github.com/shabihaf/goldpriceforecasting-DeepLearning.git
cd goldpriceforecasting-DeepLearning

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📁 Struktur Project

```
goldpriceforecasting-DeepLearning/
├── app.py                         # Halaman Dashboard utama
├── utils.py                       # Helper functions
├── requirements.txt               # Python dependencies
├── README.md
├── .gitignore
├── data/                          # Data artifacts
│   ├── historical_data.csv
│   ├── test_predictions.csv
│   └── evaluation_metrics.csv
├── models/                        # Trained models
│   ├── best_model_LSTM.h5
│   ├── best_model_GRU.h5
│   ├── best_model_BiLSTM.h5
│   ├── scaler.pkl
│   └── scaler_target.pkl
└── pages/                         # Halaman multi-page
    ├── 1_🔮_Prediksi.py
    ├── 2_📊_Data_Explorer.py
    └── 3_🧠_Tentang_Model.py
```

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Frontend:** Streamlit
- **Visualization:** Plotly
- **Data Source:** Yahoo Finance (via `yfinance`)
- **ML Utilities:** scikit-learn, statsmodels
- **Deployment:** Streamlit Community Cloud

---

## 📊 Metodologi

| Aspek | Nilai |
|-------|-------|
| Periode data | 2 Jan 2015 – 30 Des 2025 |
| Total observasi | 2,790 hari trading |
| Uji stasioneritas | Augmented Dickey-Fuller (ADF) |
| Window selection | Hybrid: PACF + validasi empiris (grid search) |
| Window size terpilih | 1 hari |
| Train/Test split | 80:20 (kronologis) |
| Hyperparameter tuning | Grid Search |
| Optimizer | Adam |
| Loss function | MSE |

### Uji Stasioneritas (ADF)

| Variabel | Data Mentah | Log Return |
|----------|:-----------:|:----------:|
| Harga Emas | ❌ Non-stasioner | ✅ Stasioner |
| USD/IDR | ❌ Non-stasioner | ✅ Stasioner |
| IHSG | ❌ Non-stasioner | ✅ Stasioner |

Window size ditentukan melalui hybrid approach: identifikasi lag signifikan dari PACF pada data stasioner (log return), kemudian divalidasi secara empiris melalui mini grid search. Hasil: PACF menunjukkan spike signifikan pada lag-1, dikonfirmasi grid search yang memilih window = 1 sebagai optimal.

---

## ⚠️ Limitations

1. Window pendek (= 1 hari) — konsekuensi dari analisis PACF; kekuatan long-term memory RNN tidak teroptimalkan penuh
2. Model dilatih pada data 2015-2025 — prediksi setelah Des 2025 adalah ekstrapolasi
3. Multi-day forecast menggunakan iterative approach — error terakumulasi seiring horizon
4. Hanya 3 fitur input (emas USD, USD/IDR, IHSG) — belum termasuk VIX, oil price, sentimen pasar

---

## 👤 Author

**Shabiha Rahma Fauziah**
- NIM: 1206220017
- Program Studi Sains Data, Telkom University Surabaya
- Judul Tugas Akhir: *Peramalan Harga Emas Indonesia Menggunakan LSTM, GRU, dan Bi-LSTM*

---

## 📝 License

Academic project. Dibangun sebagai bagian dari Tugas Akhir S1 Sains Data, Telkom University Surabaya.
