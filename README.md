# 📈 Gold Price Forecasting Dashboard

Interactive dashboard untuk peramalan harga emas Indonesia menggunakan deep learning models (LSTM, GRU, Bi-LSTM). Dibangun sebagai bagian dari thesis program studi Sains Data, Telkom University.

**🔴 Live Demo:** [Coming soon — akan diisi setelah deploy]

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
| **GRU** ⭐ | **2.41%** | 54,216 | 40,218 |
| Bi-LSTM | 4.59% | 106,384 | 78,793 |
| LSTM | 6.98% | 161,259 | 120,196 |

Model **GRU** menjadi model terbaik dengan MAPE terendah.

---

## ✨ Fitur Dashboard

- 🏠 **Dashboard Utama** — Overview live dengan metric cards, chart aktual vs prediksi, dan forecast N hari ke depan
- 🔮 **Prediksi** — Forecasting interaktif dengan pilihan model (GRU/LSTM/Bi-LSTM) dan horizon kustom (1-60 hari), dilengkapi export CSV
- 📊 **Data Explorer** — EDA lengkap: time series, heatmap korelasi, ACF/PACF, statistik deskriptif
- 🧠 **Tentang Model** — Dokumentasi arsitektur, hyperparameter hasil grid search, dan metodologi

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.11 (rekomendasi) atau 3.10
- Git (untuk clone)

### Langkah Setup

```bash
# Clone repository
git clone https://github.com/<username>/gold-price-forecasting.git
cd gold-price-forecasting

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📁 Struktur Project

```
gold-price-forecasting/
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
│   ├── best_model_GRU.h5
│   ├── best_model_LSTM.h5
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
| Window size | 60 hari |
| Train/Test split | 80:20 (kronologis) |
| Hyperparameter tuning | Grid Search (8 kombinasi/model) |
| Optimizer | Adam |
| Loss function | MSE |

---

## ⚠️ Limitations

1. Model dilatih pada data 2015-2025 — prediksi setelah Des 2025 adalah ekstrapolasi
2. Multi-day forecast menggunakan iterative approach — error terakumulasi seiring horizon
3. Hanya 3 fitur input (emas USD, USD/IDR, IHSG) — belum termasuk VIX, oil price, sentimen pasar

---

## 👤 Author

**Shabiha Rahma Fauziah**
- NIM: 1206220017
- Program Studi Sains Data, Telkom University Surabaya
- Judul Tugas Akhir: *Peramalan Harga Emas Indonesia Menggunakan LSTM, GRU, dan Bi-LSTM*

---

## 📝 License

Academic project. Dibangun sebagai bagian dari Tugas Akhir S1 Sains Data, Telkom University Surabaya.
