"""
Utility functions untuk Gold Price Forecasting Dashboard v3
Updated: window size auto-detect dari model shape, support ADF results
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from pathlib import Path
from datetime import datetime

TROY_OZ_TO_GRAM = 31.1035
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def load_models_and_scalers():
    """Load ketiga model + scaler. Return (None, None, None, None) kalau file tidak lengkap.
    Sekarang juga return window_size yang ter-detect dari model shape."""
    try:
        from tensorflow.keras.models import load_model
        import joblib

        model_files = {
            "GRU": MODEL_DIR / "best_model_GRU.h5",
            "LSTM": MODEL_DIR / "best_model_LSTM.h5",
            "Bi-LSTM": MODEL_DIR / "best_model_BiLSTM.h5",
        }
        scaler_path = MODEL_DIR / "scaler.pkl"
        scaler_target_path = MODEL_DIR / "scaler_target.pkl"

        if not all(p.exists() for p in list(model_files.values()) + [scaler_path, scaler_target_path]):
            return None, None, None, None

        models = {name: load_model(path, compile=False) for name, path in model_files.items()}
        scaler = joblib.load(scaler_path)
        scaler_target = joblib.load(scaler_target_path)

        # Auto-detect window size dari input shape model (input_shape[0] = time_steps)
        sample_model = next(iter(models.values()))
        window_size = sample_model.input_shape[1]

        return models, scaler, scaler_target, window_size
    except Exception as e:
        st.warning(f"Model belum ter-load: {e}. Dashboard jalan di mode demo.")
        return None, None, None, None


@st.cache_data
def get_best_model_name():
    """Auto-detect best model dari evaluation_metrics.csv (model dengan MAPE terendah)."""
    metrics = load_evaluation_metrics()
    if metrics is None:
        return "LSTM"  # default fallback
    return metrics.loc[metrics["MAPE (%)"].idxmin(), "Model"]


@st.cache_data
def load_test_predictions():
    path = DATA_DIR / "test_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data
def load_evaluation_metrics():
    path = DATA_DIR / "evaluation_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_historical_data():
    """Load data historis lengkap dari CSV."""
    path = DATA_DIR / "historical_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=3600)
def try_fetch_yfinance(start_date="2015-01-01", end_date=None):
    """Coba fetch yfinance dengan curl_cffi session untuk bypass cloud IP blocks."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        session = None
        try:
            from curl_cffi import requests as curl_requests
            session = curl_requests.Session(impersonate="chrome")
        except ImportError:
            pass

        gold_usd = yf.download("GC=F", start=start_date, end=end_date,
                                auto_adjust=True, progress=False,
                                session=session)["Close"].squeeze()
        usdidr = yf.download("IDR=X", start=start_date, end=end_date,
                              auto_adjust=True, progress=False,
                              session=session)["Close"].squeeze()
        ihsg = yf.download("^JKSE", start=start_date, end=end_date,
                            auto_adjust=True, progress=False,
                            session=session)["Close"].squeeze()

        # Untuk window pendek (=1), 30 hari aja udah cukup
        if len(gold_usd) < 30 or len(usdidr) < 30 or len(ihsg) < 30:
            return None

        all_dates = gold_usd.index.union(usdidr.index).union(ihsg.index)
        df = pd.DataFrame({
            "Gold_USD_oz": gold_usd,
            "USDIDR": usdidr,
            "IHSG": ihsg,
        }, index=all_dates)

        df["Gold_IDR_gram"] = (df["Gold_USD_oz"] / TROY_OZ_TO_GRAM) * df["USDIDR"]
        df = df[["Gold_IDR_gram", "USDIDR", "IHSG"]]
        df = df.ffill(limit=1).dropna()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if len(df) < 30:
            return None
        return df
    except Exception:
        return None


def get_market_data():
    """Strategi: coba yfinance dulu, kalau gagal pakai historical CSV."""
    df_live = try_fetch_yfinance()
    if df_live is not None and len(df_live) >= 30:
        return df_live, "Yahoo Finance (live)"

    df_historical = load_historical_data()
    if df_historical is not None and len(df_historical) >= 30:
        return df_historical, "Historical CSV (offline fallback)"

    return None, "No data available"


def predict_next_day(model, scaler, scaler_target, df, window_size):
    """Prediksi harga emas besok pakai window_size hari terakhir."""
    if model is None:
        return float(df["Gold_IDR_gram"].iloc[-1]) * 1.007

    data_scaled = scaler.transform(df[["Gold_IDR_gram", "USDIDR", "IHSG"]].values)
    last_window = data_scaled[-window_size:]
    X = last_window.reshape(1, window_size, 3)
    pred_norm = model.predict(X, verbose=0).flatten()
    pred_idr = scaler_target.inverse_transform(pred_norm.reshape(-1, 1)).flatten()[0]
    return float(pred_idr)


def predict_n_days_iterative(model, scaler, scaler_target, df, window_size, n_days=7):
    """Prediksi n hari ke depan dengan iterative approach."""
    if model is None:
        last_price = df["Gold_IDR_gram"].iloc[-1]
        return np.array([last_price * (1 + 0.007 * (i + 1)) for i in range(n_days)])

    data_scaled = scaler.transform(df[["Gold_IDR_gram", "USDIDR", "IHSG"]].values)
    current_window = data_scaled[-window_size:].copy()

    predictions = []
    for _ in range(n_days):
        X = current_window.reshape(1, window_size, 3)
        pred_norm = model.predict(X, verbose=0).flatten()[0]
        new_row = np.array([pred_norm, current_window[-1, 1], current_window[-1, 2]])
        current_window = np.vstack([current_window[1:], new_row])
        predictions.append(pred_norm)

    pred_array = np.array(predictions).reshape(-1, 1)
    return scaler_target.inverse_transform(pred_array).flatten()


def format_idr(value):
    if value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f}Jt"
    return f"Rp {value:,.0f}"


# ============= ADF Test Results (dari notebook v3) =============
ADF_RESULTS = {
    "raw": {
        "Harga Emas": {"adf_stat": 5.228427, "p_value": 1.000000, "stationary": False},
        "USD/IDR":    {"adf_stat": -1.853472, "p_value": 0.354256, "stationary": False},
        "IHSG":       {"adf_stat": -0.482179, "p_value": 0.895447, "stationary": False},
    },
    "log_return": {
        "Harga Emas": {"adf_stat": -22.837159, "p_value": 0.000000, "stationary": True},
        "USD/IDR":    {"adf_stat": -10.904223, "p_value": 0.000000, "stationary": True},
        "IHSG":       {"adf_stat": -15.656576, "p_value": 0.000000, "stationary": True},
    },
    "critical_values": {"1%": -3.4327, "5%": -2.8626, "10%": -2.5673},
}
