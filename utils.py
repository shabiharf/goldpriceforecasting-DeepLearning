"""
Utility functions untuk Gold Price Forecasting Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from pathlib import Path
from datetime import datetime

TROY_OZ_TO_GRAM = 31.1035
WINDOW_SIZE = 60
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


@st.cache_resource
def load_models_and_scalers():
    """Load ketiga model + scaler. Return (None, None, None) kalau file tidak lengkap."""
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
            return None, None, None

        models = {name: load_model(path, compile=False) for name, path in model_files.items()}
        scaler = joblib.load(scaler_path)
        scaler_target = joblib.load(scaler_target_path)
        return models, scaler, scaler_target
    except Exception as e:
        st.warning(f"Model belum ter-load: {e}. Dashboard jalan di mode demo.")
        return None, None, None


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
    """Coba fetch yfinance. Return None kalau gagal atau data terlalu sedikit."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        gold_usd = yf.download("GC=F", start=start_date, end=end_date,
                                auto_adjust=True, progress=False)["Close"].squeeze()
        usdidr = yf.download("IDR=X", start=start_date, end=end_date,
                              auto_adjust=True, progress=False)["Close"].squeeze()
        ihsg = yf.download("^JKSE", start=start_date, end=end_date,
                            auto_adjust=True, progress=False)["Close"].squeeze()

        if len(gold_usd) < 60 or len(usdidr) < 60 or len(ihsg) < 60:
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

        if len(df) < 60:
            return None
        return df
    except Exception:
        return None


def get_market_data():
    """
    Strategi: coba yfinance dulu, kalau gagal pakai historical CSV.
    Return (df, source_label).
    """
    df_live = try_fetch_yfinance()
    if df_live is not None and len(df_live) >= 60:
        return df_live, "Yahoo Finance (live)"

    df_historical = load_historical_data()
    if df_historical is not None and len(df_historical) >= 60:
        return df_historical, "Historical CSV (offline fallback)"

    return None, "No data available"


def predict_next_day(model, scaler, scaler_target, df):
    if model is None:
        return float(df["Gold_IDR_gram"].iloc[-1]) * 1.007

    data_scaled = scaler.transform(df[["Gold_IDR_gram", "USDIDR", "IHSG"]].values)
    last_window = data_scaled[-WINDOW_SIZE:]
    X = last_window.reshape(1, WINDOW_SIZE, 3)
    pred_norm = model.predict(X, verbose=0).flatten()
    pred_idr = scaler_target.inverse_transform(pred_norm.reshape(-1, 1)).flatten()[0]
    return float(pred_idr)


def predict_n_days_iterative(model, scaler, scaler_target, df, n_days=7):
    if model is None:
        last_price = df["Gold_IDR_gram"].iloc[-1]
        return np.array([last_price * (1 + 0.007 * (i + 1)) for i in range(n_days)])

    data_scaled = scaler.transform(df[["Gold_IDR_gram", "USDIDR", "IHSG"]].values)
    current_window = data_scaled[-WINDOW_SIZE:].copy()

    predictions = []
    for _ in range(n_days):
        X = current_window.reshape(1, WINDOW_SIZE, 3)
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
