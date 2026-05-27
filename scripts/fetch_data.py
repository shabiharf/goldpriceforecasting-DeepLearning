"""
Script untuk GitHub Actions — fetch data terbaru dari Yahoo Finance,
update data/historical_data.csv.

Dijalankan otomatis oleh GitHub Actions setiap hari.
IP GitHub Actions tidak diblok Yahoo, jadi fetch berhasil di sini
meskipun di Streamlit Cloud gagal.
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

TROY_OZ_TO_GRAM = 31.1035
START_DATE = "2015-01-01"
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "historical_data.csv"


def fetch_with_session():
    """Fetch 3 ticker pakai curl_cffi session untuk reliability."""
    session = None
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        print("✅ Pakai curl_cffi session")
    except ImportError:
        print("⚠️ curl_cffi tidak ada, pakai default session")

    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching data {START_DATE} s/d {end_date}...")

    gold_usd = yf.download("GC=F", start=START_DATE, end=end_date,
                            auto_adjust=True, progress=False,
                            session=session)["Close"].squeeze()
    usdidr = yf.download("IDR=X", start=START_DATE, end=end_date,
                          auto_adjust=True, progress=False,
                          session=session)["Close"].squeeze()
    ihsg = yf.download("^JKSE", start=START_DATE, end=end_date,
                        auto_adjust=True, progress=False,
                        session=session)["Close"].squeeze()

    print(f"  Gold USD : {len(gold_usd)} baris")
    print(f"  USD/IDR  : {len(usdidr)} baris")
    print(f"  IHSG     : {len(ihsg)} baris")

    return gold_usd, usdidr, ihsg


def main():
    gold_usd, usdidr, ihsg = fetch_with_session()

    if len(gold_usd) < 60 or len(usdidr) < 60 or len(ihsg) < 60:
        print("❌ Data tidak cukup, abort. CSV lama tidak diubah.")
        return

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
    df.index.name = "Date"

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"✅ Data tersimpan: {OUTPUT_PATH}")
    print(f"   Total: {len(df)} baris, terakhir: {df.index[-1].date()}")


if __name__ == "__main__":
    main()
