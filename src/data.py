from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from .config import FRED_SERIES

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _monthly_last(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().resample("ME").last()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s.astype(float)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_yahoo_monthly(ticker: str, start: str, end: Optional[str] = None) -> pd.Series:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"Yahoo Finance 無資料：{ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    close.name = ticker
    return _monthly_last(close)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_fred_monthly(series_id: str, start: str, end: Optional[str] = None) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    if df.empty or len(df.columns) < 2:
        raise RuntimeError(f"FRED 無資料：{series_id}")

    date_col, value_col = df.columns[:2]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    s = df.dropna().set_index(date_col)[value_col]
    s = s.loc[pd.Timestamp(start):]
    if end:
        s = s.loc[:pd.Timestamp(end)]
    s.name = series_id
    return _monthly_last(s)


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("CSV 為空白。")

    date_candidates = [c for c in df.columns if c.lower() in {"date", "month", "年月", "日期"}]
    date_col = date_candidates[0] if date_candidates else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.resample("ME").last()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


def fetch_public_macro(start: str, end: Optional[str] = None) -> tuple[pd.DataFrame, Dict[str, str]]:
    data: Dict[str, pd.Series] = {}
    status: Dict[str, str] = {}

    for name, series_id in FRED_SERIES.items():
        try:
            data[name] = fetch_fred_monthly(series_id, start, end)
            status[name] = f"成功：FRED {series_id}"
        except Exception as exc:
            status[name] = f"失敗：{exc}"

    if not data:
        return pd.DataFrame(), status
    return pd.concat(data, axis=1), status


def build_market_features(price: pd.Series) -> pd.DataFrame:
    price = _monthly_last(price).dropna()
    ret_1m = price.pct_change()
    ret_3m = price.pct_change(3)
    ma10 = price.rolling(10).mean()

    # 價格均線訊號：高於 10 月均線為正。
    price_trend = (price / ma10 - 1.0)

    # 單一指數環境沒有完整成分股 breadth 時，使用中期報酬與短期報酬一致度做代理。
    breadth_proxy = (
        np.sign(ret_1m).rolling(6).mean()
        + np.sign(ret_3m).rolling(6).mean()
    ) / 2.0

    momentum = price.pct_change(6)
    volatility = ret_1m.rolling(12).std() * np.sqrt(12)

    return pd.DataFrame({
        "PRICE": price,
        "PRICE_TREND": price_trend,
        "MARKET_BREADTH": breadth_proxy,
        "MARKET_MOMENTUM": momentum,
        "VOLATILITY_REGIME": volatility,
    })


def derive_proxy_indicators(macro: pd.DataFrame) -> pd.DataFrame:
    out = macro.copy()

    # 美國 LEI 代理：殖利率曲線、新訂單、住宅、初領失業救濟、
    # 工業生產與流動性的等權組合，最終仍會個別標準化。
    cols = [
        c for c in [
            "US_YIELD_CURVE",
            "US_NEW_ORDERS",
            "US_HOUSING",
            "US_INITIAL_CLAIMS",
            "US_INDUSTRIAL_PRODUCTION",
            "US_LIQUIDITY",
        ] if c in out.columns
    ]
    if cols:
        temp = []
        for c in cols:
            s = out[c].copy()
            if c == "US_INITIAL_CLAIMS":
                s = -s
            temp.append(s.pct_change(12).rename(c))
        out["US_LEI_PROXY"] = pd.concat(temp, axis=1).mean(axis=1)

    # 若沒有正式台灣資料，以下欄位刻意不虛構。
    return out


def merge_all_data(
    macro: pd.DataFrame,
    features: pd.DataFrame,
    uploaded: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames = [macro, features]
    if uploaded is not None and not uploaded.empty:
        frames.append(uploaded)
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    return df.sort_index()
