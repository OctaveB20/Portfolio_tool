"""data.py — fetch and cache price data via yfinance."""

import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)   # taux EUR/USD rafraîchi toutes les 5 min
def get_eur_usd() -> float:
    """Fetch current EUR/USD rate."""
    try:
        hist = yf.Ticker("EURUSD=X").history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            if 0.5 < rate < 2.0:   # sanity check
                return rate
    except Exception:
        pass
    return 1.10


@st.cache_data(ttl=900)
def fetch_ticker_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV DataFrame for a ticker."""
    try:
        t = yf.Ticker(ticker)
        interval = "5m" if period == "1d" else "1h" if period == "5d" else "1d"
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        else:
            df.index = pd.to_datetime(df.index)
        result = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        result = result.dropna(subset=["Close"])
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)   # prix rafraîchis toutes les 5 min
def fetch_current_price(ticker: str) -> float | None:
    """
    Return the latest price in native currency.
    Returns None if unavailable — never falls back to avg_price.
    """
    try:
        t = yf.Ticker(ticker)
        price = getattr(t.fast_info, "last_price", None)
        if price and float(price) > 0:
            return round(float(price), 4)
    except Exception:
        pass
    try:
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if not hist.empty:
            closes = hist["Close"].dropna()
            if not closes.empty:
                return round(float(closes.iloc[-1]), 4)
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    """Return key fundamental data from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        keys = [
            "trailingPE", "forwardPE", "priceToBook", "dividendYield",
            "beta", "marketCap", "trailingEps", "sector", "industry",
            "52WeekChange", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "averageVolume", "shortRatio", "returnOnEquity",
        ]
        return {k: info.get(k) for k in keys}
    except Exception:
        return {}


def build_portfolio_snapshot(holdings: list, period: str = "1y") -> pd.DataFrame:
    """
    Build snapshot DataFrame with P&L for each holding.

    Convention (matches sidebar input):
      - USD tickers (no suffix): avg_price in USD → convert to EUR
      - EUR tickers (with suffix): avg_price in EUR → no conversion
    """
    eur_usd = get_eur_usd()
    rows = []

    for h in holdings:
        ticker    = h["ticker"]
        is_usd    = "." not in ticker
        avg_price = h["avg_price"]   # native currency as entered

        current_native = fetch_current_price(ticker)

        if current_native is None:
            # No data → show 0 P&L rather than misleading number
            current_native = avg_price
            price_stale    = True
        else:
            price_stale = False

        if is_usd:
            current_eur = current_native / eur_usd
            avg_eur     = avg_price      / eur_usd
        else:
            current_eur = current_native
            avg_eur     = avg_price

        cost_basis   = avg_eur     * h["shares"]
        market_value = current_eur * h["shares"]
        pnl_abs      = market_value - cost_basis
        pnl_pct      = (pnl_abs / cost_basis * 100) if cost_basis else 0.0

        rows.append({
            "Ticker":                 ticker,
            "Name":                   h["name"],
            "Category":               h["category"],
            "Shares":                 h["shares"],
            "Avg Price":              round(avg_price, 4),
            "Current Price":          round(current_eur, 4),
            "Current Price (native)": round(current_native, 4),
            "Currency":               "USD" if is_usd else "EUR",
            "Cost Basis":             cost_basis,
            "Market Value":           market_value,
            "P&L (€)":                pnl_abs,
            "P&L (%)":                pnl_pct,
            "Stale":                  price_stale,
        })

    df = pd.DataFrame(rows)
    total_mv = df["Market Value"].sum()
    df["Weight (%)"] = (df["Market Value"] / total_mv * 100) if total_mv else 0.0
    return df
