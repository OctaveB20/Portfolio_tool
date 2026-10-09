"""app.py — Portfolio Analyzer · Streamlit dashboard."""

import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import (
    fetch_ticker_data, fetch_current_price,
    fetch_fundamentals, build_portfolio_snapshot,
)
from analytics import (
    full_stats, daily_returns, drawdown_series,
    rsi, macd, bollinger_bands, zscore,
    correlation_matrix, fifty_two_week_position,
    volatility_annualised, total_return,
)
import time
# st.cache_data.clear()  # Debug only: wipes the cache on every run (disabled so the contribution board stays fast)
# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
section[data-testid="stSidebar"] { background: #010409; border-right: 1px solid #21262d; }
h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 500; color: #c9d1d9; }
hr { border-color: #21262d; }
[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True),
)
GREEN = "#3fb950"
RED   = "#f85149"
BLUE  = "#58a6ff"
AMBER = "#e3b341"

# ── Load / Edit portfolio ─────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "portfolio.json")

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

cfg = load_config()

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Portfolio")
    period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"], index=4, key="sb_period")
    st.markdown("---")
    st.markdown("### Holdings")

    edited_holdings = []
    for i, h in enumerate(cfg["holdings"]):
        with st.expander(f"{h['ticker']} · {h['category']}", expanded=False):
            name      = st.text_input("Name",      h["name"],            key=f"sb_name_{i}")
            ticker    = st.text_input("Ticker",    h["ticker"],          key=f"sb_tick_{i}")
            cat       = st.selectbox("Category",
                                     ["Core ETF","Equity","Satellite","Speculative"],
                                     index=["Core ETF","Equity","Satellite","Speculative"].index(h["category"]),
                                     key=f"sb_cat_{i}")
            shares    = st.number_input("Shares",    value=float(h["shares"]),    min_value=0.0, step=0.01, key=f"sb_sh_{i}")
            avg_price = st.number_input("Avg Price", value=float(h["avg_price"]), min_value=0.0, step=0.01, key=f"sb_pr_{i}")
            edited_holdings.append({"ticker": ticker, "name": name, "shares": shares,
                                     "avg_price": avg_price, "category": cat})

    st.markdown("---")
    if st.button("＋ Add holding", key="sb_add"):
        cfg["holdings"].append({"ticker": "NEW", "name": "New Position",
                                 "shares": 1, "avg_price": 1.0, "category": "Equity"})
        save_config(cfg)
        st.rerun()

    if st.button("💾 Save changes", key="sb_save"):
        cfg["holdings"] = edited_holdings
        save_config(cfg)
        st.success("Saved!")

    benchmark_ticker = st.text_input("Benchmark", cfg.get("benchmark", "VWCE.AS"), key="sb_bench")

# ── Fetch all data ──────────────────────────────────────────────────────────

holdings = cfg["holdings"]

with st.spinner("Fetching market data…"):
    snapshot = build_portfolio_snapshot(holdings, period)

    price_series: dict[str, pd.Series] = {}
    for h in holdings:
        df = fetch_ticker_data(h["ticker"], period)
        if not df.empty:
            ps = df["Close"].dropna()  # Remove NaN values
            if len(ps) > 0:  # Only add if we have at least 1 data point
                price_series[h["ticker"]] = ps

    bench_df = fetch_ticker_data(benchmark_ticker, period)
    bench_prices = bench_df["Close"].dropna() if not bench_df.empty else None

# ── Stats ─────────────────────────────────────────────────────────────

all_stats = []
for h in holdings:
    ps = price_series.get(h["ticker"])
    if ps is not None and len(ps) > 1:  # Need at least 2 data points for return calculation
        s = full_stats(ps, bench_prices, label=h["ticker"])
        s["Name"]     = h["name"]
        s["Category"] = h["category"]
        all_stats.append(s)

stats_df = pd.DataFrame(all_stats) if all_stats else pd.DataFrame()

# ── Contribution board helpers ────────────────────────────────────────────

CONTRIB_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"]


@st.cache_data(ttl=900, show_spinner=False)
def period_return(ticker: str, p: str):
    """Price return over period p (native currency). None if not enough data."""
    fetch_p = "5d" if p == "1d" else p      # 1d = last close vs previous close
    df = fetch_ticker_data(ticker, fetch_p)
    if df is None or df.empty:
        return None
    c = df["Close"].dropna()
    if p == "1d":
        c = c.iloc[-2:]
    if len(c) < 2 or c.iloc[0] == 0:
        return None
    return float(c.iloc[-1] / c.iloc[0] - 1)


def contribution_table(snap: pd.DataFrame, p: str) -> pd.DataFrame:
    """Ranked contribution of each position to portfolio growth over period p."""
    rows = []
    for _, r in snap.iterrows():
        ret = period_return(r["Ticker"], p)
        if ret is None:
            continue
        mv_now = float(r["Market Value"])
        mv_start = mv_now / (1 + ret)            # value at start of window
        rows.append({
            "Ticker": r["Ticker"],
            "Name": r["Name"],
            "Return (%)": ret * 100,
            "Gain (€)": mv_now - mv_start,
            "_start": mv_start,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # contribution = gain of the position / total portfolio value at the start
    df["Contribution (pts)"] = df["Gain (€)"] / df["_start"].sum() * 100
    df = df.drop(columns="_start").sort_values("Contribution (pts)", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)

# ── Header ─────────────────────────────────────────────────────────────

total_cost    = snapshot["Cost Basis"].sum()
total_value   = snapshot["Market Value"].sum()
total_pnl     = total_value - total_cost
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

st.markdown("# portfolio_analyzer")
st.caption(f"Data via Yahoo Finance · ~15 min delay · {period} window")
st.markdown("---")

hdr1, hdr2, hdr3, hdr4 = st.columns(4)
hdr1.metric("Portfolio Value", f"€{total_value:,.2f}", f"€{total_pnl:+,.2f}")
hdr2.metric("Total P&L", f"{total_pnl_pct:+.2f}%", "vs cost basis")
hdr3.metric("Positions", str(len(holdings)))
hdr4.metric("Period", period)

# ── TABS ──────────────────────────────────────────────────────────────

tab_overview, tab_individual, tab_risk, tab_correlations, tab_fundamentals = st.tabs([
    "📊 Overview", "🔍 Deep Dive", "⚡ Risk", "🔗 Correlations", "📋 Fundamentals"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("### Portfolio Snapshot")

    display_df = snapshot.copy()
    display_df["Cost Basis"]    = display_df["Cost Basis"].map("€{:,.2f}".format)
    display_df["Market Value"]  = display_df["Market Value"].map("€{:,.2f}".format)
    display_df["Avg Price"] = display_df.apply(
    lambda r: f"{r['Avg Price']:.2f} {r['Currency']}",
    axis=1
)
    display_df["Current Price"] = display_df.apply(
        lambda r: f"{r['Current Price (native)']} {r['Currency']} → €{r['Current Price']}", axis=1
    )
    display_df["Weight (%)"]    = display_df["Weight (%)"].map("{:.1f}%".format)
    display_df["P&L (%)"]       = display_df["P&L (%)"].map("{:+.2f}%".format)
    display_df["P&L (€)"]       = display_df["P&L (€)"].map("€{:+,.2f}".format)

    st.dataframe(
        display_df[["Ticker","Name","Category","Shares","Avg Price","Current Price",
                    "Cost Basis","Market Value","P&L (€)","P&L (%)","Weight (%)"]],
        width="stretch", hide_index=True,
    )

    st.markdown("---")
    ov_col1, ov_col2 = st.columns(2)

    with ov_col1:
        st.markdown("#### Allocation by market value")
        fig = px.pie(snapshot, names="Ticker", values="Market Value",
                     hole=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch", key="ov_pie")

    with ov_col2:
        st.markdown("#### P&L per position (%)")
        df_sorted = snapshot.sort_values("P&L (%)")
        bar_colors = [GREEN if v >= 0 else RED for v in df_sorted["P&L (%)"]]
        fig = go.Figure(go.Bar(
            x=df_sorted["Ticker"], y=df_sorted["P&L (%)"],
            marker_color=bar_colors,
            text=df_sorted["P&L (%)"].map("{:+.1f}%".format),
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title="P&L %")
        st.plotly_chart(fig, width="stretch", key="ov_pnl_bar")

    st.markdown("#### Category breakdown")
    cat_df = snapshot.groupby("Category").agg(
        Value=("Market Value", "sum"),
        PnL=("P&L (€)", "sum"),
        Positions=("Ticker", "count")
    ).reset_index()
    cat_df["Weight"] = (cat_df["Value"] / cat_df["Value"].sum() * 100).map("{:.1f}%".format)
    cat_df["Value"]  = cat_df["Value"].map("€{:,.2f}".format)
    cat_df["PnL"]    = cat_df["PnL"].map("€{:+,.2f}".format)
    st.dataframe(cat_df, width="stretch", hide_index=True, key="ov_cat_table")

    if price_series:
        st.markdown("#### Cumulative return vs benchmark")
        fig = go.Figure()
        port_returns = pd.DataFrame({k: daily_returns(v) for k, v in price_series.items()})
        weights_raw = snapshot.set_index("Ticker")["Market Value"]
        common_tickers = [t for t in port_returns.columns if t in weights_raw.index]
        if common_tickers:
            w = weights_raw[common_tickers]
            w = w / w.sum()
            weighted_ret = port_returns[common_tickers].mul(w.values, axis=1).sum(axis=1)
            port_cum = (1 + weighted_ret).cumprod() - 1
            fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum * 100,
                                     name="Portfolio", line=dict(color=BLUE, width=2)))
        if bench_prices is not None:
            bench_ret = daily_returns(bench_prices)
            bench_cum = (1 + bench_ret).cumprod() - 1
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum * 100,
                                     name=benchmark_ticker,
                                     line=dict(color=AMBER, width=1.5, dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, height=340,
                          yaxis_title="Cumulative Return (%)", hovermode="x unified")
        st.plotly_chart(fig, width="stretch", key="ov_cumret")

    # ── Contribution board ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Contribution to portfolio growth")
    st.caption("Which positions drove the portfolio's performance, ranked. "
               "Contribution (pts) = position gain ÷ total portfolio value at the start of the window.")

    with st.spinner("Computing contributions…"):
        contrib = {p: contribution_table(snapshot, p) for p in CONTRIB_PERIODS}

    # Heatmap: all periods at a glance
    valid = {p: d.set_index("Ticker")["Contribution (pts)"]
             for p, d in contrib.items() if not d.empty}
    if valid:
        heat = pd.DataFrame(valid)
        sort_col = period if period in heat.columns else heat.columns[0]
        heat = heat.sort_values(sort_col, ascending=False)
        zmax = float(np.nanmax(np.abs(heat.values))) or 1
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale=[[0.0, RED], [0.5, "#0d1117"], [1.0, GREEN]],
            zmid=0, zmin=-zmax, zmax=zmax,
            text=np.round(heat.values, 2), texttemplate="%{text}",
            hoverongaps=False, colorbar=dict(title="pts"),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=max(260, 34 * len(heat) + 80))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch", key="ov_contrib_heat")

    # Ranked detail per period
    ctabs = st.tabs(CONTRIB_PERIODS)
    for ctab, p in zip(ctabs, CONTRIB_PERIODS):
        with ctab:
            d = contrib[p]
            if d.empty:
                st.info("Not enough data for this period.")
                continue

            total_pts = d["Contribution (pts)"].sum()
            total_eur = d["Gain (€)"].sum()
            k1, k2, k3 = st.columns(3)
            k1.metric("Portfolio growth", f"{total_pts:+.2f}%", f"€{total_eur:+,.2f}")
            k2.metric("Top contributor", d.iloc[0]["Ticker"],
                      f"{d.iloc[0]['Contribution (pts)']:+.2f} pts")
            k3.metric("Biggest drag", d.iloc[-1]["Ticker"],
                      f"{d.iloc[-1]['Contribution (pts)']:+.2f} pts")

            fig = go.Figure(go.Bar(
                x=d["Contribution (pts)"], y=d["Ticker"], orientation="h",
                marker_color=[GREEN if v >= 0 else RED for v in d["Contribution (pts)"]],
                text=d["Contribution (pts)"].map("{:+.2f}".format),
                textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(240, 32 * len(d) + 80),
                              xaxis_title="Contribution (pts)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch", key=f"ov_contrib_bar_{p}")

            show = d.copy()
            show["Return (%)"] = show["Return (%)"].map("{:+.2f}%".format)
            show["Gain (€)"] = show["Gain (€)"].map("€{:+,.2f}".format)
            show["Contribution (pts)"] = show["Contribution (pts)"].map("{:+.2f}".format)
            st.dataframe(show, width="stretch", hide_index=True, key=f"ov_contrib_tbl_{p}")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — INDIVIDUAL DEEP DIVE
# ══════════════════════════════════════════════════════════════════

with tab_individual:
    ticker_choice = st.selectbox(
        "Select position",
        options=[h["ticker"] for h in holdings],
        format_func=lambda t: f"{t} · {next((h['name'] for h in holdings if h['ticker']==t), t)}",
        key="dd_ticker_select",
    )

    ps      = price_series.get(ticker_choice)
    holding = next((h for h in holdings if h["ticker"] == ticker_choice), None)

    if ps is None or holding is None:
        st.warning("No price data available for this ticker.")
    else:
        s  = full_stats(ps, bench_prices, label=ticker_choice)
        fw = fifty_two_week_position(ps)

        dd_kpi1 = st.columns(5)
        dd_kpi1[0].metric("Total Return", f"{s['Total Return (%)']:+.2f}%")
        dd_kpi1[1].metric("Ann. Return",  f"{s['Ann. Return (%)']:+.2f}%")
        dd_kpi1[2].metric("Sharpe",       f"{s['Sharpe']:.2f}")
        dd_kpi1[3].metric("Max Drawdown", f"{s['Max Drawdown (%)']:.2f}%")
        dd_kpi1[4].metric("Volatility",   f"{s['Volatility (%)']:.2f}%")

        dd_kpi2 = st.columns(4)
        dd_kpi2[0].metric("Sortino",      f"{s['Sortino']:.2f}")
        dd_kpi2[1].metric("Calmar",       f"{s['Calmar']:.2f}")
        dd_kpi2[2].metric("Beta",         f"{s['Beta']:.2f}" if s['Beta'] else "—")
        dd_kpi2[3].metric("Alpha (ann.)", f"{s['Alpha (ann. %)']:+.2f}%" if s['Alpha (ann. %)'] else "—")

        if fw is not None:
            position_pct = fw.get("position_pct")

            if pd.notna(position_pct):
                # Streamlit progress expects a value between 0 and 100
                position_pct = max(0, min(100, float(position_pct)))

                st.markdown("#### 52-week range")
                st.progress(int(position_pct))
                st.caption(
                    f"Low: {fw['52w_low']}  ·  "
                    f"Current: {fw['current']}  ·  "
                    f"High: {fw['52w_high']}  ·  "
                    f"Position: {position_pct:.0f}%ile"
                )
            else:
                st.markdown("#### 52-week range")
                st.info("52-week position unavailable for this ticker.")

        st.markdown("---")

        st.markdown("#### Price · Bollinger Bands (20, 2σ)")
        upper, mid, lower = bollinger_bands(ps)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ps.index, y=upper, name="Upper", line=dict(color=RED,   width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=ps.index, y=mid,   name="MA20",  line=dict(color=AMBER, width=1)))
        fig.add_trace(go.Scatter(x=ps.index, y=lower, name="Lower", line=dict(color=GREEN, width=1, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(63,185,80,0.05)"))
        fig.add_trace(go.Scatter(x=ps.index, y=ps,    name="Price", line=dict(color=BLUE,  width=2)))
        fig.add_hline(y=holding["avg_price"], line=dict(color=AMBER, dash="dash", width=1),
                      annotation_text=f"PRU {holding['avg_price']}", annotation_position="right")
        fig.update_layout(**PLOTLY_LAYOUT, height=340, yaxis_title="Price")
        st.plotly_chart(fig, width="stretch", key="dd_bollinger")

        dd_col1, dd_col2 = st.columns(2)

        with dd_col1:
            st.markdown("#### RSI (14)")
            rsi_series = rsi(ps)
            fig = go.Figure()
            fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.07, line_width=0)
            fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.07, line_width=0)
            fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series,
                                     line=dict(color=BLUE, width=2), name="RSI"))
            fig.add_hline(y=70, line=dict(color=RED,   width=0.8, dash="dot"))
            fig.add_hline(y=30, line=dict(color=GREEN, width=0.8, dash="dot"))
            fig.update_layout(**PLOTLY_LAYOUT, height=260)
            fig.update_yaxes(range=[0, 100], gridcolor="#21262d")
            st.plotly_chart(fig, width="stretch", key="dd_rsi")
            rsi_now = rsi_series.iloc[-1]
            if rsi_now > 70:
                st.warning(f"RSI {rsi_now:.0f} — overbought zone")
            elif rsi_now < 30:
                st.success(f"RSI {rsi_now:.0f} — oversold zone")
            else:
                st.info(f"RSI {rsi_now:.0f} — neutral")

        with dd_col2:
            st.markdown("#### MACD (12/26/9)")
            macd_line, signal_line, histogram = macd(ps)
            fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                                shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=ps.index, y=ps, name="Price",
                                     line=dict(color=BLUE, width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=macd_line.index,   y=macd_line,   name="MACD",
                                     line=dict(color=BLUE,  width=1.5)), row=2, col=1)
            fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, name="Signal",
                                     line=dict(color=AMBER, width=1.5)), row=2, col=1)
            hist_colors = [GREEN if v >= 0 else RED for v in histogram]
            fig.add_trace(go.Bar(x=histogram.index, y=histogram, name="Hist",
                                 marker_color=hist_colors, opacity=0.6), row=2, col=1)
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            fig.update_yaxes(gridcolor="#21262d")
            st.plotly_chart(fig, width="stretch", key="dd_macd")

        st.markdown("#### Drawdown from peak")
        dd_series = drawdown_series(ps)
        fig = go.Figure(go.Scatter(x=dd_series.index, y=dd_series, fill="tozeroy",
                                   fillcolor="rgba(248,81,73,0.15)",
                                   line=dict(color=RED, width=1.5)))
        fig.update_layout(**PLOTLY_LAYOUT, height=240, yaxis_title="Drawdown %")
        st.plotly_chart(fig, width="stretch", key="dd_drawdown")

        st.markdown("#### Rolling Z-score (20d) — mean reversion signal")
        z = zscore(ps)
        fig = go.Figure(go.Scatter(x=z.index, y=z, line=dict(color=BLUE, width=1.5)))
        fig.add_hline(y=2,  line=dict(color=RED,      dash="dot", width=1))
        fig.add_hline(y=-2, line=dict(color=GREEN,    dash="dot", width=1))
        fig.add_hline(y=0,  line=dict(color="#8b949e", width=0.5))
        fig.update_layout(**PLOTLY_LAYOUT, height=240, yaxis_title="Z-score")
        st.plotly_chart(fig, width="stretch", key="dd_zscore")
        z_now = z.iloc[-1]
        if abs(z_now) > 2:
            st.warning(f"Z-score = {z_now:.2f} → price significantly {'above' if z_now>0 else 'below'} recent average")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — RISK
# ══════════════════════════════════════════════════════════════════

with tab_risk:
    st.markdown("### Risk metrics across all positions")

    if not stats_df.empty:
        risk_cols = ["Ticker","Name","Category","Volatility (%)","Max Drawdown (%)",
                     "Sharpe","Sortino","Calmar","VaR 95% (daily %)","Beta","Alpha (ann. %)"]
        st.dataframe(stats_df[[c for c in risk_cols if c in stats_df.columns]],
                     width="stretch", hide_index=True, key="risk_table")

    st.markdown("---")
    risk_col1, risk_col2 = st.columns(2)

    if not stats_df.empty:
        with risk_col1:
            st.markdown("#### Risk / Return scatter")
            fig = px.scatter(
                stats_df, x="Volatility (%)", y="Ann. Return (%)",
                color="Category", text="Ticker", size_max=18,
                color_discrete_map={
                    "Core ETF": BLUE, "Equity": GREEN,
                    "Satellite": AMBER, "Speculative": RED,
                },
            )
            fig.update_traces(textposition="top center")
            fig.add_hline(y=0, line=dict(color="#8b949e", width=0.8, dash="dot"))
            fig.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig, width="stretch", key="risk_scatter")

        with risk_col2:
            st.markdown("#### Sharpe ratio by position")
            df_s = stats_df.sort_values("Sharpe")
            sharpe_colors = [GREEN if v >= 1 else AMBER if v >= 0 else RED for v in df_s["Sharpe"]]
            fig = go.Figure(go.Bar(
                x=df_s["Sharpe"], y=df_s["Ticker"], orientation="h",
                marker_color=sharpe_colors,
                text=df_s["Sharpe"].map("{:.2f}".format),
                textposition="outside",
            ))
            fig.add_vline(x=1, line=dict(color=GREEN, dash="dot", width=1))
            fig.add_vline(x=0, line=dict(color=RED,   dash="dot", width=1))
            fig.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_title="Sharpe ratio")
            st.plotly_chart(fig, width="stretch", key="risk_sharpe")

    if not stats_df.empty:
        st.markdown("#### Max drawdown by position")
        df_d = stats_df.sort_values("Max Drawdown (%)")
        fig = go.Figure(go.Bar(
            x=df_d["Ticker"], y=df_d["Max Drawdown (%)"],
            marker_color=RED, opacity=0.8,
            text=df_d["Max Drawdown (%)"].map("{:.1f}%".format),
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Max Drawdown %")
        st.plotly_chart(fig, width="stretch", key="risk_mdd")

        st.markdown("#### Value at Risk (95%, daily)")
        fig = go.Figure(go.Bar(
            x=stats_df["Ticker"], y=stats_df["VaR 95% (daily %)"],
            marker_color=AMBER,
            text=stats_df["VaR 95% (daily %)"].map("{:.2f}%".format),
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig, width="stretch", key="risk_var")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════

with tab_correlations:
    st.markdown("### Correlation matrix — daily returns")

    if len(price_series) >= 2:
        corr = correlation_matrix(price_series)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0.0, "#f85149"], [0.5, "#0d1117"], [1.0, "#3fb950"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            hoverongaps=False,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=520)
        st.plotly_chart(fig, width="stretch", key="corr_heatmap")

        st.markdown("#### Insights")
        corr_vals  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_stack = corr_vals.stack().sort_values()
        if not corr_stack.empty:
            most_pos = corr_stack.iloc[-1]
            most_neg = corr_stack.iloc[0]
            st.info(f"**Most correlated pair:** {corr_stack.index[-1][0]} × {corr_stack.index[-1][1]} → {most_pos:.2f}")
            st.info(f"**Least correlated pair:** {corr_stack.index[0][0]} × {corr_stack.index[0][1]} → {most_neg:.2f}")
    else:
        st.info("Need at least 2 positions with data to build a correlation matrix.")

# ══════════════════════════════════════════════════════════════════
# TAB 5 — FUNDAMENTALS
# ══════════════════════════════════════════════════════════════════

with tab_fundamentals:
    st.markdown("### Fundamental data (equities & ETFs)")

    fund_ticker = st.selectbox(
        "Select position",
        options=[h["ticker"] for h in holdings],
        format_func=lambda t: f"{t} · {next((h['name'] for h in holdings if h['ticker']==t), t)}",
        key="fund_ticker_select",
    )

    with st.spinner("Loading fundamentals…"):
        fund = fetch_fundamentals(fund_ticker)

    if not any(v is not None for v in fund.values()):
        st.info("No fundamental data available for this ticker (common for ETFs).")
    else:
        fund_cols = st.columns(3)
        metrics = [
            ("Trailing P/E",   fund.get("trailingPE"),     None),
            ("Forward P/E",    fund.get("forwardPE"),      None),
            ("Price / Book",   fund.get("priceToBook"),    None),
            ("Dividend Yield", fund.get("dividendYield"),  "%"),
            ("Beta",           fund.get("beta"),           None),
            ("ROE",            fund.get("returnOnEquity"), "%"),
            ("52W Change",     fund.get("52WeekChange"),   "%"),
            ("Short Ratio",    fund.get("shortRatio"),     None),
            ("EPS (trailing)", fund.get("trailingEps"),    None),
        ]
        for i, (label, val, suffix) in enumerate(metrics):
            if val is not None:
                display = f"{val*100:.2f}%" if suffix == "%" else f"{val:.2f}"
                fund_cols[i % 3].metric(label, display)

        if fund.get("sector"):
            st.caption(f"Sector: **{fund.get('sector')}**   ·   Industry: {fund.get('industry','—')}")
        if fund.get("marketCap"):
            mc = fund["marketCap"]
            st.caption(f"Market cap: **{'${:.1f}B'.format(mc/1e9) if mc >= 1e9 else '${:.0f}M'.format(mc/1e6)}**")

        low  = fund.get("fiftyTwoWeekLow")
        high = fund.get("fiftyTwoWeekHigh")
        cur  = fetch_current_price(fund_ticker)
        if low and high and cur:
            pct = (cur - low) / (high - low) * 100 if high != low else 50
            st.markdown("#### 52-week range")
            st.progress(int(pct))
            st.caption(f"Low: {low}  ·  Current: {cur}  ·  High: {high}")
