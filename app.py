import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

DAYS = 252
PRESETS = {
    "Stocks": "JPM GS MS AAPL MSFT NVDA AMZN XOM CVX COP SLB",
    "US equity ETFs": "SPY QQQ IWM VTI",
    "Intl equity ETFs": "EFA EEM VEA",
    "Bond ETFs": "AGG TLT LQD HYG TIP",
    "Commodity ETFs": "GLD SLV USO DBC",
    "Real estate ETFs": "VNQ",
    "Crypto": "BTC-USD ETH-USD",
}

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer")


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(tickers, start):
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    return raw.to_frame(tickers[0]) if isinstance(raw, pd.Series) else raw


@st.cache_data(ttl=3600)
def latest_rf():
    # 13-week T-bill yield (^IRX is quoted in percent)
    try:
        return float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    except Exception:
        return 0.04


def stats(w, mu, cov, rf):
    r, v = w @ mu, np.sqrt(w @ cov @ w)
    return r, v, (r - rf) / v


def optimise(mu, cov, rf, cap, objective, target=None):
    n = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target})
    f = {"sharpe": lambda w: -stats(w, mu, cov, rf)[2],
         "vol": lambda w: w @ cov @ w}[objective]
    res = minimize(f, np.full(n, 1 / n), bounds=[(0, cap)] * n, constraints=cons, method="SLSQP")
    return res.x if res.success else None


# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("Universe")
    picked = st.multiselect("Asset classes", list(PRESETS), default=["Stocks"])
    extra = st.text_input("Extra tickers (space separated)", "")
    start = st.date_input("Start date", pd.Timestamp("2010-01-01")).isoformat()
    st.header("Settings")
    rf = st.number_input("Risk-free rate", 0.0, 0.2, round(latest_rf(), 4), 0.005, format="%.3f")
    cap = st.slider("Max weight per asset", 0.05, 1.0, 1.0, 0.05)
    min_years = st.slider("Drop assets with less history than (years)", 0, 15, 3)

tickers = sorted({t for p in picked for t in PRESETS[p].split()} | set(extra.upper().split()))
if len(tickers) < 2:
    st.info("Pick at least two assets.")
    st.stop()
if cap * len(tickers) < 1:
    st.error(f"Max weight {cap:.0%} is too low for {len(tickers)} assets (needs >= {1/len(tickers):.0%}).")
    st.stop()

# ---- data ----------------------------------------------------------------
prices = load_prices(tuple(tickers), start)
missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
prices = prices.drop(columns=missing)
short = [t for t in prices if prices[t].dropna().shape[0] < min_years * DAYS]
prices = prices.drop(columns=short)
if missing or short:
    st.warning(f"No data: {missing or '-'} | too little history: {short or '-'}")
if prices.shape[1] < 2:
    st.stop()

# Calendars differ (crypto trades 24/7): align on common dates, never forward-fill into fake returns.
prices = prices.dropna()
rets = prices.pct_change().dropna()
names = list(rets.columns)
mu, cov = rets.mean().values * DAYS, rets.cov().values * DAYS
st.caption(f"{len(names)} assets | {rets.index[0].date()} to {rets.index[-1].date()} "
           f"({len(rets)} trading days, limited by the shortest-history asset)")

# ---- optimise ------------------------------------------------------------
w_sh, w_mv = optimise(mu, cov, rf, cap, "sharpe"), optimise(mu, cov, rf, cap, "vol")
if w_sh is None or w_mv is None:
    st.error("Optimizer failed - try loosening the constraints.")
    st.stop()
w_eq = np.full(len(names), 1 / len(names))
portfolios = {"Max Sharpe": w_sh, "Min Volatility": w_mv, "Equal Weight": w_eq}

table = pd.DataFrame(
    {k: stats(w, mu, cov, rf) for k, w in portfolios.items()},
    index=["Return", "Volatility", "Sharpe"]).T
st.subheader("Results")
st.dataframe(table.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}"}))

# efficient frontier: min variance for a grid of target returns
fr = []
for t in np.linspace(stats(w_mv, mu, cov, rf)[0], mu.max() * 0.999 if cap == 1 else mu.max(), 40):
    w = optimise(mu, cov, rf, cap, "vol", t)
    if w is not None:
        fr.append(stats(w, mu, cov, rf)[:2])
fig = go.Figure()
if fr:
    fr = np.array(fr)
    fig.add_scatter(x=fr[:, 1], y=fr[:, 0], mode="lines", name="Efficient frontier")
fig.add_scatter(x=np.sqrt(np.diag(cov)), y=mu, mode="markers+text", text=names,
                textposition="top center", name="Assets")
for k, w in portfolios.items():
    r, v, _ = stats(w, mu, cov, rf)
    fig.add_scatter(x=[v], y=[r], mode="markers", name=k, marker=dict(size=14, symbol="star"))
fig.update_layout(xaxis_title="Volatility", yaxis_title="Expected return",
                  xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=550)

c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, use_container_width=True)
wdf = pd.DataFrame(portfolios, index=names)
c2.plotly_chart(go.Figure([go.Bar(x=names, y=wdf[k], name=k) for k in ["Max Sharpe", "Min Volatility"]])
                .update_layout(barmode="group", yaxis_tickformat=".0%", height=550), use_container_width=True)

corr = rets.corr()
st.subheader("Correlation")
st.plotly_chart(go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1,
                                     colorscale="RdBu_r", text=corr.round(2).values,
                                     texttemplate="%{text}")).update_layout(height=600),
                use_container_width=True)
st.caption("Inputs are historical, in-sample estimates - not a forecast or investment advice.")
