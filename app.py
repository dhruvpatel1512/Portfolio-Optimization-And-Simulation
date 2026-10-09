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
# ---- walk-forward backtest -----------------------------------------------
def backtest(rets, years, freq, objective, cap, rf, cost):
    """Re-optimise on the trailing window at each period start; hold until the next one."""
    lb, n = int(years * DAYS), rets.shape[1]
    starts = [rets.index.get_loc(d) for d in rets.groupby(rets.index.to_period(freq)).head(1).index]
    starts = [p for p in starts if p >= lb] + [len(rets)]
    out = {"Optimised": [], "Equal Weight": []}
    prev = {k: None for k in out}
    for a, b in zip(starts, starts[1:]):
        hist = rets.iloc[a - lb:a]  # strictly before the rebalance day: no look-ahead
        w = optimise(hist.mean().values * DAYS, hist.cov().values * DAYS, rf, cap, objective)
        targets = {"Optimised": w if w is not None else (prev["Optimised"] if prev["Optimised"] is not None else np.full(n, 1 / n)),
                   "Equal Weight": np.full(n, 1 / n)}
        for k, w in targets.items():
            value = ((1 + rets.iloc[a:b]).cumprod() * w).sum(axis=1)  # buy-and-hold drift inside the period
            r = value.pct_change()
            r.iloc[0] = value.iloc[0] - 1
            # ponytail: turnover measured vs previous target, ignoring intra-period drift
            r.iloc[0] -= cost * (np.abs(w - prev[k]).sum() if prev[k] is not None else 1)
            out[k].append(r)
            prev[k] = w
    return pd.DataFrame({k: pd.concat(v) for k, v in out.items()})


def perf(r, rf):
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(DAYS)
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": (cagr - rf) / vol,
            "Max drawdown": (wealth / wealth.cummax() - 1).min()}


st.subheader("Walk-forward backtest (out-of-sample)")
b1, b2, b3, b4, b5 = st.columns(5)
bt_years = b1.slider("Lookback (years)", 1, 10, 3)
bt_freq = {"Monthly": "M", "Quarterly": "Q"}[b2.selectbox("Rebalance", ["Monthly", "Quarterly"])]
bt_obj = {"Max Sharpe": "sharpe", "Min Volatility": "vol"}[b3.selectbox("Objective", ["Max Sharpe", "Min Volatility"])]
bt_cost = b4.number_input("Cost (bps per unit turnover)", 0, 100, 10) / 1e4
if len(rets) <= bt_years * DAYS + DAYS:
    st.info("Not enough history for this lookback - reduce it or pick assets with longer history.")
else:
    bt = backtest(rets, bt_years, bt_freq, bt_obj, cap, rf, bt_cost)
    fig_bt = go.Figure([go.Scatter(x=bt.index, y=(1 + bt[k]).cumprod(), name=k) for k in bt])
    fig_bt.update_layout(yaxis_title="Growth of 1", height=450)
    st.plotly_chart(fig_bt, use_container_width=True)
    st.dataframe(pd.DataFrame({k: perf(bt[k], rf) for k in bt}).T.style.format(
        {"CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}", "Max drawdown": "{:.2%}"}))
    st.caption(f"Weights re-estimated each period from the prior {bt_years} years only; "
               f"out-of-sample from {bt.index[0].date()}.")

st.caption("Inputs are historical estimates - not a forecast or investment advice.")
