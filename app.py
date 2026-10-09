import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import linprog, minimize
from sklearn.covariance import LedoitWolf

DAYS = 252
PACKS = {
    "Stocks (original 11)": "JPM GS MS AAPL MSFT NVDA AMZN XOM CVX COP SLB",
    "US equity ETFs": "SPY QQQ IWM VTI",
    "Intl equity ETFs": "EFA EEM VEA",
    "Bond ETFs": "AGG TLT LQD HYG TIP",
    "Commodity ETFs": "GLD SLV USO DBC",
    "Real estate ETFs": "VNQ",
    "Crypto": "BTC-USD ETH-USD",
}
TYPES = {"EQUITY": "Stock", "ETF": "ETF", "MUTUALFUND": "Fund", "CRYPTOCURRENCY": "Crypto"}

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimizer")
st.caption("Pick stocks, ETFs, bond funds or crypto, then see how to split your money, how that split would "
           "have held up in a backtest, and what it could be worth later. Built on past prices from Yahoo Finance.")
ss = st.session_state

# ---- universe state: ss.known maps symbol -> label, ss.universe is the selection ----
if "known" not in ss:
    ss.known, ss.universe = {}, []
    for t in PACKS["Stocks (original 11)"].split():
        ss.known[t] = "Stock"
    ss.universe = sorted(ss.known)


def add(symbols, label):
    for s in symbols:
        ss.known.setdefault(s, label)
    ss.universe = sorted(set(ss.universe) | set(symbols))


@st.cache_data(ttl=3600)
def search(q):
    try:
        return [(x["symbol"], f'{x["symbol"]} - {x.get("shortname") or x.get("longname", "")} ({TYPES[x["quoteType"]]})',
                 TYPES[x["quoteType"]]) for x in yf.Search(q, max_results=10).quotes
                if x.get("quoteType") in TYPES]
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(tickers, start):
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    return raw.to_frame(tickers[0]) if isinstance(raw, pd.Series) else raw


@st.cache_data(ttl=3600)
def latest_rf():
    try:  # 13-week T-bill yield, quoted in percent
        return float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
    except Exception:
        return 0.04


# ---- maths ---------------------------------------------------------------
def estimate(r, lw, shrink, mu_tilt=None):
    """Annual mean/cov. Returns are shrunk toward their cross-asset mean; cov optionally Ledoit-Wolf.

    mu_tilt: optional per-asset annual return adjustment (array aligned to r's columns). This is the single
    extension point for outside views such as market sentiment (see README roadmap).
    """
    mu = r.mean().values * DAYS
    mu = (1 - shrink) * mu + shrink * mu.mean()
    if mu_tilt is not None:
        mu = mu + mu_tilt
    cov = (LedoitWolf().fit(r.values).covariance_ if lw else r.cov().values) * DAYS
    return mu, cov


def stats(w, mu, cov, rf):
    r, v = w @ mu, np.sqrt(w @ cov @ w)
    return r, v, (r - rf) / v


def optimise(mu, cov, rf, lo, hi, objective, target=None):
    n = len(mu)
    if objective == "sharpe":
        def f(w):
            return -(w @ mu - rf) / np.sqrt(w @ cov @ w)

        def g(w):
            v = np.sqrt(w @ cov @ w)
            return -(mu / v - (w @ mu - rf) * (cov @ w) / v ** 3)
    else:
        def f(w):
            return w @ cov @ w

        def g(w):
            return 2 * cov @ w
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    if target is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target, "jac": lambda w: mu})
    x0 = lo + (hi - lo) * (1 - lo.sum()) / max((hi - lo).sum(), 1e-12)  # feasible start
    res = minimize(f, x0, jac=g, bounds=list(zip(lo, hi)), constraints=cons, method="SLSQP")
    return res.x if res.success else None


@st.cache_data
def frontier(mu, cov, rf, lo, hi):
    w0 = optimise(mu, cov, rf, lo, hi, "vol")
    top = -linprog(-mu, A_eq=[np.ones(len(mu))], b_eq=[1], bounds=list(zip(lo, hi))).fun
    pts = []
    for t in np.linspace(w0 @ mu, top, 40):
        w = optimise(mu, cov, rf, lo, hi, "vol", t)
        if w is not None:
            pts.append(stats(w, mu, cov, rf)[:2])
    return np.array(pts)


@st.cache_data(show_spinner="Running backtest...")
def backtest(rets, years, freq, objective, lo, hi, rf, cost, lw, shrink):
    """Re-optimise on the trailing window at each period start; hold until the next one."""
    lb, n = int(years * DAYS), rets.shape[1]
    starts = [rets.index.get_loc(d) for d in rets.groupby(rets.index.to_period(freq)).head(1).index]
    starts = [p for p in starts if p >= lb] + [len(rets)]
    out = {"Optimised": [], "Equal Weight": []}
    prev = {k: None for k in out}
    for a, b in zip(starts, starts[1:]):
        mu, cov = estimate(rets.iloc[a - lb:a], lw, shrink)  # strictly before the rebalance day: no look-ahead
        w = optimise(mu, cov, rf, lo, hi, objective)
        if w is None:
            w = prev["Optimised"] if prev["Optimised"] is not None else np.full(n, 1 / n)
        for k, wk in {"Optimised": w, "Equal Weight": np.full(n, 1 / n)}.items():
            value = ((1 + rets.iloc[a:b]).cumprod() * wk).sum(axis=1)  # buy-and-hold drift inside the period
            r = value.pct_change()
            r.iloc[0] = value.iloc[0] - 1
            # ponytail: turnover vs previous target, ignoring intra-period drift
            r.iloc[0] -= cost * (np.abs(wk - prev[k]).sum() if prev[k] is not None else 1)
            out[k].append(r)
            prev[k] = wk
    return pd.DataFrame({k: pd.concat(v) for k, v in out.items()})


def perf(r, rf):
    wealth = (1 + r).cumprod()
    cagr = wealth.iloc[-1] ** (DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(DAYS)
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": (cagr - rf) / vol,
            "Max drawdown": (wealth / wealth.cummax() - 1).min()}


# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("1. Build your universe")
    q = st.text_input("Search any stock, ETF, bond fund or crypto", placeholder="e.g. vanguard bond, TLT, bitcoin")
    hits = search(q) if q else []
    if q and not hits:
        st.caption("No results.")
    if hits:
        pick = st.selectbox("Results", hits, format_func=lambda h: h[1])
        st.button("Add to universe", on_click=add, args=([pick[0]], pick[2]), type="primary")
    pack = st.selectbox("...or add a ready-made pack", list(PACKS))
    st.button("Add pack", on_click=add, args=(PACKS[pack].split(), pack.split(" (")[0]))
    st.multiselect("Your assets (remove with x)", sorted(ss.known), key="universe",
                   format_func=lambda s: f"{s} · {ss.known[s]}")

    st.header("2. Settings")
    start = st.date_input("History from", pd.Timestamp("2010-01-01")).isoformat()
    rf = st.number_input("Risk-free rate", 0.0, 0.2, round(latest_rf(), 4), 0.005, format="%.3f")
    cap = st.slider("Default max weight per asset", 0.05, 1.0, 1.0, 0.05)
    min_years = st.slider("Drop assets with less history than (years)", 0, 15, 3)
    st.header("3. Estimation")
    lw = st.checkbox("Ledoit-Wolf shrinkage covariance", True,
                     help="Smooths the covariance matrix so a few noisy pairs of assets don't drive the weights.")
    shrink = st.slider("Shrink expected returns toward the average", 0.0, 1.0, 0.0, 0.1,
                       help="Pulls each asset's historical return toward the group average. On a test universe of stocks, "
                            "bonds and gold it lowered out-of-sample Sharpe, so it starts at 0.")
    st.checkbox("Market-sentiment tilt (planned)", False, disabled=True,
                help="Not built yet. See the README roadmap.")

tickers = list(ss.universe)
if len(tickers) < 2:
    st.info("Add at least two assets in the sidebar.")
    st.stop()

# ---- data ----------------------------------------------------------------
prices = load_prices(tuple(tickers), start)
missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
prices = prices.drop(columns=missing)
short = [t for t in prices if prices[t].dropna().shape[0] < min_years * DAYS]
prices = prices.drop(columns=short)
if missing or short:
    st.warning(f"No data for: {missing or 'none'}. Too little history: {short or 'none'}.")
if prices.shape[1] < 2:
    st.stop()

# Calendars differ (crypto trades 24/7): align on common dates, never forward-fill into fake returns.
rets = prices.dropna().pct_change().dropna()
names = list(rets.columns)
classes = [ss.known.get(t, "") for t in names]
n = len(names)
mu, cov = estimate(rets, lw, shrink)
st.caption(f"{n} assets, {rets.index[0].date()} to {rets.index[-1].date()} "
           f"({len(rets)} trading days; the asset with the shortest history sets the start date)")

tab_opt, tab_me, tab_bt, tab_fc, tab_risk = st.tabs(["Optimize", "My portfolio", "Backtest", "Forecast", "Risk & simulation"])
key = "|".join(names)

# per-asset limits
with tab_opt:
    with st.expander("Per-asset limits (edit min / max weight)"):
        lim = st.data_editor(
            pd.DataFrame({"Asset": names, "Min %": 0.0, "Max %": cap * 100}), hide_index=True, key="lim" + key,
            column_config={"Asset": st.column_config.TextColumn(disabled=True),
                           "Min %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0),
                           "Max %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0)})
lo, hi = lim["Min %"].values / 100, lim["Max %"].values / 100
if (lo > hi).any() or lo.sum() > 1 or hi.sum() < 1:
    st.error("These limits can't work. Each min must be at or below its max, the mins must add up to 100% or less, and the maxes to 100% or more.")
    st.stop()

# money to invest / already invested
with tab_me:
    mode = st.radio("What do you want to do?", ["Invest new money", "I already hold these assets"], horizontal=True)
    if mode == "Invest new money":
        held = np.zeros(n)
        cash = st.number_input("Amount to invest ($)", 0, 10 ** 9, 10000, 500)
    else:
        h = st.data_editor(
            pd.DataFrame({"Asset": names, "Class": classes, "Invested $": 0.0}), hide_index=True, key="held" + key,
            column_config={"Asset": st.column_config.TextColumn(disabled=True),
                           "Class": st.column_config.TextColumn(disabled=True),
                           "Invested $": st.column_config.NumberColumn(min_value=0.0, format="%.2f")})
        held = h["Invested $"].values.astype(float)
        cash = st.number_input("Extra cash to add ($, optional)", 0, 10 ** 9, 0, 500)
    total = held.sum() + cash
    w_me = held / held.sum() if held.sum() > 0 else None
    whole = st.checkbox("Whole shares only", False, help="Otherwise fractional shares/units are assumed.")

w_sh, w_mv = optimise(mu, cov, rf, lo, hi, "sharpe"), optimise(mu, cov, rf, lo, hi, "vol")
if w_sh is None or w_mv is None:
    st.error("The optimizer couldn't find a solution. Try loosening the limits.")
    st.stop()
portfolios = {"Max Sharpe": w_sh, "Min Volatility": w_mv, "Equal Weight": np.full(n, 1 / n)}
if w_me is not None:
    portfolios["My Portfolio"] = w_me

# ---- Optimize tab --------------------------------------------------------
with tab_opt:
    table = pd.DataFrame({k: stats(w, mu, cov, rf) for k, w in portfolios.items()},
                         index=["Return", "Volatility", "Sharpe"]).T
    st.dataframe(table.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}"}))
    st.caption("Returns use the estimation settings in the sidebar.")

    fr = frontier(mu, cov, rf, lo, hi)
    fig = go.Figure()
    if len(fr):
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

    out = wdf.copy()
    out.insert(0, "Class", classes)
    st.dataframe(out.style.format({c: "{:.1%}" for c in portfolios}))
    st.download_button("Download weights (CSV)", out.to_csv().encode(), "weights.csv")

    corr = rets.corr()
    st.subheader("Correlation")
    st.plotly_chart(go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
                                         text=corr.round(2).values, texttemplate="%{text}")
                              ).update_layout(height=600), use_container_width=True)

# ---- My portfolio tab ----------------------------------------------------
with tab_me:
    target_name = st.selectbox("Recommended allocation to follow", ["Max Sharpe", "Min Volatility", "Equal Weight"])
    tgt = portfolios[target_name]
    if total <= 0:
        st.info("Enter an amount to invest, or the amounts you already hold.")
    else:
        px = np.array([prices[t].dropna().iloc[-1] for t in names])
        trade = tgt * total - held
        shares = np.trunc(trade / px) if whole else trade / px  # truncate toward zero: never oversell
        after = held + shares * px
        plan = pd.DataFrame({"Class": classes, "Current $": held, "Current %": held / total * 100,
                             "Target %": tgt * 100, "Target $": tgt * total, "Trade $": shares * px,
                             "Price": px, "Shares": shares}, index=names)
        st.dataframe(plan.style.format({"Current $": "{:,.0f}", "Current %": "{:.1f}", "Target %": "{:.1f}",
                                        "Target $": "{:,.0f}", "Trade $": "{:+,.0f}", "Price": "{:,.2f}",
                                        "Shares": "{:+,.0f}" if whole else "{:+,.4f}"}))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total after plan", f"${after.sum():,.0f}")
        c2.metric("Uninvested cash", f"${total - after.sum():,.2f}")
        c3.metric("Prices as of", str(prices.index[-1].date()))
        st.download_button("Download plan (CSV)", plan.to_csv().encode(), "investment_plan.csv")
        if w_me is not None:
            a, b = stats(w_me, mu, cov, rf), stats(tgt, mu, cov, rf)
            st.write(f"**Your holdings:** return {a[0]:.2%}, vol {a[1]:.2%}, Sharpe {a[2]:.2f}  |  "
                     f"**{target_name}:** return {b[0]:.2%}, vol {b[1]:.2%}, Sharpe {b[2]:.2f}")
        byclass = pd.DataFrame({"Current": held / total, target_name: tgt}, index=names).groupby(classes).sum()
        st.dataframe(byclass.style.format("{:.1%}"))
        st.caption("If you already hold assets, the plan may include sells. Taxes, fees and minimum order sizes are ignored.")

# ---- Backtest tab --------------------------------------------------------
with tab_bt:
    b1, b2, b3, b4 = st.columns(4)
    bt_years = b1.slider("Lookback (years)", 1, 10, 3)
    bt_freq = {"Monthly": "M", "Quarterly": "Q"}[b2.selectbox("Rebalance", ["Monthly", "Quarterly"])]
    bt_obj = {"Max Sharpe": "sharpe", "Min Volatility": "vol"}[b3.selectbox("Objective", ["Max Sharpe", "Min Volatility"])]
    bt_cost = b4.number_input("Cost (bps per unit turnover)", 0, 100, 10) / 1e4
    if len(rets) <= (bt_years + 1) * DAYS:
        st.info("Not enough history for this lookback. Shorten it or pick assets with longer histories.")
    else:
        bt = backtest(rets, bt_years, bt_freq, bt_obj, lo, hi, rf, bt_cost, lw, shrink)
        st.plotly_chart(go.Figure([go.Scatter(x=bt.index, y=(1 + bt[k]).cumprod(), name=k) for k in bt])
                        .update_layout(yaxis_title="Growth of 1", height=450), use_container_width=True)
        st.dataframe(pd.DataFrame({k: perf(bt[k], rf) for k in bt}).T.style.format(
            {"CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.3f}", "Max drawdown": "{:.2%}"}))
        st.caption(f"Each period's weights use only the prior {bt_years} years of data. Out-of-sample from "
                   f"{bt.index[0].date()}, with the estimation settings from the sidebar.")

# ---- Forecast tab --------------------------------------------------------
Z = {5: -1.645, 25: -0.674, 50: 0.0, 75: 0.674, 95: 1.645}


def band(m, sd, v0, t):
    """Normal-return (geometric Brownian motion) percentiles of portfolio value after t years."""
    centre = np.log(v0) + (m - 0.5 * sd ** 2) * t
    return {p: np.exp(centre + z * sd * np.sqrt(t)) for p, z in Z.items()}


def fan_chart(x, b):
    fig = go.Figure()
    for lo_p, hi_p, name in [(5, 95, "5-95%"), (25, 75, "25-75%")]:
        fig.add_scatter(x=x, y=b[hi_p], line_width=0, showlegend=False)
        fig.add_scatter(x=x, y=b[lo_p], fill="tonexty", line_width=0, name=name)
    fig.add_scatter(x=x, y=b[50], name="Median")
    return fig


with tab_fc:
    st.write("Projects portfolio value assuming normally distributed returns (geometric Brownian motion), using "
             "the expected return and volatility from the sidebar settings.")
    f1, f2, f3 = st.columns(3)
    fc_pf = f1.selectbox("Portfolio", list(portfolios), key="fc_pf")
    fc_yrs = f2.slider("Horizon (years)", 1, 10, 3, key="fc_h")
    fc_v0 = f3.number_input("Starting value ($)", 1000, 10 ** 9, 10000, 1000, key="fc_v")
    m_, sd_, _ = stats(portfolios[fc_pf], mu, cov, rf)
    t = np.arange(1, fc_yrs * DAYS + 1) / DAYS
    bd = band(m_, sd_, fc_v0, t)
    fig_fc = fan_chart(t, bd)
    fig_fc.update_layout(xaxis_title="Years", yaxis_title="Portfolio value ($)", height=450)
    st.plotly_chart(fig_fc, use_container_width=True)
    yrs = list(range(1, fc_yrs + 1))
    st.dataframe(pd.DataFrame({
        "Expected value": [fc_v0 * np.exp(m_ * y) for y in yrs],
        "Median": [bd[50][y * DAYS - 1] for y in yrs],
        "Low (5%)": [bd[5][y * DAYS - 1] for y in yrs],
        "High (95%)": [bd[95][y * DAYS - 1] for y in yrs]}, index=[f"Year {y}" for y in yrs]
    ).style.format("${:,.0f}"))
    st.caption(f"Assumes a constant {m_:.1%} return and {sd_:.1%} volatility per year. Real returns have fatter "
               "tails and volatility that moves, so extreme outcomes are understated. Not a prediction.")

    st.subheader("Reality check: the same method on the last year")
    pr_f = rets.values @ portfolios[fc_pf]
    m_h, sd_h = pr_f[:-DAYS].mean() * DAYS, pr_f[:-DAYS].std() * np.sqrt(DAYS)  # fitted on data before the last year
    actual = np.cumprod(1 + pr_f[-DAYS:])
    th = np.arange(1, DAYS + 1) / DAYS
    bh = band(m_h, sd_h, 1.0, th)
    inside = ((actual >= bh[5]) & (actual <= bh[95])).mean()
    fig_rc = fan_chart(th, bh)
    fig_rc.add_scatter(x=th, y=actual, name="Actual", line=dict(color="black"))
    fig_rc.update_layout(xaxis_title="Years into the test year", yaxis_title="Growth of 1", height=350)
    st.plotly_chart(fig_rc, use_container_width=True)
    st.write(f"Fitted on data from before the last year, the 5-95% band held the actual path on **{inside:.0%}** of days. "
             "One year is a single draw, so this is a plausibility check, not proof. The weights came from the "
             "full history, so it isn't a clean out-of-sample test. The Backtest tab is.")

# ---- Risk tab ------------------------------------------------------------
with tab_risk:
    which = st.selectbox("Portfolio", list(portfolios), key="risk_pf")
    pr = rets.values @ portfolios[which]  # ponytail: assumes daily rebalancing to fixed weights
    q5 = np.percentile(pr, 5)
    m1, m2, m3 = st.columns(3)
    m1.metric("1-day VaR (95%)", f"{-q5:.2%}")
    m2.metric("1-day CVaR (95%)", f"{-pr[pr <= q5].mean():.2%}")
    wealth = np.cumprod(1 + pr)
    m3.metric("Max drawdown", f"{(wealth / np.maximum.accumulate(wealth) - 1).min():.1%}")
    st.plotly_chart(go.Figure(go.Scatter(x=rets.index, y=wealth / np.maximum.accumulate(wealth) - 1, fill="tozeroy"))
                    .update_layout(title="Drawdown", yaxis_tickformat=".0%", height=300), use_container_width=True)

    st.subheader("Forward simulation (bootstrap of historical daily returns)")
    s1, s2 = st.columns(2)
    horizon = s1.slider("Horizon (years)", 1, 10, 5)
    start_val = s2.number_input("Starting value ($)", 1000, 10 ** 9, 10000, 1000)
    rng = np.random.default_rng(0)
    paths = start_val * np.cumprod(1 + pr[rng.integers(0, len(pr), (1000, horizon * DAYS))], axis=1)
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(1, horizon * DAYS + 1) / DAYS
    fan = go.Figure()
    for lo_i, hi_i, name in [(0, 4, "5-95%"), (1, 3, "25-75%")]:
        fan.add_scatter(x=x, y=pct[hi_i], line_width=0, showlegend=False)
        fan.add_scatter(x=x, y=pct[lo_i], fill="tonexty", line_width=0, name=name)
    fan.add_scatter(x=x, y=pct[2], name="Median")
    fan.update_layout(xaxis_title="Years", yaxis_title="Portfolio value ($)", height=450)
    st.plotly_chart(fan, use_container_width=True)
    st.write(f"After {horizon}y: median **\\${pct[2, -1]:,.0f}**, 5th pct \\${pct[0, -1]:,.0f}, "
             f"95th pct \\${pct[4, -1]:,.0f}. Chance of ending below the start: **{(paths[:, -1] < start_val).mean():.0%}**.")
    st.caption("Resamples past returns, so it can't produce a market regime that never happened. Not a forecast.")

st.caption("Everything here comes from past prices. It is a study tool, not investment advice.")
