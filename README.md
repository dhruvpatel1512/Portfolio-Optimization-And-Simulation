# Portfolio Optimization & Simulation

This started as a notebook study of 11 stocks from 2000 to 2025: random Monte Carlo portfolios, an efficient frontier,
and XGBoost and Random Forest models that tried to predict returns. It is now also a web app where you choose your own
stocks, ETFs, bond funds or crypto and get a suggested split, a backtest and a forecast.

**Live app:** https://portfolio-optimization-and-simulation-tool.streamlit.app/

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- **Universe.** Search Yahoo Finance for any stock, ETF, bond fund or crypto, or add a ready-made pack such as the
  original 11 stocks or a set of bond ETFs.
- **Optimize.** Finds the max-Sharpe and min-volatility portfolios and draws the exact efficient frontier. You can set a
  minimum and maximum weight for each asset. Covariance uses Ledoit-Wolf shrinkage, and expected returns can
  optionally be shrunk toward the average.
- **My portfolio.** Enter an amount to invest, or what you already hold plus any extra cash. The app shows the target
  split, the dollar trades and the share counts, and can round to whole shares.
- **Backtest.** Re-optimizes monthly or quarterly using only the data available at that date, charges transaction
  costs, and compares the result with equal-weight.
- **Forecast.** Projects portfolio value assuming normally distributed returns, with 5-95% bands. A check on the last
  year shows how often the real path stayed inside the band.
- **Risk & simulation.** Daily VaR and CVaR, drawdown, and a bootstrap simulation that resamples past returns.

Everything comes from past prices. It is a study tool, not investment advice.

## What the backtest showed

In-sample, max-Sharpe beat equal-weight clearly: 1.08 against 0.75 on the original 11 stocks since 2010. Out-of-sample,
with a 3-year lookback and monthly rebalancing, the gap fell to 0.92 against 0.84. Shrinking expected returns toward
the average made max-Sharpe worse on a stocks, bonds and gold test (0.69 against 0.94 without it), so it is off by
default. Equal-weight is a hard benchmark, and any new signal should beat it in the Backtest tab before it is trusted.

## Roadmap: market-sentiment-aware optimization

Not built yet. The code leaves one place for it: `estimate()` in `app.py` takes an optional `mu_tilt`, a per-asset
annual return adjustment added to the historical estimate. A sentiment model only needs to produce that vector, and the
optimizer, frontier and backtest will use it.

| Piece | Where it goes |
|---|---|
| Sentiment as a return view | `mu_tilt` in `estimate()`. Black-Litterman views are a more principled alternative to a raw tilt. |
| Sentiment as a risk signal | Scale the covariance in the same function, for example higher volatility in fearful markets. |
| Regime shifts | Change the weight limits (`lo`, `hi`), for example cap equities when the regime turns risk-off. |
| Validation | The Backtest tab, fed the tilt that was known at each rebalance date. |
| Interface | The disabled "Market-sentiment tilt (planned)" checkbox in the sidebar. |

Steps, in order:

1. Find sentiment data with point-in-time history (news, social posts or analyst revisions with their original
   timestamps). A backtest is invalid if a date's sentiment was revised or collected after that date.
2. Score each asset. A finance-tuned language model such as FinBERT on headlines works for single names, and a
   fear/greed index for broad market regimes. Aggregate daily or weekly.
3. Test whether the score predicts next-period returns, out-of-sample, before the optimizer sees it. The notebooks
   found an R² near 0 for daily returns from price features, so expect weak signals.
4. Map the score to `mu_tilt` on a small, capped scale so one noisy signal can't dominate.
5. Backtest with and without the tilt, including costs, and keep it only if it wins out-of-sample.
6. Optionally detect regimes, for example with a hidden Markov model on volatility and sentiment, and use them to
   switch constraints.

Likely additions: a `sentiment.py` module for fetching, scoring and caching, API keys in Streamlit secrets, and a
sentiment tab showing the scores and how they moved the weights.
