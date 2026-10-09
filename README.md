# Portfolio Optimization & Simulation

Originally a notebook study of 11 stocks (2000-2025: Monte Carlo portfolios, efficient frontier, XGBoost / Random Forest
return prediction). It is now also a Streamlit web app for building and optimizing a multi-asset portfolio.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- **Universe:** search Yahoo Finance for any stock, ETF, bond fund or crypto, or add a ready-made pack.
- **Optimize:** max-Sharpe and min-volatility portfolios, exact efficient frontier, per-asset min/max limits,
  Ledoit-Wolf covariance, optional return shrinkage.
- **My portfolio:** enter an amount to invest, or what you already hold (plus optional extra cash), and get the
  recommended distribution, dollar trades and share counts (optionally whole shares).
- **Backtest:** walk-forward rebalancing with transaction costs, versus equal-weight. Out-of-sample, no look-ahead.
- **Forecast:** standard normal-return (geometric Brownian motion) projection of portfolio value with confidence
  bands, plus a reality check of the method on the last year.
- **Risk & simulation:** VaR / CVaR, drawdown, bootstrap forward simulation.

All inputs are historical estimates. This is a study tool, not investment advice.

## Key finding

In-sample, max-Sharpe beats equal-weight comfortably. Out-of-sample (walk-forward) the edge shrinks a lot, and
shrinking expected returns toward the average made it worse on the tested universe. Treat any new signal as
unproven until it beats equal-weight in the Backtest tab.

## Roadmap: market-sentiment-aware optimization

Not built yet. The app is structured so it can be added without a rewrite.

**Idea:** turn sentiment (news, filings, social media, analyst revisions, volatility/fear indices) into forward-looking
views on asset returns or risk, and let the optimizer use them, so allocations shift as sentiment shifts.

**Where it plugs in**

| Piece | Where | Notes |
|---|---|---|
| Signal -> expected-return adjustment | `estimate(..., mu_tilt=...)` in `app.py` | Per-asset annual return tilt added to the historical estimate. A sentiment model only has to output this vector. Black-Litterman style views are the principled alternative to a raw tilt. |
| Sentiment affecting risk | same function, scale `cov` | e.g. raise volatility estimates in fear regimes. |
| Regime shifts | new input to `optimise` limits (`lo`, `hi`) | e.g. cap equity weight when a risk-off regime is detected. |
| Validation | existing Backtest tab | Call `estimate` with the tilt available *at each rebalance date*. |
| UI | sidebar "Market-sentiment tilt (planned)" checkbox | Currently disabled. |

**Suggested steps**

1. Pick a data source with *point-in-time* history (news / social / analyst data with original timestamps).
   Backtests are invalid if the sentiment shown for a date was revised or scraped after that date.
2. Build a per-asset sentiment score with a documented method (e.g. a finance-tuned language model such as FinBERT
   over headlines, or a fear/greed index for broad regimes). Aggregate to daily or weekly.
3. Test whether the score predicts next-period returns *before* using it (rank correlation / information
   coefficient, out-of-sample). The notebooks already showed daily return prediction from price features has R^2 near 0,
   so expect weak signals and test honestly.
4. Map the score to `mu_tilt` with a small, capped scale so one noisy signal cannot dominate the optimizer.
5. Backtest with and without the tilt, including transaction costs. Ship only if it beats the no-tilt version
   out-of-sample.
6. Optional: regime detection (e.g. hidden Markov model on volatility and sentiment) to switch constraints.

**Likely additions:** a `sentiment.py` module (data fetch + scoring + caching), API keys via environment variables /
Streamlit secrets, and a sentiment tab showing scores and their effect on weights.
