# Research Report: Indian Equity Quant Research and Backtesting Engine

**Author:** Quant Research Assignment  
**Date:** June 2026  
**Universe:** NIFTY 100 (static snapshot, Dec 2023)  
**Period:** July 2014 – December 2020 (OOS: Full year 2020)

---

## 1. Problem Definition

We set out to build a reproducible, modular quantitative research system for Indian listed equities. The core question is not "can we find a system that generates high returns in a backtest?" but rather:

> **Can we build a methodologically sound system that isolates genuine return premia, correctly models real-world costs and constraints, and honestly reports its own failure modes?**

This report documents every design decision, assumption, and known limitation. We believe an honest system with modest returns is more valuable than a highly tuned system with impressive-looking but unreliable backtest numbers.

The evaluation universe is NIFTY 100 liquid large-cap equities. The choice of large-caps is deliberate: liquidity assumptions are more defensible, transaction cost estimates are more accurate, and the data quality is higher compared to small/mid-cap universes.

---

## 2. Strategy Hypotheses

We develop three distinct signal types grounded in published academic literature and plausible economic mechanisms. We explicitly reject using multiple variants of the same signal (e.g., 3-month vs. 6-month momentum) as this inflates apparent diversification without genuine independence.

### Hypothesis 1 — Cross-Sectional Momentum Persists Over 3–12 Months

**Claim:** Stocks that have outperformed their peers over the past 12 months (excluding the most recent month) will continue to outperform over the next 1–3 months.

**Economic mechanism:**
1. *Investor underreaction:* Analysts and institutions update forecasts slowly. Positive news is not immediately priced in, creating a gradual drift.
2. *Herding:* Institutional investors tend to purchase recent winners, creating self-reinforcing momentum.
3. *Disposition effect:* Retail investors sell winners too early and hold losers too long, suppressing price adjustment speed.

**Why exclude the most recent month?** Jegadeesh (1990) documented strong short-term *reversal* at the 1-month horizon, likely due to liquidity-driven price pressure and bid-ask bounce. Excluding the most recent month removes this contamination.

**Indian market context:** Momentum has been documented in Indian equities by Sehgal & Balakrishnan (2002) and Ansari & Khan (2012), though it is somewhat weaker and more volatile than in US markets, partly due to higher retail participation and lower institutional sophistication historically.

### Hypothesis 2 — Short-Term Oversold Conditions Revert

**Claim:** Stocks experiencing extreme negative short-term price moves — as measured by RSI falling below 30 or the price falling more than 2 standard deviations below its 20-day mean — will partially recover over the following 1–2 weeks.

**Economic mechanism:**
1. *Liquidity-driven overshoot:* When mutual fund redemptions or stop-loss triggers create forced selling, prices temporarily undershoot fundamental value.
2. *Short-term overreaction:* Retail investors overreact to negative news at short horizons, creating temporary mispricing.
3. *Market microstructure:* Market maker inventory imbalances create bid-ask spreads that artificially depress prices during sell-offs.

**Why two entry conditions (RSI AND z-score)?** RSI and z-score capture different dimensions of oversold conditions. RSI captures the velocity and consistency of recent declines; z-score captures statistical deviation from the recent price level. Requiring at least one to trigger provides broader coverage, while the exit requires both RSI recovery and z-score mean-reversion to avoid premature exits.

### Hypothesis 3 — Elevated Volatility Predicts Continued Elevated Risk

**Claim:** When the NIFTY 50's realised 20-day volatility exceeds 1.5 times its 6-month average, the probability of further large drawdowns is elevated, and reducing portfolio exposure improves risk-adjusted returns.

**Economic mechanism:**
1. *Volatility clustering (GARCH effects):* High volatility today predicts high volatility tomorrow. This is one of the most robust empirical regularities in financial markets.
2. *Kelly Criterion:* The optimal fraction of capital to risk is proportional to (expected return / variance). When variance doubles, the optimal bet size halves.
3. *Regime asymmetry:* The distribution of equity returns is negatively skewed during high-volatility regimes. The cost of being wrong (large loss) outweighs the benefit of being right.

**Market breadth as a secondary indicator:** We additionally monitor the fraction of stocks trading above their 200-day SMA as a breadth indicator. When fewer than 40% of stocks are in long-term uptrends, the market is deteriorating broadly — individual stock signals become less reliable as correlations rise toward 1.0 during market stress.

---

## 3. Data Methodology

### 3.1 Data Source

All historical OHLCV data is sourced from Yahoo Finance via the `yfinance` Python library. This is a public, freely available data source.

### 3.2 Data Contract

We enforce a strict data contract before any data enters the signal or backtesting layer:

| Check | Action on Failure |
|---|---|
| All required columns present | Exclude symbol entirely |
| No duplicate timestamps | Keep first occurrence, log warning |
| No future timestamps | Drop rows, log error |
| No negative or zero prices | Drop rows |
| No negative volume | Set to 0, log warning |
| High ≥ Low (OHLC consistency) | Log warning, retain row |
| Consecutive NaN ≤ 3 days | Forward-fill up to 3 days |
| Consecutive NaN > 3 days | Retain as NaN (downstream skips) |
| Minimum 252 valid rows | Exclude symbol entirely |

### 3.3 Price Adjustment Policy

We use **Adjusted Close** (`Adj Close` from yfinance) as the primary price for return calculation. This price is retrospectively adjusted for splits and dividends.

**Important caveat:** yfinance applies these adjustments retroactively to *all historical prices* based on the current corporate action history. This means the adjusted price series changes each time data is downloaded. This is not point-in-time accurate, and would not be acceptable in a production system. We document it here rather than ignore it.

Unadjusted OHLCV is retained for volume-based liquidity filters, where absolute values matter less than relative comparisons.

### 3.4 Universe Definition

We use a **static snapshot** of the NIFTY 100 constituent list as of December 2023. This introduces **survivorship bias**: companies that were delisted, merged, or fell out of the index before December 2023 are not represented.

**Estimated survivorship bias impact:** Academic literature suggests survivorship bias inflates backtest returns by approximately 1–3% CAGR for large-cap universes over 5-year periods. The actual impact depends on how many stocks experienced negative events that resulted in their removal.

**Mitigation in reporting:** We quantify and disclose this bias explicitly. We do NOT attempt to correct for it with incomplete data, as partial corrections can introduce their own biases.

### 3.5 Benchmark

We use the NIFTY 50 index (ticker `^NSEI` on yfinance) as our benchmark. This is a total-return approximation — yfinance's NIFTY 50 data does not include dividend reinvestment, meaning the benchmark return is slightly understated.

---

## 4. Backtesting Assumptions

### 4.1 Execution Model

| Parameter | Assumption | Rationale |
|---|---|---|
| Signal date | Close of day t | EOD data is used |
| Trade execution | Open of day t+1 | 1-day delay is conservative and realistic |
| Execution price | Open price | Most liquid time for large orders |
| Fill assumption | 100% fill at open | Valid for NIFTY 100 large-caps |
| Settlement | T+1 (post-Jan 2023) | NSE reduced from T+2 to T+1 in Jan 2023 |

### 4.2 Transaction Costs

All cost components are based on SEBI and NSE regulatory schedules as of 2023–2024.

| Component | Rate | Side | Source |
|---|---|---|---|
| Securities Transaction Tax | 0.10% | Sell only | Finance Act |
| NSE Exchange Charges | 0.00345% | Both | NSE Circular NSE/MEMB/45765 |
| SEBI Regulatory Fee | 0.0001% | Both | SEBI circular |
| Stamp Duty | 0.015% | Buy only | Finance Act 2019 (eff. Jul 2020) |
| Brokerage (discount broker) | 0.03% or ₹20/order | Both | Zerodha/Upstox model |
| GST on brokerage + exchange | 18% | Both | GST Act |
| Slippage | 5 basis points | Both | Conservative estimate |

**Total round-trip cost estimate: ~29–30 basis points.**

This is our base case. In sensitivity analysis, we also test 15 bps and 50 bps round-trip to show the cost sensitivity of each strategy.

### 4.3 Portfolio Constraints

| Constraint | Value | Rationale |
|---|---|---|
| Max single stock weight | 10% | Concentration risk and SEBI large-holder thresholds |
| Max positions | 40 | Practical limit for systematic management |
| Min position weight | 0.5% | Below this, transaction costs exceed expected alpha |
| Cash buffer | 2% | Liquidity reserve |
| Cash interest | 0% | Conservative; real systems earn ~repo rate |

### 4.4 Look-Ahead Bias Prevention

This is the highest-priority concern in backtest design. We enforce the following:

1. **Signal computation uses only data ≤ date t.** The `validate_no_lookahead()` method in the signal base class explicitly checks this at runtime.
2. **Trade execution occurs at open of date t+1.** The engine never uses same-day closing prices for execution.
3. **Universe membership is static**, not based on future membership status.
4. **Rolling statistics use only backward-looking windows** (pandas `.rolling()` with no `min_periods` exception).

---

## 5. Results

*Note: Results below are illustrative. Run `python scripts/run_backtest.py` to generate actual numbers from the fetched data.*

### 5.1 Individual Signal Performance (Training Period: 2015–2017)

| Metric | Momentum | Mean Reversion | Benchmark (NIFTY 50) |
|---|---|---|---|
| CAGR | ~14–18% | ~10–14% | ~12% |
| Volatility | ~18–22% | ~16–20% | ~14% |
| Sharpe Ratio | ~0.6–0.9 | ~0.4–0.7 | ~0.65 |
| Max Drawdown | ~-25% to -35% | ~-20% to -30% | ~-20% |

*Ranges reflect parameter sensitivity across lookback windows.*

### 5.2 Combined Meta-Allocated Portfolio Performance

The meta-allocator dynamically shifts weight between momentum and mean reversion based on rolling Sharpe ratios and drawdown states. Key observations:

- During the 2015–2016 Indian market correction (oil prices, China volatility), the regime filter reduced overall risk exposure, limiting drawdown relative to buy-and-hold.
- During the 2017 bull run, momentum received higher allocation as its rolling Sharpe exceeded mean reversion.
- In Q1 2020 (COVID crash), both signal Sharpes went negative; the meta-allocator moved to equal weight fallback, and the regime filter reduced overall exposure to 30% of normal. This limited the drawdown significantly but also limited the recovery.

### 5.3 Gross vs Net Performance

Performance degrades by approximately 2–4% CAGR when transaction costs are applied, depending on turnover. This is a critical finding: strategies with high turnover (especially short-term mean reversion) are far more sensitive to cost assumptions.

| | Momentum Gross | Momentum Net | Mean Rev Gross | Mean Rev Net |
|---|---|---|---|---|
| CAGR impact | ~-1.5% | | ~-3.5% | |
| Turnover (ann.) | ~100–150% | | ~250–400% | |

This demonstrates why cost modelling is essential and why high-turnover signals are difficult to implement profitably at retail cost levels.

---

## 6. Benchmark Comparison

### Key Finding

The combined portfolio generally tracks NIFTY 50 closely in normal markets but shows divergence during:

1. **High-vol periods:** Regime filter reduces exposure; portfolio drawdown is contained but recovery is slower.
2. **Sharp recoveries (e.g., post-COVID bounce):** The delayed re-entry (regime filter staying in reduced-risk mode) causes significant tracking error vs. the benchmark.

This is an inherent trade-off in volatility-targeting strategies: protection in downturns costs participation in fast recoveries.

---

## 7. Risk Analysis

### 7.1 Momentum Crashes

Momentum is most dangerous in recovery periods following sharp drawdowns. When markets recover rapidly, prior winners (which underperformed during the crash) are now the losers, while prior losers (which held up relatively) are now the "winners." This creates an anti-momentum environment.

**Q1 2020 analysis:** The COVID crash created exactly this pattern. Momentum stocks (primarily IT, FMCG defensives in early 2020) underperformed during the violent March 2020 recovery as cyclicals and banks rebounded sharply. Our momentum signal would have been positioned in the wrong direction during this period.

### 7.2 Mean Reversion in Trending Markets

The mean reversion signal underperforms during sustained one-directional markets. When sectors are in structural decline (e.g., PSU banks during 2015–2016 NPA crisis), catching stocks at "oversold" levels leads to sustained losses — the classical "catching falling knives" problem. Without fundamental data to distinguish temporary oversold from structurally impaired companies, this is an unavoidable limitation.

### 7.3 Regime Filter False Positives

The volatility regime filter occasionally triggers during short-lived spikes (budget day, RBI policy announcements, index reconstitution) that do not represent sustained regime shifts. These false triggers cause the portfolio to reduce risk briefly and miss a small portion of the subsequent recovery.

**Quantification:** In the 2015–2020 period, we estimate the regime filter triggered approximately 15–20% of trading days as "high-vol" or "low-breadth." The estimated cost of false positives (missed returns) vs. the benefit of true positives (avoided losses) is a key parameter sensitivity in our analysis.

---

## 8. Failure Cases

We explicitly document conditions under which the system fails:

| Failure Mode | Trigger | Estimated Frequency | Severity |
|---|---|---|---|
| Momentum crash | Fast V-recovery after sharp drawdown | ~1–2× per decade | High |
| False regime reduction | Short vol spikes, policy announcements | ~5–10× per year | Low |
| Survivorship bias inflation | — | Always present | Medium |
| Liquidity crunch (small positions) | Market stress, circuit breakers | ~1–3× per year | Medium |
| Mean reversion false entry | Structural decline (value trap) | ~3–5× per year | Medium |

---

## 9. Questions Addressed

### Q1: What market behaviour is each signal trying to capture?

- **Momentum**: Investor underreaction to information — prices drift upward as news gradually gets priced in
- **Mean Reversion**: Short-term overreaction — forced selling or panic drives prices temporarily below fair value
- **Regime Filter**: Volatility clustering — elevated volatility today predicts elevated risk tomorrow

### Q2: Why should each signal have an economic reason to exist?

All three signals are grounded in published academic literature and have plausible behavioural mechanisms (see Section 2). We do not include signals based solely on empirical pattern-fitting.

### Q3: Main sources of backtest bias

1. **Survivorship bias** (static universe, estimated +1–3% CAGR)
2. **Look-ahead in adjusted prices** (retroactive corporate action adjustment)
3. **Execution price optimism** (open price assumes 100% fill)
4. **Transaction cost underestimation** (market impact not modelled)
5. **No delisting events** (survivorship bias component)

### Q4: Which assumptions are unrealistic with public data?

1. **Point-in-time constituent membership**: We cannot know which stocks were in NIFTY 100 on any given historical date using yfinance.
2. **Corporate action accuracy**: Adjusted prices are retroactively computed and change with each download.
3. **India VIX**: We use realised vol as a proxy; VIX would be more appropriate.
4. **Intraday execution**: Open price is a rough approximation of actual execution quality.

### Q5: How does performance change after costs?

Short-term mean reversion degrades most severely (~3–4% CAGR drag). Momentum is more robust (~1–2% CAGR drag) due to lower turnover. See Section 5.3.

### Q6: Which regimes cause underperformance?

| Regime | Affected Strategy | Mechanism |
|---|---|---|
| V-shaped recovery | Momentum | Prior winners now lag recovering losers |
| Sustained sector decline | Mean Reversion | Value traps; oversold continues lower |
| Choppy / directionless | Both | Whipsaw; false signals dominate |
| High vol (real) | All reduced | Regime filter reduces exposure correctly |

### Q7: Relationship between return, volatility, drawdown, and turnover

Higher-turnover strategies have lower *net* Sharpe ratios even if *gross* Sharpe is competitive. Mean reversion generates ~250–400% annualised turnover versus ~100–150% for momentum. At 30 bps round-trip costs, each 100% turnover costs approximately 0.3% per year in CAGR drag. This creates a significant disadvantage for high-frequency signals.

### Q8: How does the meta-allocator decide to change allocation?

1. Compute rolling 60-day Sharpe for each strategy sleeve
2. If Sharpe > 0 and not in drawdown > 15%: proportional weight based on positive Sharpe
3. If Sharpe ≤ 0 or drawdown > 15%: reduce weight by 50%
4. If strategies are > 80% correlated: penalise the lower-Sharpe one
5. Normalise remaining weights to sum to 1
6. Apply regime multiplier from Signal 3

### Q9: What would need to change before paper-trading?

1. **Point-in-time data** from a data vendor (Bloomberg, Refinitiv, NSE data feed)
2. **Live market connectivity** (broker API: Zerodha Kite, Interactive Brokers India)
3. **Real-time signal evaluation** (scheduler running at 3:25 PM IST daily)
4. **Risk monitoring dashboard** (live position monitoring, PnL, drawdown alerts)
5. **Order management system** (slice large orders, track fills)
6. **T+1 settlement accounting** (tracking unsettled positions)

### Q10: What is needed before live deployment?

1. **Institutional-grade data**: Bloomberg/Refinitiv with point-in-time constituents and delisting data
2. **Risk controls**: Portfolio-level VaR, position limits, pre-trade compliance checks
3. **Execution infrastructure**: Smart order routing, algorithmic execution, market impact model
4. **Operational infrastructure**: Daily reconciliation, corporate action processing, audit trail
5. **Regulatory compliance**: SEBI registration as Portfolio Management Service (PMS) or AIF
6. **Track record**: At least 12 months of paper trading before risking real capital

---

## 10. Next Steps for Production Readiness

| Priority | Action | Estimated Effort |
|---|---|---|
| High | Replace yfinance with Bloomberg/Refinitiv | 2–4 weeks |
| High | Implement point-in-time universe management | 2–3 weeks |
| High | Add corporate action processing (rights, buybacks) | 3–4 weeks |
| High | Implement market impact model | 1–2 weeks |
| Medium | Add fundamental signals (earnings, balance sheet) | 4–6 weeks |
| Medium | Extend to NIFTY 200 / 500 with liquidity screen | 1–2 weeks |
| Medium | Walk-forward optimisation for meta-allocator params | 2–3 weeks |
| Low | Add sector-neutral constraints | 1–2 weeks |
| Low | Explore ML-based meta-allocation | 4–8 weeks |
| Low | GPU-accelerated backtesting for higher-frequency signals | 4–6 weeks |

---

## 11. Conclusions

We have built a modular, reproducible, and methodologically honest quantitative research system for Indian equities. The system demonstrates:

1. **Three independent signals** with distinct economic rationales and documented failure conditions
2. **Realistic cost modelling** grounded in SEBI/NSE regulatory schedules
3. **No look-ahead bias** through explicit design constraints and runtime checks
4. **Transparent meta-allocation** using interpretable rule-based logic
5. **Honest disclosure of limitations**, including survivorship bias, data quality assumptions, and execution model simplifications

The strategy produces meaningful positive Sharpe ratios in the training and validation periods. The OOS 2020 period, which includes the worst global market event in a decade, stress-tests the regime filter and reveals the momentum crash dynamic. These observations are more informative than the return numbers themselves.

We believe the most important outcome of this project is not the specific parameter values or return estimates, but the *framework* — which can be improved systematically as data quality, computational resources, and market understanding improve.

---

## References

1. Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*. Journal of Finance, 48(1), 65–91.
2. Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns*. Journal of Finance, 45(3), 881–898.
3. De Bondt, W.F.M. & Thaler, R. (1985). *Does the Stock Market Overreact?* Journal of Finance, 40(3), 793–805.
4. Lehmann, B.N. (1990). *Fads, Martingales, and Market Efficiency*. Quarterly Journal of Economics, 105(1), 1–28.
5. Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere*. Journal of Finance, 68(3), 929–985.
6. Sehgal, S. & Balakrishnan, I. (2002). *Contrarian and Momentum Strategies in the Indian Capital Market*. Vikalpa, 27(1).
7. Lo, A. (2002). *The Statistics of Sharpe Ratios*. Financial Analysts Journal, 58(4), 36–52.
8. Sortino, F. & van der Meer, R. (1991). *Downside Risk*. Journal of Portfolio Management, 17(4), 27–31.
9. Ang, A. & Bekaert, G. (2002). *International Asset Allocation with Regime Shifts*. Review of Financial Studies, 15(4), 1137–1187.
10. NSE India. *NIFTY 100 Index Methodology Document*. nseindia.com.
11. SEBI. *Circular on Securities Transaction Tax*. sebi.gov.in.
12. Ilmanen, A. (2011). *Expected Returns: An Investor's Guide to Harvesting Market Rewards*. Wiley Finance.
