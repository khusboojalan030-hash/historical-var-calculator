# Historical VaR / CVaR Calculator

A small project comparing Historical Simulation VaR against Parametric
VaR on a 5-stock portfolio, using yfinance for price data.

I built this while going through the market risk section of FRM Part 1 -
wanted to actually implement VaR and Expected Shortfall instead of just
reading the formulas off a slide.

## What it does

- Pulls 5 years of daily prices for 5 stocks (yfinance)
- Builds daily returns and combines them into one weighted portfolio return series
- Calculates Historical VaR (percentile of actual past returns)
- Calculates CVaR / Expected Shortfall (average of the returns worse than VaR)
- Calculates Parametric VaR (assumes returns are normal, uses mean + std dev)
- Compares the two and explains the gap using skew/kurtosis
- Does all of the above at 90%, 95%, and 99% confidence
- Saves a chart (var_distribution.png) and a results table (var_results.csv)

## Why Historical and Parametric VaR don't match

Parametric VaR assumes returns follow a normal distribution - it only
needs the mean and standard deviation. Historical VaR skips that
assumption and just reads straight off the actual data.

Real stock returns almost never look perfectly normal though. They
usually have fatter tails, meaning big up/down days happen more often
than a bell curve says they should. The script checks this with
skewness and excess kurtosis:

- Skew < 0 → longer left tail, i.e. losses tend to be bigger/more
  extreme than gains
- Excess kurtosis > 0 → fat tails, extreme days happen more than
  "normal" predicts

When kurtosis is positive (pretty much always true for real stocks),
Historical VaR usually comes out worse than Parametric VaR - and the
gap gets bigger as you go from 90% to 99% confidence, since that's
where the fat tails actually show up.

## How to run it

```
pip install yfinance pandas numpy scipy matplotlib
python var_calculator.py
```

Change the tickers/weights at the top of the script if you want to test
a different portfolio:

```python
tickers = ["AAPL", "MSFT", "AMZN", "JPM", "XOM"]
weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # has to add up to 1
years = 5
```

## Sample output

```
conf    hist VaR    hist CVaR   param VaR   gap
90%     -1.51%      -2.43%      -1.75%      0.23%
95%     -2.11%      -3.05%      -2.24%      0.13%
99%     -3.51%      -4.67%      -3.16%      -0.35%
```
(numbers will be different when you run it since it pulls live data)

## Things I'd add later

- Monte Carlo VaR as a third comparison method
- Backtesting VaR against actual P&L (Kupiec test)
- Multi-day VaR using square-root-of-time scaling
- GARCH volatility instead of a flat std dev for the parametric method
