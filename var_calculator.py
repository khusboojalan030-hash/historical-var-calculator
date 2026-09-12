"""
Historical Simulation VaR Calculator
-------------------------------------
Quick project to compare Historical VaR/CVaR against Parametric VaR
on a small 5-stock portfolio, using yfinance for price data.

I built this while studying for FRM Part 1 - VaR and Expected Shortfall
come up a lot in the market risk section, so I wanted to actually code
it out instead of just reading formulas.

Confidence levels: 90%, 95%, 99%
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

# ----- portfolio setup -----
# change these if you want to try a different set of stocks
tickers = ["AAPL", "MSFT", "AMZN", "JPM", "XOM"]
weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # has to add up to 1
years = 5
portfolio_value = 1_000_000  # just an assumed value so we can show $ figures too
confidence_levels = [0.90, 0.95, 0.99]

if abs(sum(weights) - 1) > 1e-6:
    raise ValueError("weights don't add up to 1, fix that first")


def get_prices(tickers, years):
    print(f"pulling {years}y of data for {tickers}...")
    data = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    data = data.dropna()
    print(f"got {len(data)} days of data, {data.index[0].date()} -> {data.index[-1].date()}")
    return data


def get_portfolio_returns(prices, weights):
    # simple daily % returns per stock
    rets = prices.pct_change().dropna()
    # weighted sum across stocks = portfolio return for each day
    port_rets = rets.dot(weights)
    return rets, port_rets


def hist_var(returns, conf):
    # historical VaR = just the percentile of actual past returns
    # e.g. 95% VaR is the 5th percentile (worst 5% of days)
    return np.percentile(returns, (1 - conf) * 100)


def hist_cvar(returns, conf):
    # CVaR = average of everything worse than the VaR cutoff
    # (this is basically "ok it's a bad day, how bad on average")
    cutoff = hist_var(returns, conf)
    tail = returns[returns <= cutoff]
    return tail.mean()


def parametric_var(returns, conf):
    # assumes returns are normal, uses mean/std + z-score
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - conf)
    return mu + z * sigma


def main():
    prices = get_prices(tickers, years)
    daily_returns, port_returns = get_portfolio_returns(prices, weights)

    print("\n--- portfolio stats ---")
    for t, w in zip(tickers, weights):
        print(f"{t}: {w*100:.0f}%")
    print(f"mean daily return: {port_returns.mean():.4%}")
    print(f"daily volatility:  {port_returns.std():.4%}")

    skew = stats.skew(port_returns)
    kurt = stats.kurtosis(port_returns)
    print(f"skew: {skew:.3f}, excess kurtosis: {kurt:.3f}")

    print("\n--- VaR / CVaR results ---")
    print(f"{'conf':<8}{'hist VaR':<12}{'hist CVaR':<12}{'param VaR':<12}{'gap':<10}")

    rows = []
    for c in confidence_levels:
        hv = hist_var(port_returns, c)
        hc = hist_cvar(port_returns, c)
        pv = parametric_var(port_returns, c)
        gap = hv - pv
        rows.append([c, hv, hc, pv, gap])
        print(f"{c:.0%}     {hv:<12.4%}{hc:<12.4%}{pv:<12.4%}{gap:<10.4%}")

    print("\n--- $ figures (on assumed $1M portfolio) ---")
    for c, hv, hc, pv, gap in rows:
        print(f"{c:.0%}: hist VaR = ${-hv*portfolio_value:,.0f}, "
              f"hist CVaR = ${-hc*portfolio_value:,.0f}, "
              f"param VaR = ${-pv*portfolio_value:,.0f}")

    # why do these two methods disagree?
    print("\n--- historical vs parametric ---")
    print("Parametric VaR assumes returns are normally distributed and only")
    print("uses mean + std dev. Historical VaR just reads straight off the")
    print("actual data, no assumptions.")
    print()
    if kurt > 0:
        print(f"Excess kurtosis here is {kurt:.3f}, which is positive - meaning")
        print("this portfolio has fatter tails than a normal distribution would")
        print("predict (extreme days happen more often than 'normal' assumes).")
        print("That's why Historical VaR usually ends up worse than Parametric")
        print("VaR, especially at 99% confidence where the tail matters most -")
        print("the normal curve just doesn't capture those extreme moves.")
    else:
        print("Excess kurtosis is close to zero here so the normal assumption")
        print("isn't too far off - the two VaR numbers should be fairly close.")

    # save a chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(port_returns, bins=80, density=True, alpha=0.6, color="steelblue", label="actual returns")
    x = np.linspace(port_returns.min(), port_returns.max(), 500)
    ax.plot(x, stats.norm.pdf(x, port_returns.mean(), port_returns.std()), color="black", label="normal fit")
    for c, hv, *_ in rows:
        ax.axvline(hv, linestyle="--", label=f"{c:.0%} hist VaR")
    ax.set_title("Portfolio Returns vs Normal Distribution")
    ax.set_xlabel("daily return")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("var_distribution.png", dpi=150)

    pd.DataFrame(rows, columns=["confidence", "hist_var", "hist_cvar", "param_var", "gap"]).to_csv("var_results.csv", index=False)
    print("\nsaved var_distribution.png and var_results.csv")


if __name__ == "__main__":
    main()
