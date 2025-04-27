import pandas as pd
import mplfinance as mpf

# 1) Load data
df = pd.read_csv("data/raw/AAPL.csv", index_col="Date", parse_dates=True)

# 2) Plot candlestick with volume underneath
mpf.plot(
    df,
    type="candle",
    volume=True,
    mav=(20,50),            # show 20‑ & 50‑day moving averages
    title="AAPL Candlestick + Volume",
    style="yahoo",          # one of the built‑in styles
    savefig="charts/aapl_candle.png"
)
