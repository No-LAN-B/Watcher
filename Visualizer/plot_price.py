import pandas as pd
import matplotlib.pyplot as plt

# 1) Load your CSV into a DataFrame
#    (adjust the path to match where your raw CSV lives)
df = pd.read_csv("data/raw/AAPL.csv", index_col="Date", parse_dates=True)

# 2) Compute a moving average (if you haven’t already):
df["sma_20"] = df["Close"].rolling(20).mean()

# 3) Plot Close price
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Close"], label="Close Price")

# 4) Overlay the moving average
plt.plot(df.index, df["sma_20"], label="20‑day SMA")

# 5) Polish the chart
plt.title("AAPL Close Price & 20‑day SMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)

# 6) Show or save
plt.tight_layout()
plt.show()
# plt.savefig("charts/aapl_price_sma.png")
