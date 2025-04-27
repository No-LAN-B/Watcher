import os, sys

# Note: This project is partially a learning exercise so commenting why / how something works will be very common as I use those comments to learn and remember. 

sys.path.append(
    os.path.abspath(
      os.path.join(os.path.dirname(__file__), "..")
    )
)

from data.provider import YFinanceProvider  # or IEXCloudProvider
import os

def main():
    tickers = ["AAPL", "MSFT", "GOOG"]
    provider = YFinanceProvider()
    raw_data = provider.fetch(tickers, "2015-01-01", "2025-05-01")

    os.makedirs("data/raw", exist_ok=True)
    for ticker, df in raw_data.items():
        df.to_csv(f"data/raw/{ticker}.csv", index_label="Date")
        print(f"Saved {ticker}.csv ({len(df)} rows)")

if __name__ == "__main__":
    main()
