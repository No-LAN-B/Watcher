import os
import pandas as pd

def compute_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Given a raw price DataFrame with columns [Open, High, Low, Close, Adj Close, Volume],
    returns a new DataFrame with:
      - return:     (Close_t / Close_{t-1}) - 1
      - sma:        rolling mean of Close over 'window' days
      - vol:        rolling std of return over 'window' days
    """
    features = pd.DataFrame(index=df.index)  # start empty, keep same dates

    # 1. Daily return
    # (Close_t / Close_{t-1}) - 1
    features['return'] = df['Close'].pct_change()

    # 2. Simple Moving Average of Close
    # rolling(window).mean()
    features[f'sma_{window}'] = df['Close'].rolling(window).mean()

    # 3. Volatility: rolling std of returns
    features[f'vol_{window}'] = features['return'].rolling(window).std()

    return features

def main():
    os.makedirs("data/processed", exist_ok=True)

    # process each ticker CSV
    raw_folder = "data/raw"
    for fname in os.listdir(raw_folder):
        if not fname.endswith(".csv"):
            continue

        ticker = fname[:-4]  # strip ".csv"
        df = pd.read_csv(os.path.join(raw_folder, fname), index_col="Date", parse_dates=True)

        feats = compute_features(df, window=10)
        out_path = os.path.join("data/processed", f"{ticker}_features.csv")
        feats.to_csv(out_path)
        print(f"Processed {ticker}: {len(feats)} rows → saved to {out_path}")

if __name__ == "__main__":
    main()
