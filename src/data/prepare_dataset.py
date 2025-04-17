# src/data/prepare_dataset.py

import os
import pandas as pd


def prepare_dataset(
    processed_folder: str = "data/processed",
    output_path: str = "data/processed/model_dataset.csv"
) -> None:
    """
    Combine per-ticker time-series features with graph metadata into one flat dataset.
    Saves the result to `output_path`.
    """
    # 1) Load all time-series feature files
    ts_frames = []
    for fname in os.listdir(processed_folder):
        if not fname.endswith("_features.csv"):
            continue
        # Extract ticker symbol from filename, e.g. 'AAPL' from 'AAPL_features.csv'
        ticker = fname.split("_")[0]
        # Read the features CSV, parsing 'Date' as the index
        df_ts = pd.read_csv(
            os.path.join(processed_folder, fname),
            index_col="Date",
            parse_dates=True
        )
        # Add a column for the ticker, so we can merge later
        df_ts["Ticker"] = ticker
        ts_frames.append(df_ts)

    # Concatenate all per-ticker DataFrames into one big DataFrame
    # Rows are stacked; index is Date, columns include dynamic features + Ticker
    all_ts = pd.concat(ts_frames, axis=0)

    # 2) Load graph metadata (static features) for each ticker
    graph_path = os.path.join(processed_folder, "graph_features.csv")
    df_meta = pd.read_csv(graph_path, index_col="Ticker")

    # 3) Merge dynamic and static features
    # Reset index to turn 'Date' into a column, then merge on 'Ticker'
    merged = (
        all_ts
        .reset_index()
        .merge(df_meta.reset_index(), on="Ticker", how="left")
    )

    # Optionally set a multi-index of (Date, Ticker)
    merged.set_index(["Date", "Ticker"], inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save to CSV for modeling
    merged.to_csv(output_path)
    print(f"Saved prepared dataset to {output_path}")


if __name__ == "__main__":
    prepare_dataset()
