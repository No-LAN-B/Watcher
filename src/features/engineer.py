import os
import pandas as pd

# Keep this function mostly as is, but maybe add more features here later
def compute_technical_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Calculates technical features based on the input DataFrame.
    Input should have at least 'Close' and 'Volume' columns.
    Returns a DataFrame with ONLY the calculated features.
    """
    # Ensure required columns exist
    if not all(col in df.columns for col in ['Close']):
         raise ValueError("Input DataFrame must contain 'Close' column.")
         
    features = pd.DataFrame(index=df.index) # Use the same index

    # 1. Daily return percentage change
    features['return'] = df['Close'].pct_change()

    # 2. Simple Moving Average of Close 
    features[f'sma_{window}'] = df['Close'].rolling(window).mean()

    # 3. Volatility: rolling std of returns
    # Ensure 'return' column exists before calculating vol
    if 'return' in features.columns:
         features[f'vol_{window}'] = features['return'].rolling(window).std()
    else:
         # Handle case where return couldn't be calculated (e.g., first row)
         # This might result in NaNs anyway, which is fine for now.
         features[f'vol_{window}'] = pd.Series(index=df.index, dtype=float)


    # 4. Lagged Close Price (Yesterday's Close) - Often useful!
    features['Close_Lag_1'] = df['Close'].shift(1)
    
    # --- ADD MORE FEATURES HERE ---
    # Example: RSI (requires a separate function or library like talib/pandas_ta)
    # if 'Close' in df.columns:
    #     from your_rsi_module import calculate_rsi # Assuming you have this
    #     features['RSI_14'] = calculate_rsi(df['Close'], window=14)

    # Example: MACD (requires a separate function or library)
    # if 'Close' in df.columns:
    #     from your_macd_module import calculate_macd # Assuming you have this
    #     macd_line, signal_line = calculate_macd(df['Close'])
    #     features['MACD_line'] = macd_line
    #     features['Signal_line'] = signal_line
    # -----------------------------

    return features

# --- MODIFIED Reusable Function ---
def engineer_features_for_stock(input_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Takes a raw DataFrame (Date index, OHLCV) for a single ticker,
    calculates features, and returns a new DataFrame containing
    both the essential original columns AND the calculated features.

    IMPORTANT: Does NOT drop NaNs here, as the calling function 
               needs to align target variable first.
    """
    # Select essential original columns needed for target creation or as direct features
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Adjust if needed
    missing_essentials = [col for col in essential_cols if col not in input_df.columns]
    if missing_essentials:
         raise ValueError(f"Input DataFrame missing essential columns: {missing_essentials}")
         
    df_out = input_df[essential_cols].copy() # Start with essential original data

    # Compute technical features
    df_features = compute_technical_features(input_df, window=window)

    # Combine original essentials with new features
    # Use pd.concat for robust joining on the index
    df_out = pd.concat([df_out, df_features], axis=1)

    # --- Not Dropping NaNs here ---
    # NaNs from rolling windows and shifts are expected at the start/end.
    # The calling script (train_generic_model.py or app.py) will handle 
    # dropping NaNs *after* creating the 'Target_Price' from the 'Close' column.

    return df_out

# --- Original main function (Modified for Clarity/Context) ---
# This main function might not be directly used for the generic model training,
# but shows how the new engineer_features_for_stock would be used if you 
# wanted to save the combined data per ticker.
def main_save_combined_features():
    raw_folder = "data/raw"
    processed_folder = "data/processed_combined" # Save to a different folder
    os.makedirs(processed_folder, exist_ok=True)

    for fname in os.listdir(raw_folder):
        if not fname.endswith(".csv"):
            continue

        ticker = fname[:-4]
        fpath = os.path.join(raw_folder, fname)
        out_path = os.path.join(processed_folder, f"{ticker}_combined_features.csv")
        
        try:
            # Load raw data
            df_raw = pd.read_csv(fpath, index_col="Date", parse_dates=True)
            df_raw.sort_index(inplace=True)

            # Engineer features and combine with essentials
            df_combined = engineer_features_for_stock(df_raw, window=10)

            # Optionally drop NaNs *before* saving if this is just for inspection
            # df_combined.dropna(inplace=True) 
            
            df_combined.to_csv(out_path)
            print(f"Processed {ticker}: {len(df_combined)} rows saved to {out_path}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")


# You wouldn't typically run the main function directly from here when training.
# The train_generic_model.py script will import engineer_features_for_stock.
# if __name__ == "__main__":
#    main_save_combined_features()