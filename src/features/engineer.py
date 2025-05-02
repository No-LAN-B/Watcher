import os
import pandas as pd
import pandas_ta as ta 
import numpy as np 
import traceback 

# --- Default parameters for indicators ---
SMA_WINDOW_SHORT = 10 # Keep the short one if needed elsewhere
SMA_WINDOW_LONG = 200 # <<< Add constant for the long SMA
VOL_WINDOW = 10
RSI_WINDOW = 14
ATR_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2.0 # Ensure float for pandas-ta column naming
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9



def compute_technical_features(df: pd.DataFrame, 
                               sma_window_short: int = SMA_WINDOW_SHORT, 
                               sma_window_long: int = SMA_WINDOW_LONG, # <<< Add parameter
                               vol_window: int = VOL_WINDOW,
                               rsi_window: int = RSI_WINDOW,
                               atr_window: int = ATR_WINDOW,
                               bb_window: int = BB_WINDOW, 
                               bb_std: float = BB_STD, 
                               macd_fast: int = MACD_FAST, 
                               macd_slow: int = MACD_SLOW, 
                               macd_signal: int = MACD_SIGNAL
                               ) -> pd.DataFrame:
    """
    Calculates technical features based on the input DataFrame.
    Input DataFrame MUST contain 'Open', 'High', 'Low', 'Close', 'Volume' columns.
    Returns a DataFrame with ONLY the calculated features (and NaNs where applicable).
    """
    required_original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_original_cols):
         missing = [col for col in required_original_cols if col not in df.columns]
         raise ValueError(f"Input DataFrame must contain columns: {missing}")
         
    df_copy = df.copy()
    features = pd.DataFrame(index=df_copy.index) 

    # --- Feature Calculations ---
    features['return'] = df_copy['Close'].pct_change()
    features[f'sma_{sma_window_short}'] = df_copy['Close'].rolling(sma_window_short).mean()
    # <<< Add SMA 200 >>>
    features[f'sma_{sma_window_long}'] = df_copy['Close'].rolling(sma_window_long).mean() 
    
    if 'return' in features.columns:
         features[f'vol_{vol_window}'] = features['return'].rolling(vol_window).std()
    else:
         features[f'vol_{vol_window}'] = pd.Series(index=df.index, dtype=float)
         
    features['Close_Lag_1'] = df_copy['Close'].shift(1)
    features[f'RSI_{rsi_window}'] = df_copy.ta.rsi(length=rsi_window)
    
    try: # MACD
        df_copy.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
        macd_line_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
        macd_hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
        macd_signal_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
        if all(c in df_copy.columns for c in [macd_line_col, macd_hist_col, macd_signal_col]):
            features['MACD_line'] = df_copy[macd_line_col]
            features['MACD_hist'] = df_copy[macd_hist_col]
            features['MACD_signal'] = df_copy[macd_signal_col]
        else: raise RuntimeError("MACD columns not generated") # Raise error if missing
    except Exception as e:
        print(f"Warning: Error calculating MACD: {e}. Filling with NaN.")
        features['MACD_line'], features['MACD_hist'], features['MACD_signal'] = np.nan, np.nan, np.nan

    features[f'ATR_{atr_window}'] = df_copy.ta.atr(length=atr_window)

    try: # Bollinger Bands
        df_copy.ta.bbands(length=bb_window, std=bb_std, append=True)
        bb_lower_col = f'BBL_{bb_window}_{float(bb_std)}'
        bb_middle_col = f'BBM_{bb_window}_{float(bb_std)}'
        bb_upper_col = f'BBU_{bb_window}_{float(bb_std)}'
        bb_width_col = f'BBB_{bb_window}_{float(bb_std)}'
        bb_percent_col = f'BBP_{bb_window}_{float(bb_std)}'
        bb_cols_to_copy = [bb_lower_col, bb_middle_col, bb_upper_col, bb_width_col, bb_percent_col]
        if all(col in df_copy.columns for col in bb_cols_to_copy):
            features['BB_Lower'] = df_copy[bb_lower_col]
            features['BB_Middle'] = df_copy[bb_middle_col]
            features['BB_Upper'] = df_copy[bb_upper_col]
            features['BB_Width'] = df_copy[bb_width_col]
            features['BB_Percent'] = df_copy[bb_percent_col]
        else: raise RuntimeError("Bollinger Band columns not generated")
    except Exception as e:
        print(f"Warning: Error calculating Bollinger Bands: {e}. Filling with NaN.")
        features['BB_Lower'], features['BB_Middle'], features['BB_Upper'] = np.nan, np.nan, np.nan
        features['BB_Width'], features['BB_Percent'] = np.nan, np.nan
        
    return features

def engineer_features_for_stock(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame (Date index, OHLCV) for a single ticker,
    calculates features using compute_technical_features (with default windows), 
    and returns a new DataFrame containing both the essential original columns 
    AND the calculated features. Does NOT drop NaNs here.
    """
    essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    missing_essentials = [col for col in essential_cols if col not in input_df.columns]
    if missing_essentials:
         raise ValueError(f"Input DataFrame missing essential columns: {missing_essentials}")
         
    df_out = input_df[essential_cols].copy() 
    df_features = compute_technical_features(input_df.copy()) # Pass copy
    df_out = df_out.join(df_features) 
    return df_out

# (No main() function needed for file I/O in this library version)
