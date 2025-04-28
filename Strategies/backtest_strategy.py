import pandas as pd
import numpy as np
import argparse
import os
import traceback

# --- Configuration ---
INPUT_FEATURE_DIR = "data/complete" # Where the engineered files are
SMA_LONG_WINDOW = 200 # Must match the window used in engineer.py

def load_data_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Loads the complete  CSV file for a single ticker."""
    filepath = os.path.join(INPUT_FEATURE_DIR, f"{ticker}_complete.csv")
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found for {ticker} at '{filepath}'.")
        print(f"Please ensure you have run 'ticker_script.py {ticker}' (using the latest engineer.py).")
        return None
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        print(f"Loaded {len(df)} rows for {ticker} from '{filepath}'.")
        return df
    except Exception as e:
        print(f"Error loading data for {ticker} from '{filepath}': {e}")
        traceback.print_exc()
        return None

def backtest_strategy(df: pd.DataFrame, ticker: str):
    """
    Performs a backtest of the MACD + SMA 200 strategy.
    Assumes df contains 'Close', 'Open', 'MACD_line', 'MACD_signal', 'sma_200'.
    """
    print(f"\n--- Backtesting Strategy for {ticker} ---")
    
    # --- Verify Required Columns ---
    required_cols = ['Close', 'Open', 'MACD_line', 'MACD_signal', f'sma_{SMA_LONG_WINDOW}']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: DataFrame is missing required columns for backtest: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # --- Prepare Data & Signals ---
    # Make sure we have enough data after dropping NaNs from indicators
    df_clean = df[required_cols].dropna().copy() 
    if len(df_clean) < 2: # Need at least 2 rows to check crossovers
        print("Error: Not enough data after dropping NaNs to perform backtest.")
        return
        
    sma_col_name = f'sma_{SMA_LONG_WINDOW}' # Get the exact SMA column name

    # 1. Trend Condition
    df_clean['Uptrend'] = df_clean['Close'] > df_clean[sma_col_name]
    df_clean['Downtrend'] = df_clean['Close'] < df_clean[sma_col_name]

    # 2. MACD Crossover Conditions
    #   Check if MACD crossed above signal *today* (compared to yesterday)
    df_clean['MACD_Cross_Up'] = (df_clean['MACD_line'] > df_clean['MACD_signal']) & \
                               (df_clean['MACD_line'].shift(1) < df_clean['MACD_signal'].shift(1))
    #   Check if MACD crossed below signal *today*
    df_clean['MACD_Cross_Down'] = (df_clean['MACD_line'] < df_clean['MACD_signal']) & \
                                 (df_clean['MACD_line'].shift(1) > df_clean['MACD_signal'].shift(1))

    # 3. Zero Line Conditions (at the time of crossover)
    df_clean['Below_Zero'] = (df_clean['MACD_line'] < 0) & (df_clean['MACD_signal'] < 0)
    df_clean['Above_Zero'] = (df_clean['MACD_line'] > 0) & (df_clean['MACD_signal'] > 0)

    # 4. Combine for Entry Signals
    df_clean['Buy_Signal'] = df_clean['Uptrend'] & df_clean['MACD_Cross_Up'] & df_clean['Below_Zero']
    df_clean['Short_Signal'] = df_clean['Downtrend'] & df_clean['MACD_Cross_Down'] & df_clean['Above_Zero']

    # 5. Define Exit Signals (Simple crossover in opposite direction)
    df_clean['Exit_Long_Signal'] = df_clean['MACD_Cross_Down'] 
    df_clean['Exit_Short_Signal'] = df_clean['MACD_Cross_Up']

    # --- Simulate Trades ---
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0
    trades = [] # List to store trade results [entry_date, exit_date, entry_price, exit_price, pnl]

    # Loop through the DataFrame (index gives the date)
    # We need to look ahead one day for entry/exit prices, so iterate up to second-to-last row
    for i in range(len(df_clean) - 1):
        current_date = df_clean.index[i]
        next_date = df_clean.index[i+1]
        next_open_price = df_clean['Open'].iloc[i+1] # Use next day's open for entry/exit

        # Check for Long Entry
        if position == 0 and df_clean['Buy_Signal'].iloc[i]:
            position = 1
            entry_price = next_open_price
            entry_date = next_date
            print(f"{entry_date.date()}: Buy Signal -> Enter Long @ {entry_price:.2f}")

        # Check for Long Exit
        elif position == 1 and df_clean['Exit_Long_Signal'].iloc[i]:
            exit_price = next_open_price
            pnl = exit_price - entry_price
            trades.append([entry_date, next_date, entry_price, exit_price, pnl])
            print(f"{next_date.date()}: Exit Long Signal -> Exit @ {exit_price:.2f}, PnL: {pnl:.2f}")
            position = 0
            entry_price = 0

        # Check for Short Entry
        elif position == 0 and df_clean['Short_Signal'].iloc[i]:
            position = -1
            entry_price = next_open_price
            entry_date = next_date
            print(f"{entry_date.date()}: Short Signal -> Enter Short @ {entry_price:.2f}")

        # Check for Short Exit
        elif position == -1 and df_clean['Exit_Short_Signal'].iloc[i]:
            exit_price = next_open_price
            pnl = entry_price - exit_price # PnL is Entry - Exit for shorts
            trades.append([entry_date, next_date, entry_price, exit_price, pnl])
            print(f"{next_date.date()}: Exit Short Signal -> Exit @ {exit_price:.2f}, PnL: {pnl:.2f}")
            position = 0
            entry_price = 0
            
    # If still holding a position at the end, close it using the last available price (e.g., last close)
    if position == 1:
        exit_price = df_clean['Close'].iloc[-1]
        pnl = exit_price - entry_price
        trades.append([entry_date, df_clean.index[-1], entry_price, exit_price, pnl])
        print(f"{df_clean.index[-1].date()}: End of Data -> Exit Long @ {exit_price:.2f}, PnL: {pnl:.2f}")
    elif position == -1:
         exit_price = df_clean['Close'].iloc[-1]
         pnl = entry_price - exit_price
         trades.append([entry_date, df_clean.index[-1], entry_price, exit_price, pnl])
         print(f"{df_clean.index[-1].date()}: End of Data -> Exit Short @ {exit_price:.2f}, PnL: {pnl:.2f}")


    # --- Calculate Performance Metrics ---
    if not trades:
        print("\nNo trades were executed.")
        return

    trades_df = pd.DataFrame(trades, columns=['Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'PnL'])
    
    total_pnl = trades_df['PnL'].sum()
    num_trades = len(trades_df)
    num_wins = (trades_df['PnL'] > 0).sum()
    num_losses = (trades_df['PnL'] <= 0).sum()
    win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
    avg_win = trades_df[trades_df['PnL'] > 0]['PnL'].mean() if num_wins > 0 else 0
    avg_loss = trades_df[trades_df['PnL'] <= 0]['PnL'].mean() if num_losses > 0 else 0
    profit_factor = abs(trades_df[trades_df['PnL'] > 0]['PnL'].sum() / trades_df[trades_df['PnL'] <= 0]['PnL'].sum()) if num_losses > 0 and trades_df[trades_df['PnL'] <= 0]['PnL'].sum() != 0 else float('inf')

    print("\n--- Backtest Performance Summary ---")
    print(f"Total Trades:        {num_trades}")
    print(f"Winning Trades:      {num_wins}")
    print(f"Losing Trades:       {num_losses}")
    print(f"Win Rate:            {win_rate:.2f}%")
    print(f"Average Win ($):     {avg_win:.2f}")
    print(f"Average Loss ($):    {avg_loss:.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Total PnL ($):       {total_pnl:.2f}") # Note: This is absolute PnL, not % return
    print("------------------------------------")
    
    # Optional: Save trades to CSV
    # trades_df.to_csv(f"backtest_results_{ticker}.csv")

def main():
    """Main function to load data and run backtest for a ticker."""
    parser = argparse.ArgumentParser(description="Backtest MACD + SMA 200 strategy on a single stock.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol to backtest")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    df_data = load_data_for_ticker(ticker)
    if df_data is not None:
        backtest_strategy(df_data, ticker)

if __name__ == "__main__":
    main()
