import pandas as pd
import numpy as np
import argparse
import os
import traceback

# --- Configuration ---
INPUT_FEATURE_DIR = "data/complete" # Where the engineered files are
SMA_LONG_WINDOW = 200 # Must match the window used in engineer.py
INITIAL_CAPITAL = 5000.0 
COMMISSION_PER_TRADE = 1.00 

def load_data_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Loads the complete features CSV file for a single ticker."""
    filepath = os.path.join(INPUT_FEATURE_DIR, f"{ticker}_complete.csv")
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found for {ticker} at '{filepath}'.")
        return None
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        print(f"Loaded {len(df)} rows for {ticker} from '{filepath}'.")
        # Ensure essential columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', f'sma_{SMA_LONG_WINDOW}', 'MACD_line', 'MACD_signal']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                 # If a required column for the strategy is missing entirely, raise error early
                 if col in ['Open', 'High', 'Low', 'Close', f'sma_{SMA_LONG_WINDOW}', 'MACD_line', 'MACD_signal']:
                     raise ValueError(f"Essential column '{col}' missing from data.")
        # Drop rows if essential price/vol or the specific indicators are missing AFTER converting
        df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume', f'sma_{SMA_LONG_WINDOW}', 'MACD_line', 'MACD_signal'], inplace=True) 
        return df
    except Exception as e:
        print(f"Error loading or preparing data for {ticker} from '{filepath}': {e}")
        traceback.print_exc()
        return None

def calculate_max_drawdown(portfolio_values: pd.Series) -> tuple[float, float]:
    """Calculates the maximum drawdown percentage and value."""
    if portfolio_values.empty or portfolio_values.isnull().all():
        print("Warning: Portfolio value history is empty or all NaN, cannot calculate drawdown.")
        return 0.0, 0.0
    cumulative_max = portfolio_values.cummax()
    drawdown = (portfolio_values - cumulative_max) / cumulative_max 
    max_drawdown_date_index = drawdown.idxmin() 
    if pd.isna(max_drawdown_date_index):
         print("Warning: Could not determine max drawdown date index.")
         return 0.0, 0.0
    max_drawdown_pct = drawdown.loc[max_drawdown_date_index] 
    try:
        peak_value = cumulative_max.loc[max_drawdown_date_index] 
        trough_value = portfolio_values.loc[max_drawdown_date_index]
        max_drawdown_value = peak_value - trough_value
    except KeyError:
         print(f"Warning: Could not find date {max_drawdown_date_index} in series for drawdown value calculation.")
         return abs(max_drawdown_pct * 100), 0.0
    return abs(max_drawdown_pct * 100), max_drawdown_value 

def backtest_strategy(df: pd.DataFrame, ticker: str):
    """
    Performs a backtest of the MACD + SMA 200 strategy with starting capital
    and an SMA 200 based stop-loss.
    """
    print(f"\n--- Backtesting Strategy for {ticker} with ${INITIAL_CAPITAL:.2f} initial capital & SMA {SMA_LONG_WINDOW} Stop-Loss ---")
    
    # --- Verify Required Columns ---
    sma_col_name = f'sma_{SMA_LONG_WINDOW}'
    required_cols = ['Close', 'Open', 'High', 'Low', 'MACD_line', 'MACD_signal', sma_col_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: DataFrame is missing required columns for backtest: {missing_cols}")
        return

    # --- Prepare Data & Signals ---
    # Use required cols directly, dropna already handled in load function for these
    df_clean = df[required_cols].copy() 
    if len(df_clean) < 2: 
        print("Error: Not enough data to perform backtest.")
        return
        
    # Calculate signals (same as before)
    df_clean['Uptrend'] = df_clean['Close'] > df_clean[sma_col_name]
    df_clean['Downtrend'] = df_clean['Close'] < df_clean[sma_col_name]
    df_clean['MACD_Cross_Up'] = (df_clean['MACD_line'] > df_clean['MACD_signal']) & \
                               (df_clean['MACD_line'].shift(1) < df_clean['MACD_signal'].shift(1))
    df_clean['MACD_Cross_Down'] = (df_clean['MACD_line'] < df_clean['MACD_signal']) & \
                                 (df_clean['MACD_line'].shift(1) > df_clean['MACD_signal'].shift(1))
    df_clean['Below_Zero'] = (df_clean['MACD_line'] < 0) & (df_clean['MACD_signal'] < 0)
    df_clean['Above_Zero'] = (df_clean['MACD_line'] > 0) & (df_clean['MACD_signal'] > 0)
    df_clean['Buy_Signal'] = df_clean['Uptrend'] & df_clean['MACD_Cross_Up'] & df_clean['Below_Zero']
    df_clean['Short_Signal'] = df_clean['Downtrend'] & df_clean['MACD_Cross_Down'] & df_clean['Above_Zero']
    df_clean['Exit_Long_Signal'] = df_clean['MACD_Cross_Down'] 
    df_clean['Exit_Short_Signal'] = df_clean['MACD_Cross_Up']
    
    # ** NEW: Define Stop-Loss Conditions **
    # Stop Long if Low price crosses below the long SMA
    df_clean['Stop_Long_Signal'] = df_clean['Low'] < df_clean[sma_col_name]
    # Stop Short if High price crosses above the long SMA
    df_clean['Stop_Short_Signal'] = df_clean['High'] > df_clean[sma_col_name]

    # --- Initialize Portfolio ---
    cash = INITIAL_CAPITAL
    shares_held = 0
    portfolio_value_history = [] 
    trades = [] 
    entry_price = 0 
    entry_date = None 

    # --- Simulate Trades ---
    print("\n--- Trade Log ---")
    for i in range(len(df_clean) - 1):
        current_date = df_clean.index[i]
        next_date = df_clean.index[i+1]
        try:
            next_open_price = float(df_clean['Open'].iloc[i+1]) 
            current_close_price = float(df_clean['Close'].iloc[i]) 
        except (ValueError, TypeError):
            print(f"Warning: Skipping date {current_date} due to non-numeric price data.")
            if portfolio_value_history:
                 portfolio_value_history.append({'Date': current_date, 'Portfolio_Value': portfolio_value_history[-1]['Portfolio_Value']})
            else:
                 portfolio_value_history.append({'Date': current_date, 'Portfolio_Value': cash})
            continue 

        current_portfolio_value = cash + (shares_held * current_close_price)
        portfolio_value_history.append({'Date': current_date, 'Portfolio_Value': current_portfolio_value})

        # --- Check Exits FIRST (Stop-Loss has priority) ---
        stop_loss_triggered = False
        
        # Check Stop-Loss for Long Position
        if shares_held > 0 and df_clean['Stop_Long_Signal'].iloc[i]:
            exit_price = next_open_price # Exit at next open after SL breach
            exit_reason = "Stop-Loss"
            stop_loss_triggered = True
            
        # Check Stop-Loss for Short Position
        elif shares_held < 0 and df_clean['Stop_Short_Signal'].iloc[i]:
            exit_price = next_open_price # Exit at next open after SL breach
            exit_reason = "Stop-Loss"
            stop_loss_triggered = True
            
        # Check Regular Exit Signal for Long Position (if not stopped out)
        elif shares_held > 0 and df_clean['Exit_Long_Signal'].iloc[i]:
            exit_price = next_open_price
            exit_reason = "Exit Signal"
            
        # Check Regular Exit Signal for Short Position (if not stopped out)
        elif shares_held < 0 and df_clean['Exit_Short_Signal'].iloc[i]:
            exit_price = next_open_price
            exit_reason = "Exit Signal"
            
        # --- Process Exit ---
        if shares_held != 0 and 'exit_price' in locals() and exit_price is not None: # Check if an exit was triggered
            if shares_held > 0: # Exiting a long position
                exit_value = shares_held * exit_price
                cash += (exit_value - COMMISSION_PER_TRADE)
                pnl = (exit_price - entry_price) * shares_held - (2 * COMMISSION_PER_TRADE) 
                trades.append([entry_date, next_date, entry_price, exit_price, shares_held, pnl, exit_reason])
                print(f"{next_date.date()}: {exit_reason} -> Exit Long {shares_held} shares @ {exit_price:.2f}, PnL: {pnl:.2f}")
            else: # Exiting a short position
                cost_to_cover = abs(shares_held) * exit_price
                cash -= (cost_to_cover + COMMISSION_PER_TRADE) 
                pnl = (entry_price - exit_price) * abs(shares_held) - (2 * COMMISSION_PER_TRADE) 
                trades.append([entry_date, next_date, entry_price, exit_price, abs(shares_held), pnl, exit_reason])
                print(f"{next_date.date()}: {exit_reason} -> Cover {abs(shares_held)} shares @ {exit_price:.2f}, PnL: {pnl:.2f}")
                
            # Reset position state AFTER processing exit
            shares_held = 0
            entry_price = 0
            entry_date = None
            del exit_price # Clear exit_price for next iteration
            stop_loss_triggered = False # Reset stop loss flag

        # --- Check Entries (Only if currently flat) ---
        if shares_held == 0:
            # Check for Long Entry
            if df_clean['Buy_Signal'].iloc[i]:
                if cash > next_open_price + COMMISSION_PER_TRADE: 
                    shares_to_buy = int((cash - COMMISSION_PER_TRADE) // next_open_price) 
                    if shares_to_buy > 0:
                        entry_cost = shares_to_buy * next_open_price
                        cash -= (entry_cost + COMMISSION_PER_TRADE)
                        shares_held = shares_to_buy
                        entry_price = next_open_price
                        entry_date = next_date
                        print(f"{entry_date.date()}: Buy Signal -> Enter Long {shares_held} shares @ {entry_price:.2f}")
                # else: # Optional: Reduce noise
                #      print(f"{next_date.date()}: Buy Signal -> Insufficient cash.")

            # Check for Short Entry (Can only enter one or the other)
            elif df_clean['Short_Signal'].iloc[i]:
                affordable_shares = int((cash - COMMISSION_PER_TRADE) // next_open_price) 
                if affordable_shares > 0:
                     shares_to_short = affordable_shares 
                     entry_price = next_open_price 
                     cash += (shares_to_short * entry_price - COMMISSION_PER_TRADE) 
                     shares_held = -shares_to_short 
                     entry_date = next_date
                     print(f"{entry_date.date()}: Short Signal -> Enter Short {abs(shares_held)} shares @ {entry_price:.2f}")
                # else: # Optional: Reduce noise
                #      print(f"{next_date.date()}: Short Signal -> Insufficient buying power.")
            
    # --- Final Portfolio Value & Closeout ---
    if not df_clean.empty:
        last_close_price = float(df_clean['Close'].iloc[-1])
        if shares_held > 0: # Close out final long position
             exit_value = shares_held * last_close_price
             cash += (exit_value - COMMISSION_PER_TRADE)
             pnl = (last_close_price - entry_price) * shares_held - (2 * COMMISSION_PER_TRADE) 
             trades.append([entry_date, df_clean.index[-1], entry_price, last_close_price, shares_held, pnl, "End of Data"])
             print(f"{df_clean.index[-1].date()}: End of Data -> Exit Long {shares_held} shares @ {last_close_price:.2f}, PnL: {pnl:.2f}")
             shares_held = 0
        elif shares_held < 0: # Close out final short position
             cost_to_cover = abs(shares_held) * last_close_price
             cash -= (cost_to_cover + COMMISSION_PER_TRADE)
             pnl = (entry_price - last_close_price) * abs(shares_held) - (2 * COMMISSION_PER_TRADE) 
             trades.append([entry_date, df_clean.index[-1], entry_price, last_close_price, abs(shares_held), pnl, "End of Data"])
             print(f"{df_clean.index[-1].date()}: End of Data -> Cover {abs(shares_held)} shares @ {last_close_price:.2f}, PnL: {pnl:.2f}")
             shares_held = 0
             
        final_portfolio_value = cash # Final value is cash after closing last position
        portfolio_value_history.append({'Date': df_clean.index[-1], 'Portfolio_Value': final_portfolio_value})
        print(f"\nFinal Portfolio Value: ${final_portfolio_value:.2f}")
    else:
        final_portfolio_value = cash 
        print("\nFinal Portfolio Value (No data): ${final_portfolio_value:.2f}")

    # --- Calculate Performance Metrics ---
    if not trades:
        print("\nNo trades were executed.")
        return

    # Add Exit_Reason column
    trades_df = pd.DataFrame(trades, columns=['Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Shares', 'PnL', 'Exit_Reason']) 
    
    if portfolio_value_history:
         portfolio_history_df = pd.DataFrame(portfolio_value_history).set_index('Date')
         portfolio_values_series = portfolio_history_df['Portfolio_Value'].astype(float)
         max_drawdown_pct, max_drawdown_value = calculate_max_drawdown(portfolio_values_series)
    else:
         max_drawdown_pct, max_drawdown_value = 0.0, 0.0
         print("Warning: Portfolio history empty, cannot calculate drawdown.")

    
    total_pnl = trades_df['PnL'].sum()
    num_trades = len(trades_df)
    num_wins = (trades_df['PnL'] > 0).sum()
    num_losses = num_trades - num_wins 
    win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0
    avg_win = trades_df[trades_df['PnL'] > 0]['PnL'].mean() if num_wins > 0 else 0
    avg_loss = trades_df[trades_df['PnL'] <= 0]['PnL'].mean() if num_losses > 0 else 0
    gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(trades_df[trades_df['PnL'] <= 0]['PnL'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_return_pct = ((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL > 0 else float('inf')
    
    # Count stop-loss exits
    stop_loss_exits = trades_df[trades_df['Exit_Reason'] == 'Stop-Loss'].shape[0]

    print("\n--- Backtest Performance Summary ---")
    print(f"Initial Capital:     ${INITIAL_CAPITAL:.2f}")
    print(f"Final Portfolio Val: ${final_portfolio_value:.2f}")
    print(f"Total PnL ($):       ${total_pnl:.2f}") 
    print(f"Total Return:        {total_return_pct:.2f}%")
    print(f"Max Drawdown:        {max_drawdown_pct:.2f}% (${max_drawdown_value:.2f})")
    print("-" * 20)
    print(f"Total Trades:        {num_trades}")
    print(f"Stop-Loss Exits:     {stop_loss_exits}") # New metric
    print(f"Winning Trades:      {num_wins}")
    print(f"Losing Trades:       {num_losses}")
    print(f"Win Rate:            {win_rate:.2f}%")
    print(f"Average Win ($):     {avg_win:.2f}")
    print(f"Average Loss ($):    {avg_loss:.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print("------------------------------------")
    
    # Optional: Save trades to CSV
    # trades_df.to_csv(f"backtest_results_{ticker}_capital_sl.csv")

def main():
    """Main function to load data and run backtest for a ticker."""
    parser = argparse.ArgumentParser(description="Backtest MACD + SMA 200 strategy on a single stock with initial capital and stop-loss.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol to backtest (e.g., AAPL)")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    df_data = load_data_for_ticker(ticker)
    if df_data is not None:
        backtest_strategy(df_data, ticker)

if __name__ == "__main__":
    main()
