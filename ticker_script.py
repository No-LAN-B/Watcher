import requests
import pandas as pd
import argparse
import os
import sys
import traceback 

# --- Configuration ---
FLASK_API_BASE_URL = "http://127.0.0.1:5000" 
# Default output directory for complete data (including Close)
DEFAULT_OUTPUT_DIR = "data/complete" 
# Output directory when generating data specifically for training (without Close)
# Note: We determined the --train flag wasn't needed for the current training workflow
# TRAINING_OUTPUT_DIR = "data/training_features" 

# --- Attempt to import the feature engineering function ---
try:
    from src.features.engineer import engineer_features_for_stock 
except ImportError:
    print("Error: Could not import 'engineer_features_for_stock' from 'src.features.engineer'.")
    # ... (rest of import error message) ...
    sys.exit(1) 
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

def fetch_stock_data_from_api(ticker: str) -> pd.DataFrame | None:
    """
    Fetches historical stock data from the Flask API endpoint /api/stockdata.
    Handles potential lowercase column names from the API and renames them 
    to the standard uppercase format expected by the feature engineering function.
    """
    api_url = f"{FLASK_API_BASE_URL}/api/stockdata"
    params = {'symbol': ticker}
    print(f"Requesting data for {ticker} from {api_url}...")
    
    try:
        response = requests.get(api_url, params=params, timeout=20) 
        response.raise_for_status() 
        data = response.json()
        
        if isinstance(data, dict) and 'error' in data:
             print(f"API Error for {ticker}: {data['error']}")
             return None
             
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame.from_records(data)
            
            # --- Rename columns received from API ---
            rename_map = {
                'time': 'Date', 'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }
            columns_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
            if columns_to_rename:
                 print(f"Renaming columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
                 df.rename(columns=columns_to_rename, inplace=True)
            # --------------------------------------------

            if 'Date' in df.columns:
                 if df.index.name != 'Date':
                     df['Date'] = pd.to_datetime(df['Date'])
                     df.set_index('Date', inplace=True)
                 else:
                     df.index = pd.to_datetime(df.index)
                 df.sort_index(inplace=True) 
                 
                 essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
                 if all(col in df.columns for col in essential_cols):
                      print(f"Successfully fetched and prepared {len(df)} rows for {ticker}.")
                      return df
                 else:
                      missing = [col for col in essential_cols if col not in df.columns]
                      print(f"Error: DataFrame for {ticker} missing essential columns after preparation: {missing}")
                      print("Columns received and prepared:", df.columns.tolist())
                      return None
            else:
                 print(f"Error: Response for {ticker} missing 'time'/'Date' column.")
                 return None
        elif isinstance(data, list) and len(data) == 0:
             print(f"API returned empty data list for {ticker}.")
             return None
        else:
            print(f"Error: Unexpected data format received from API for {ticker}: {type(data)}")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error fetching data for {ticker}: {http_err}")
        try: error_details = response.json(); print(f"API Response Error: {error_details}")
        except Exception: print(f"Raw Response Content: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Flask API at {FLASK_API_BASE_URL}. Is app.py running?")
        return None
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for {ticker}.")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error fetching data for {ticker}: {req_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data fetching for {ticker}: {e}")
        traceback.print_exc()
        return None

def main():
    """Main function to fetch, process, and save data for a ticker."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fetch and engineer features for a specific stock ticker using the backend API.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol (e.g., AAPL)")
    # Removed the --train flag as it's not needed for the current training workflow
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    # --- Set Output Directory (Always save complete features now) ---
    output_dir = DEFAULT_OUTPUT_DIR
    print(f"\n--- Running script for {ticker} ---")
    print(f"Output will be saved to: {output_dir}")
        
    # --- Fetch Data ---
    df_fetched = fetch_stock_data_from_api(ticker)
    if df_fetched is None:
        print(f"\nCould not fetch or prepare data for ticker '{ticker}'. Exiting.")
        return 

    # --- Engineer Features ---
    print(f"\nEngineering features for {ticker}...")
    try:
        # ** FIX: Call engineer_features_for_stock WITHOUT the 'window' argument **
        df_engineered = engineer_features_for_stock(df_fetched) 
        print("Features engineered successfully.")
    except ValueError as ve: 
         print(f"Error during feature engineering for {ticker}: {ve}")
         print("DataFrame columns passed to engineer function:", df_fetched.columns.tolist())
         return
    except Exception as e:
        print(f"An unexpected error occurred during feature engineering for {ticker}: {e}")
        traceback.print_exc()
        return

    # --- Save the Result ---
    os.makedirs(output_dir, exist_ok=True) 
    # Always save as '_complete_features.csv' now
    output_filename = os.path.join(output_dir, f"{ticker}_complete.csv")
    
    try:
        df_engineered.to_csv(output_filename)
        print(f"\nSuccessfully saved data for {ticker} to: {output_filename}")
        print("\nFinal DataFrame Info:")
        df_engineered.info() 
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
