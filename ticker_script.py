import requests
import pandas as pd
import argparse
import os
import sys
#script for dynamic ticker data conversion for training the ml model on the flask server
# This script fetches stock data from a Flask API, engineers features using a function from engineer.py,
# It will then send it to my training script for the ml model to train on the data

# Note: This project is partially a learning exercise so commenting why / how something works will be very common as I use those comments to learn and remember. 


# Configuration
# Assuming Flask app runs locally on port 5000 change if not the case
FLASK_API_BASE_URL = "http://127.0.0.1:5000" 

# Output directory for the final engineered CSVs
DEFAULT_OUTPUT_DIR = "data/complete"

# Output directory when generating data specifically for training (without Close)
TRAINING_OUTPUT_DIR = "data/training_data" 

# Attempt to import the feature engineering function
# Adjust the path if your structure is different (e.g., if ticker_script.py is not in the root)
try:
    from src.features.engineer import engineer_features_for_stock 
except ImportError:
    print("Error: Could not import 'engineer_features_for_stock' from 'src.features.engineer'.")
    # ... (rest of import error message if I need to extend it) ...
    sys.exit(1) 
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)

def fetch_stock_data_from_api(ticker: str) -> pd.DataFrame | None:
    """Fetches historical stock data from the Flask API."""
    api_url = f"{FLASK_API_BASE_URL}/api/stockdata"
    params = {'symbol': ticker}
    print(f"Requesting data for {ticker} from {api_url}...")
    
    try:
        response = requests.get(api_url, params=params, timeout=15) # Increased timeout slightly
        
        # Check for HTTP errors (like 404 Not Found, 5xx Server Error)
        response.raise_for_status() 
        
        data = response.json()
        
        # Check for application-level errors returned in JSON
        if isinstance(data, dict) and 'error' in data:
             print(f"API Error for {ticker}: {data['error']}")
             return None
             
        # Convert JSON list of records to DataFrame
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame.from_records(data)
            # --- FIX: Rename columns received from API ---
            # Define the mapping from potential lowercase API keys to standard uppercase
            rename_map = {
                'time': 'Date',   # Expected by set_index
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume' # Add volume if your API returns it
            }
            
            # Create a dictionary of columns that actually exist in the DataFrame and need renaming
            columns_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
            
            if columns_to_rename:
                 print(f"Renaming columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
                 df.rename(columns=columns_to_rename, inplace=True)
            
            # Convert 'Date' column to datetime and set as index
            if 'Date' in df.columns:
                 # Ensure the 'Date' column is not already the index before setting it
                 if df.index.name != 'Date':
                     df['Date'] = pd.to_datetime(df['Date'])
                     df.set_index('Date', inplace=True)
                 else:
                     # If 'Date' is already the index, ensure it's datetime
                     df.index = pd.to_datetime(df.index)
                     
                 df.sort_index(inplace=True) # Ensure chronological order
                 
                 # Check if essential columns are present AFTER potential rename
                 # These are needed by the engineer_features_for_stock function
                 essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
                 if all(col in df.columns for col in essential_cols):
                      print(f"Successfully fetched and prepared {len(df)} rows for {ticker}.")
                      return df
                 else:
                      missing = [col for col in essential_cols if col not in df.columns]
                      print(f"Error: DataFrame for {ticker} is missing essential columns after preparation: {missing}")
                      print("Columns received and prepared:", df.columns.tolist())
                      return None
            else:
                 print(f"Error: Response for {ticker} is missing the 'time' column.")
                 return None
        elif isinstance(data, list) and len(data) == 0:
             print(f"API returned empty data list for {ticker}. Ticker might be valid but have no recent data?")
             return None
        else:
            print(f"Error: Unexpected data format received from API for {ticker}: {type(data)}")
            return None

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
             print(f"Error: Ticker '{ticker}' not found or no data available via API (404).")
        else:
             print(f"HTTP Error fetching data for {ticker}: {http_err}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the Flask API at {FLASK_API_BASE_URL}.")
        print("Please ensure the Flask server (app.py) is running.")
        return None
    except requests.exceptions.Timeout:
        print(f"Error: Request timed out for {ticker}.")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request Error fetching data for {ticker}: {req_err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data fetching for {ticker}: {e}")
        return None

def main():
    """Main function to fetch, process, and save data for a ticker."""
    # Argument Parsing (for CLI usage) 
    parser = argparse.ArgumentParser(description="Fetch and engineer features for a specific stock ticker using the backend API.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol (e.g., AAPL)")
    
    # Add the optional --train flag for training models
    parser.add_argument(
        "--train", 
        action="store_true", # Makes it a boolean flag, True if present, False otherwise
        help="Generate data specifically for training (removes 'Close' column and saves to training folder)."
    )
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    is_training_mode = args.train # Check if the flag was used

    # Determine Output Directory
    if is_training_mode:
        output_dir = TRAINING_OUTPUT_DIR
        print(f"\n--- Running in TRAINING mode for {ticker} ---")
        print(f"Output will be saved to: {output_dir}")
        print("The 'Close' column will be REMOVED.")
    else:
        output_dir = DEFAULT_OUTPUT_DIR
        print(f"\n--- Running in standard mode for {ticker} ---")
        print(f"Output will be saved to: {output_dir}")
        print("The 'Close' column will be KEPT.")

    # Fetch Data
    df_fetched = fetch_stock_data_from_api(ticker)

    if df_fetched is None:
        print(f"Could not proceed for ticker {ticker}.")
        return # Exit if fetching failed

    # Engineer Features
    # This assumes df_fetched contains the necessary OHLCV columns
    print(f"Engineering features for {ticker}...")
    try:
        df_engineered = engineer_features_for_stock(df_fetched, window=10) # Use the imported function
        print("Features engineered successfully.")
    except Exception as e:
        print(f"Error during feature engineering for {ticker}: {e}")
        return

    # --- Modify DataFrame based on Mode ---
    if is_training_mode:
        # Remove the 'Close' column if it exists
        if 'Close' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['Close'])
            print("Removed 'Close' column for training data.")
        else:
            print("Warning: 'Close' column not found, could not remove it.")
            
    # --- Save the Result ---
    os.makedirs(output_dir, exist_ok=True) # Create output dir if needed
    # Use a different filename suffix based on mode for clarity
    file_suffix = "training_data" if is_training_mode else "complete"
    output_filename = os.path.join(output_dir, f"{ticker}_{file_suffix}.csv")
    
    try:
        df_engineered.to_csv(output_filename)
        print(f"\nSuccessfully saved data for {ticker} to: {output_filename}")
        print("\nFinal DataFrame Info:")
        df_engineered.info() # Show info of the saved DataFrame
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure Flask server (app.py) is running in a separate terminal
    # before executing this script.
    main()
