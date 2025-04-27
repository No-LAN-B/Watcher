import requests
import pandas as pd
import argparse
import os
import sys
#script for dynamic ticker data conversion for training the ml model on the flask server
# This script fetches stock data from a Flask API, engineers features using a function from engineer.py,
# It will then send it to my training script for the ml model to train on the data



# --- Configuration ---
# Assuming Flask app runs locally on port 5000 change if not the case
FLASK_API_BASE_URL = "http://127.0.0.1:5000" 
# Output directory for the final engineered CSVs
OUTPUT_DIR = "data/complete" 

# --- Attempt to import the feature engineering function ---
# Adjust the path if your structure is different (e.g., if ticker_script.py is not in the root)
try:
    # Assumes ticker_script.py is in the project root, and engineer.py is in src/features/
    from src.features.engineer import engineer_features_for_stock 
except ImportError:
    print("Error: Could not import 'engineer_features_for_stock' from 'src.features.engineer'.")
    print("Please ensure:")
    print("1. 'engineer.py' exists in the 'src/features/' directory.")
    print("2. Your 'src' and 'src/features' directories contain an empty '__init__.py' file.")
    print("3. You are running this script from the project's root directory.")
    sys.exit(1) # Exit if the crucial function can't be imported

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
            # Convert 'time' column to datetime and set as index
            if 'time' in df.columns:
                 df['Date'] = pd.to_datetime(df['time'])
                 df.set_index('Date', inplace=True)
                 df.drop(columns=['time'], inplace=True)
                 df.sort_index(inplace=True) # Ensure chronological order
                 print(f"Successfully fetched and parsed {len(df)} rows for {ticker}.")
                 return df
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
    # --- Argument Parsing (for CLI usage) ---
    parser = argparse.ArgumentParser(description="Fetch and engineer features for a specific stock ticker using the backend API.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol (e.g., AAPL)")
    args = parser.parse_args()
    ticker = args.ticker.upper() # Standardize to uppercase

    # --- Fetch Data ---
    df_fetched = fetch_stock_data_from_api(ticker)
    
    if df_fetched is None:
        print(f"Could not proceed for ticker {ticker}.")
        return # Exit if fetching failed

    # --- Engineer Features ---
    # This assumes df_fetched contains the necessary OHLCV columns
    print(f"Engineering features for {ticker}...")
    try:
        df_engineered = engineer_features_for_stock(df_fetched, window=10) # Use the imported function
        print("Features engineered successfully.")
    except Exception as e:
        print(f"Error during feature engineering for {ticker}: {e}")
        return

    # --- Save the Result ---
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output dir if needed
    output_filename = os.path.join(OUTPUT_DIR, f"{ticker}_complete_features.csv")
    
    try:
        df_engineered.to_csv(output_filename)
        print(f"Successfully saved engineered data for {ticker} to: {output_filename}")
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")

if __name__ == "__main__":
    # Make sure your Flask server (app.py) is running in a separate terminal
    # before executing this script.
    main()
