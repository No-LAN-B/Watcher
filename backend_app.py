from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import datetime # To get today's date

# Note: This project is partially a learning exercise so commenting why / how something works will be very common as I use those comments to learn and remember. 

# Import the specific provider class from your provider script
try:
    # Adjust 'src.data.provider' if your path is different
    from src.data.provider import YFinanceProvider 
except ImportError:
    print("Error: Could not import 'YFinanceProvider' from 'src.data.provider'.")
    print("Please ensure the file exists and is importable (check __init__.py files).")
    # Exit or provide a dummy class if import fails, otherwise Flask won't start
    # For demonstration, exiting is safer than potentially masking the error later.
    import sys
    sys.exit("Flask app cannot start without the DataProvider.")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/stockdata')
def get_stockdata_api():
    """
    API endpoint to fetch stock data for the chart.
    Takes 'symbol' as a query parameter.
    """
    symbol = request.args.get('symbol')
    # Future enhancement: Get start/end dates from request args?
    # start_date = request.args.get('start', '2015-01-01') 
    # end_date = request.args.get('end', datetime.date.today().strftime('%Y-%m-%d'))
    start_date = '2015-01-01' # Example start date
    end_date = datetime.date.today().strftime('%Y-%m-%d') # Today's date

    if not symbol:
        return jsonify({"error": "Missing 'symbol' query parameter"}), 400

    try:
        # --- Use your DataProvider class ---
        # Instantiate the provider
        provider = YFinanceProvider() # Using YFinance directly for now
        
        # Fetch data - provider.fetch expects a list of tickers
        # and returns a dictionary: {ticker: DataFrame}
        # We pass only the requested symbol in a list.
        raw_data_dict = provider.fetch(tickers=[symbol], start=start_date, end=end_date) #
        
        # Extract the DataFrame for the specific symbol
        df = raw_data_dict.get(symbol)
        # ------------------------------------

        if df is None or df.empty:
             return jsonify({"error": f"No data found for symbol: {symbol}"}), 404

        # --- Format data for Lightweight Charts (same as before) ---
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if isinstance(df.index, pd.DatetimeIndex):
             df = df.reset_index() 
        
        date_col_name = None
        for col in ['Date', 'index', 'timestamp', 'time']: # Check common date column names
             if col in df.columns:
                 date_col_name = col
                 break
        
        if date_col_name is None:
            return jsonify({"error": "DataFrame missing a recognizable date column/index"}), 500

        # Ensure correct 'time' format and column names
        df['time'] = pd.to_datetime(df[date_col_name]).dt.strftime('%Y-%m-%d')
        
        # Check if required OHLC columns exist before renaming
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"DataFrame missing required columns: {', '.join(missing_cols)}"}), 500

        df_chart = df[['time'] + required_cols].copy()
        df_chart.rename(columns={
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close'
            }, inplace=True) 

        chart_data = df_chart.to_dict(orient='records')
        
        return jsonify(chart_data)

    except Exception as e:
        print(f"Error processing request for {symbol}: {e}")
        # It's helpful to log the full traceback in debug mode
        import traceback
        traceback.print_exc() 
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)