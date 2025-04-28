import pandas as pd
import numpy as np
import xgboost as xgb 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import argparse
import sys
import traceback 

# --- Configuration ---
# Directory where individual stock complete CSVs are saved
INPUT_FEATURE_DIR = "data/complete" 
# Directory where the trained model and scaler were saved
MODEL_DIR = "models" 
# Model and scaler filenames (MUST match the saved files from training)
MODEL_FILENAME = "best_xgbregressor_predict_change.pkl"
SCALER_FILENAME = "scaler_for_best_xgbregressor_predict_change.pkl"

# List of feature columns the model was trained on
# IMPORTANT: This MUST exactly match the list used in train_generic_model.py
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', #'Close', 
    'Volume', 'return', 'sma_10', 'vol_10', 'Close_Lag_1',
    'RSI_14', 'MACD_line', 'MACD_hist', 'MACD_signal', 'ATR_14',
    'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Percent'
]
# The target variable the model was trained to predict
TARGET_COLUMN = 'Target_Change' 

def load_data_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Loads the complete CSV file for a single ticker."""
    filepath = os.path.join(INPUT_FEATURE_DIR, f"{ticker}_complete.csv")
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found for {ticker} at '{filepath}'.")
        print(f"Please ensure you have run 'ticker_script.py {ticker}' (without --train).")
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

def load_model_and_scaler() -> tuple[any, StandardScaler | None]:
    """Loads the pre-trained model and scaler."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)
    
    model = None
    scaler = None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please run the training script first.")
    else:
        try:
            model = joblib.load(model_path)
            print(f"Successfully loaded model from '{model_path}'.")
        except Exception as e:
            print(f"Error loading model from '{model_path}': {e}")
            traceback.print_exc()

    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at '{scaler_path}'.")
        print("Please run the training script first.")
    else:
        try:
            scaler = joblib.load(scaler_path)
            print(f"Successfully loaded scaler from '{scaler_path}'.")
        except Exception as e:
            print(f"Error loading scaler from '{scaler_path}': {e}")
            traceback.print_exc()
            
    return model, scaler

def main():
    """Evaluates the generic pre-trained model on a single specified stock."""
    parser = argparse.ArgumentParser(description="Evaluate the generic model on a single stock.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol to evaluate (e.g., AAPL)")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    print(f"\n--- Evaluating Generic Model on Single Stock: {ticker} ---")

    # --- 1. Load Model and Scaler ---
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        print("Exiting due to missing model or scaler.")
        return

    # --- 2. Load Data for the Specific Ticker ---
    df_engineered = load_data_for_ticker(ticker)
    if df_engineered is None:
        return

    # --- 3. Create Target Variable for this stock's data ---
    print(f"\nCreating target variable ('{TARGET_COLUMN}') for {ticker}...")
    if 'Close' not in df_engineered.columns:
        print("Error: 'Close' column missing. Cannot create target.")
        return
    df_engineered[TARGET_COLUMN] = df_engineered['Close'].shift(-1) - df_engineered['Close']
    print("Target created.")

    # --- 4. Drop NaN Rows ---
    initial_rows = len(df_engineered)
    df_complete = df_engineered.dropna()
    rows_dropped = initial_rows - len(df_complete)
    if df_complete.empty:
        print("Error: DataFrame empty after dropping NaNs.")
        return
    print(f"Dropped {rows_dropped} rows containing NaNs.")
    print(f"Final dataset size for evaluation: {len(df_complete)} rows.")

    # --- 5. Select Features (X) and Target (y) ---
    print(f"\nSelecting features and target for {ticker}...")
    available_columns = df_complete.columns.tolist()
    feature_columns_to_use = []
    missing_features = []
    for col in FEATURE_COLUMNS:
        if col in available_columns:
            feature_columns_to_use.append(col)
        else:
            missing_features.append(col)
            
    if missing_features:
        print(f"Error: The data for {ticker} is missing required feature columns used for training: {missing_features}")
        return
        
    if not feature_columns_to_use:
         print(f"Error: No valid features found in {ticker}'s data!")
         return
         
    X_eval = df_complete[feature_columns_to_use]
    y_eval = df_complete[TARGET_COLUMN] # Actual changes for this stock
    eval_index = X_eval.index # Keep index for comparison table
    print(f"Selected {X_eval.shape[1]} features.")

    # --- 6. Scale Features using LOADED Scaler ---
    print("\nScaling features using the pre-fitted scaler...")
    try:
        # IMPORTANT: Use transform() only, DO NOT use fit() or fit_transform() here!
        X_eval_scaled = scaler.transform(X_eval) 
        print("Features scaled.")
    except Exception as e:
        print(f"Error scaling features for {ticker}: {e}")
        print("Ensure the feature columns match those the scaler was trained on.")
        traceback.print_exc()
        return

    # --- 7. Make Predictions using LOADED Model ---
    print(f"\nMaking predictions for {ticker} using the generic model...")
    try:
        y_pred = model.predict(X_eval_scaled) # Predicted changes
        print("Predictions made.")
    except Exception as e:
        print(f"Error during prediction for {ticker}: {e}")
        traceback.print_exc()
        return

    # --- 8. Evaluate Performance on THIS Stock ---
    print(f"\nEvaluating generic model performance specifically on {ticker} data...")
    
    mae = mean_absolute_error(y_eval, y_pred)
    mse = mean_squared_error(y_eval, y_pred)
    rmse = mse**0.5 
    r2 = r2_score(y_eval, y_pred) 

    print(f"\n--- Evaluation Metrics for {ticker} (Predicting Price Change) ---")
    print(f"Mean Absolute Error (MAE):  {mae:.4f}") 
    print(f"Mean Squared Error (MSE):   {mse:.4f}")
    print(f"Root Mean Squared Err (RMSE):{rmse:.4f}") 
    print(f"R-squared (R²):           {r2:.4f}") 
    print("-------------------------------------------------------------")

    # --- Actual vs. Predicted Change Comparison ---
    print(f"\n--- Actual vs. Predicted Change for {ticker} (Sample) ---")
    comparison_df = pd.DataFrame({
        'Actual_Change': y_eval.values, 
        'Predicted_Change': y_pred
        }, index=eval_index) 
    comparison_df['Prediction_Error'] = comparison_df['Actual_Change'] - comparison_df['Predicted_Change']
    
    print("Head:")
    print(comparison_df.head())
    print("\nTail:")
    print(comparison_df.tail())
    print("-------------------------------------------------------------")
    
    print(f"\nEvaluation script for {ticker} finished.")

if __name__ == "__main__":
    main()
