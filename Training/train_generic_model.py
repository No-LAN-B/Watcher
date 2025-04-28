import pandas as pd
import numpy as np
import xgboost as xgb 
# Imports for tuning and time series CV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import glob 
import traceback 
from scipy.stats import randint, uniform # For defining parameter distributions

# --- Configuration ---
INPUT_FEATURE_DIR = "data/complete" 
MODEL_OUTPUT_DIR = "models" 
# Use the expanded feature list
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', #'Close', 
    'Volume', 'return', 'sma_10', 'sma_200', 'vol_10', 'Close_Lag_1',
    'RSI_14', 'MACD_line', 'MACD_hist', 'MACD_signal', 'ATR_14',
    'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Percent'
]
TARGET_COLUMN = 'Target_Change' 
TRAIN_TEST_SPLIT_RATIO = 0.8 

# --- Hyperparameter Tuning Configuration ---
N_ITER_SEARCH = 50  # Number of parameter settings that are sampled. Increase for more thorough search.
CV_SPLITS = 5       # Number of folds for TimeSeriesSplit cross-validation.

def load_and_combine_data(input_dir: str) -> pd.DataFrame | None:
    """Loads all '*_complete.csv' files and combines them."""
    all_files = glob.glob(os.path.join(input_dir, "*_complete.csv")) 
    if not all_files:
        print(f"Error: No '*_complete.csv' files found in '{input_dir}'.")
        return None
        
    df_list = []
    print(f"Found {len(all_files)} feature files. Loading...")
    for f in all_files:
        try:
            df = pd.read_csv(f, index_col='Date', parse_dates=True) 
            df_list.append(df)
            print(f"  Loaded {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"  Warning: Could not load or parse {f}. Error: {e}")
            
    if not df_list:
        print("Error: Failed to load any valid data.")
        return None
        
    combined_df = pd.concat(df_list)
    combined_df.sort_index(inplace=True) 
    print(f"\nCombined data has {len(combined_df)} total rows.")
    return combined_df

def main():
    """Loads data, tunes XGBoost model for PRICE CHANGE, evaluates best model, and saves."""
    
    # --- Steps 1-4: Load, Create Target, Drop NaNs, Select X/y ---
    # (Keep these steps the same as the previous version)
    df_engineered = load_and_combine_data(INPUT_FEATURE_DIR)
    if df_engineered is None: return
    
    print(f"\nCreating target variable ('{TARGET_COLUMN}')...")
    if 'Close' not in df_engineered.columns:
        print("Error: 'Close' column is missing.")
        return
    df_engineered[TARGET_COLUMN] = df_engineered['Close'].shift(-1) - df_engineered['Close'] 
    print(f"Target '{TARGET_COLUMN}' created.")

    initial_rows = len(df_engineered)
    df_complete = df_engineered.dropna()
    rows_dropped = initial_rows - len(df_complete)
    if df_complete.empty:
        print("Error: DataFrame empty after dropping NaNs.")
        return
    print(f"Dropped {rows_dropped} rows containing NaNs.")
    print(f"Final dataset size for training/testing: {len(df_complete)} rows.")

    print(f"\nSelecting features and target...")
    available_columns = df_complete.columns.tolist()
    feature_columns_to_use = []
    missing_features = []
    for col in FEATURE_COLUMNS:
        if col in available_columns:
            feature_columns_to_use.append(col)
        else:
            missing_features.append(col)
    if missing_features: print(f"Warning: Skipped missing features: {missing_features}")
    if not feature_columns_to_use: print("Error: No valid features found!"); return
        
    X = df_complete[feature_columns_to_use]
    y = df_complete[TARGET_COLUMN] 
    print(f"Selected {X.shape[1]} features.")
    print(f"Target is '{TARGET_COLUMN}'.")

    # --- 5. Split Data (Time Series Split) ---
    print(f"\nSplitting data...")
    split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    # Keep track of the test set index for later comparison
    test_index = X_test.index 
    print(f"  Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"  Testing data shape: {X_test.shape}, {y_test.shape}")

    # --- 6. Scale Features ---
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    # --- 7. Hyperparameter Tuning with RandomizedSearchCV ---
    print(f"\nStarting Hyperparameter Tuning (RandomizedSearchCV, {N_ITER_SEARCH} iterations, {CV_SPLITS}-fold TimeSeriesSplit)...")
    
    # Define the parameter distribution for RandomizedSearchCV
    # Using distributions (like randint, uniform) allows random sampling
    param_dist = {
        'n_estimators': randint(100, 500),              # Number of trees
        'learning_rate': uniform(0.01, 0.2),           # Learning rate (0.01 to 0.21)
        'max_depth': randint(3, 10),                   # Max depth of trees
        'subsample': uniform(0.6, 0.4),                # Subsample ratio (0.6 to 1.0)
        'colsample_bytree': uniform(0.6, 0.4),         # Feature subsample ratio per tree
        'gamma': uniform(0, 0.5),                      # Min loss reduction for split
        'reg_alpha': uniform(0, 1),                    # L1 regularization
        'reg_lambda': uniform(1, 4)                    # L2 regularization (often > L1)
    }

    # Initialize the base XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        n_jobs=-1
    )

    # Set up TimeSeriesSplit for cross-validation
    # gap=0 means no gap between train and test folds in CV
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=0) 

    # Set up RandomizedSearchCV
    # scoring='neg_mean_squared_error' -> tries to maximize this (minimize MSE)
    # verbose=2 -> prints progress updates
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_dist, 
        n_iter=N_ITER_SEARCH, 
        cv=tscv, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, # Use all cores for the search itself
        verbose=2, 
        random_state=42 # For reproducibility of the search
    )

    # Fit the RandomizedSearchCV object to the training data
    # This performs the search and finds the best parameters
    random_search.fit(X_train_scaled, y_train)

    print("\nHyperparameter tuning complete.")
    print(f"Best Parameters found: {random_search.best_params_}")
    print(f"Best Cross-validation Score (Negative MSE): {random_search.best_score_:.4f}")

    # Get the best model found by the search
    best_model = random_search.best_estimator_ 

    # --- 8. Evaluate Best Model ---
    print("\nEvaluating BEST model found on the held-out test data...")
    y_pred = best_model.predict(X_test_scaled) 
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5 
    r2 = r2_score(y_test, y_pred) 

    print("\n--- Evaluation Metrics (Predicting Price Change - Best Model) ---")
    print(f"Mean Absolute Error (MAE):  {mae:.4f}") 
    print(f"Mean Squared Error (MSE):   {mse:.4f}")
    print(f"Root Mean Squared Err (RMSE):{rmse:.4f}") 
    print(f"R-squared (R²):           {r2:.4f}") 
    print("-------------------------------------------------------------")

    # --- Actual vs. Predicted Change Comparison ---
    print("\n--- Actual vs. Predicted Change (Test Set Sample - Best Model) ---")
    comparison_df = pd.DataFrame({
        'Actual_Change': y_test.values, # Use .values to avoid potential index mismatch issues
        'Predicted_Change': y_pred
        }, index=test_index) # Use the saved test index
    comparison_df['Prediction_Error'] = comparison_df['Actual_Change'] - comparison_df['Predicted_Change']
    
    print("Head:")
    print(comparison_df.head())
    print("\nTail:")
    print(comparison_df.tail())
    print("-------------------------------------------------------------")
   
    # --- 9. Save Best Model and Scaler ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) 
    model_name = type(best_model).__name__.lower() # Get name from best model
    model_filename = f"best_{model_name}_predict_change.pkl" # Updated filename
    scaler_filename = f"scaler_for_{model_filename.replace('.pkl', '.pkl')}" # Match scaler to model
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, scaler_filename)
    
    print(f"\nSaving BEST trained model to: {model_path}")
    joblib.dump(best_model, model_path) # Save the best model
    
    print(f"Saving fitted scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    print("\nTraining script finished.")

if __name__ == "__main__":
    main()
