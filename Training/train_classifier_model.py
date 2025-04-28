import pandas as pd
import numpy as np
import xgboost as xgb 
# Import classification models and metrics
from sklearn.ensemble import RandomForestClassifier # Option to try RF later
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import glob 
import traceback 
from scipy.stats import randint, uniform 
import argparse # To specify the ticker

# --- Configuration ---
INPUT_FEATURE_DIR = "data/complete" 
MODEL_OUTPUT_DIR = "models" 
# Feature columns (same as before, likely excluding 'Close')
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', #'Close', 
    'Volume', 'return', 'sma_10', 'vol_10', 'Close_Lag_1',
    'RSI_14', 'MACD_line', 'MACD_hist', 'MACD_signal', 'ATR_14',
    'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Percent'
]
# ** CHANGE 1: Define the CLASSIFICATION target column name **
TARGET_COLUMN = 'Target_Direction' # 1 if price increases, 0 otherwise
TRAIN_TEST_SPLIT_RATIO = 0.8 

# --- Hyperparameter Tuning Configuration ---
N_ITER_SEARCH = 30  # Reduced slightly for potentially faster tuning initially
CV_SPLITS = 5       

def load_data_for_ticker(ticker: str) -> pd.DataFrame | None:
    """Loads the complete CSV file for a single ticker."""
    filepath = os.path.join(INPUT_FEATURE_DIR, f"{ticker}_complete.csv")
    if not os.path.exists(filepath):
        print(f"Error: Feature file not found for {ticker} at '{filepath}'.")
        print(f"Please ensure you have run 'ticker_script.py {ticker}'.")
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

def main(ticker_to_train: str):
    """Loads data for ONE ticker, trains classifier for PRICE DIRECTION, evaluates, and saves."""
    
    print(f"\n--- Training Classifier for Single Stock: {ticker_to_train} ---")
    
    # --- 1. Load Data for the Specific Ticker ---
    df_engineered = load_data_for_ticker(ticker_to_train)
    if df_engineered is None:
        return

    # --- 2. Create Target Variable (Direction: 1 for Up, 0 for Down/Flat) ---
    # ** CHANGE 2: Calculate the direction **
    print(f"\nCreating target variable ('{TARGET_COLUMN}')...")
    if 'Close' not in df_engineered.columns:
        print("Error: 'Close' column is missing. Cannot create target.")
        return
    price_change = df_engineered['Close'].shift(-1) - df_engineered['Close'] 
    df_engineered[TARGET_COLUMN] = (price_change > 0).astype(int) 
    print(f"Target '{TARGET_COLUMN}' created.")

    # --- 3. Drop NaN Rows ---
    initial_rows = len(df_engineered)
    df_complete = df_engineered.dropna()
    rows_dropped = initial_rows - len(df_complete)
    if df_complete.empty:
        print("Error: DataFrame empty after dropping NaNs.")
        return
    print(f"Dropped {rows_dropped} rows containing NaNs.")
    print(f"Final dataset size for training/testing: {len(df_complete)} rows.")

    # --- 4. Select Features (X) and Target (y) ---
    print(f"\nSelecting features and target...")
    available_columns = df_complete.columns.tolist()
    feature_columns_to_use = []
    missing_features = []
    for col in FEATURE_COLUMNS:
        if col in available_columns:
            feature_columns_to_use.append(col)
        else:
            missing_features.append(col)
            
    if missing_features:
        print(f"Warning: Skipped missing features: {missing_features}")
        
    if not feature_columns_to_use:
         print("Error: No valid features found for training!")
         return
        
    X = df_complete[feature_columns_to_use]
    y = df_complete[TARGET_COLUMN] # Target is now the direction (0 or 1)
    print(f"Selected {X.shape[1]} features: {feature_columns_to_use}")
    print(f"Target is '{TARGET_COLUMN}'.")
    print(f"Target value distribution:\n{y.value_counts(normalize=True)}") 

    # --- 5. Split Data (Time Series Split) ---
    print(f"\nSplitting data...")
    split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_index = X_test.index 
    print(f"  Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"  Testing data shape: {X_test.shape}, {y_test.shape}")

    # --- 6. Scale Features ---
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled.")

    # --- 7. Hyperparameter Tuning with RandomizedSearchCV for CLASSIFIER ---
    print(f"\nStarting Hyperparameter Tuning for XGBClassifier...")
    
    # Define the parameter distribution (can reuse from before or adjust)
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(1, 4) 
    }

    # ** CHANGE 3: Initialize XGBClassifier **
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss',       
        #use_label_encoder=False,     
        random_state=42, 
        n_jobs=-1
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS, gap=0) 

    # ** CHANGE 4: Use 'roc_auc' or 'accuracy' for scoring **
    random_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=param_dist, 
        n_iter=N_ITER_SEARCH, 
        cv=tscv, 
        scoring='roc_auc', # Tuning based on AUC
        n_jobs=-1, 
        verbose=1, 
        random_state=42 
    )

    random_search.fit(X_train_scaled, y_train)

    print("\nHyperparameter tuning complete.")
    print(f"Best Parameters found: {random_search.best_params_}")
    print(f"Best Cross-validation Score (AUC): {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_ 

    # --- 8. Evaluate Best CLASSIFIER Model ---
    print("\nEvaluating BEST model found on the held-out test data...")
    y_pred = best_model.predict(X_test_scaled) # Predicted directions (0 or 1)
    # Get probabilities for AUC calculation and potentially for display
    try:
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] # Probabilities for class 1 (Up)
    except AttributeError:
        print("Warning: Model does not support predict_proba. Cannot calculate AUC.")
        y_pred_proba = np.full(len(y_pred), np.nan) # Assign NaN if probabilities aren't available
        auc_score = float('nan')

    # --- Classification Metrics ---
    # ** CHANGE 5: Use classification metrics **
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Down/Flat', 'Up'], zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    try:
        # Ensure probabilities were successfully obtained before calculating AUC
        if not np.isnan(y_pred_proba).any():
             auc_score = roc_auc_score(y_test, y_pred_proba) 
        else:
             auc_score = float('nan')
    except ValueError: 
        auc_score = float('nan') 
        print("Warning: Could not calculate AUC score (likely only one class in test set).")

    print(f"\n--- Evaluation Metrics for {ticker_to_train} (Predicting Direction) ---")
    print(f"Accuracy:           {accuracy:.4f}") 
    print(f"AUC Score:          {auc_score:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print("          Down/Flat   Up")
    print(f"Down/Flat {conf_matrix[0,0]:<10} {conf_matrix[0,1]:<10}")
    print(f"Up        {conf_matrix[1,0]:<10} {conf_matrix[1,1]:<10}")
    print("-------------------------------------------------------------")

    # --- Actual vs. Predicted Direction Comparison ---
    # ** CHANGE 6: Update comparison DataFrame **
    print(f"\n--- Actual vs. Predicted Direction for {ticker_to_train} (Sample) ---")
    comparison_df = pd.DataFrame({
        'Actual_Direction': y_test.map({0: 'Down/Flat', 1: 'Up'}), 
        'Predicted_Direction': pd.Series(y_pred, index=test_index).map({0: 'Down/Flat', 1: 'Up'}),
        'Predicted_Prob_Up': y_pred_proba if not np.isnan(y_pred_proba).any() else 'N/A' # Show probability if available
        }, index=test_index) 
    
    print("Head:")
    print(comparison_df.head())
    print("\nTail:")
    print(comparison_df.tail())
    print("-------------------------------------------------------------")
   

    
    # --- 9. Save Best Model and Scaler ---
    # ** CHANGE 7: Update saved model filename **
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) 
    model_name = type(best_model).__name__.lower() # e.g., 'xgbclassifier'
    # Include ticker in filename for individual models
    model_filename = f"{ticker_to_train}_{model_name}_predict_direction.pkl" 
    scaler_filename = f"scaler_for_{ticker_to_train}_{model_name}_predict_direction.pkl" 
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, scaler_filename)
    
    print(f"\nSaving BEST trained model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    print(f"Saving fitted scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    
    print(f"\nTraining script for {ticker_to_train} finished.")

if __name__ == "__main__":
    # Add argument parsing to specify the ticker
    parser = argparse.ArgumentParser(description="Train a classifier model for predicting stock direction for a specific ticker.")
    parser.add_argument("--ticker", type=str, required=True, help="The stock ticker symbol to train (e.g., AAPL)")
    args = parser.parse_args()
    
    main(ticker_to_train=args.ticker.strip().upper())
