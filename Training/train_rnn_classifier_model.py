# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import os
import glob
import traceback
import argparse

# --- Deep Learning Imports ---
import tensorflow as tf
# Make TF log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from keras import optimizers # Keep for potential customization

# --- Scikit-learn Imports (for preprocessing and evaluation) ---
from sklearn.preprocessing import MinMaxScaler # Changed from StandardScaler
from sklearn.model_selection import train_test_split # Using simple split for demo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# --- Configuration ---
INPUT_FEATURE_DIR = "data/complete"
MODEL_OUTPUT_DIR = "models"
# Feature columns (ensure these are present in your _complete.csv)
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', #'Close', # Exclude if used only for target
    'Volume', 'return', 'sma_10', 'vol_10', 'Close_Lag_1',
    'RSI_14', 'MACD_line', 'MACD_hist', 'MACD_signal', 'ATR_14',
    'BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Width', 'BB_Percent'
]
TARGET_COLUMN = 'Target_Direction' # 1 if price increases, 0 otherwise
TRAIN_TEST_SPLIT_RATIO = 0.8
# --- LSTM Configuration ---
TIME_STEPS = 10 # How many past days of features to use for predicting the next day
BATCH_SIZE = 64
EPOCHS = 100

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

# --- NEW: Function to create sequences ---
def create_sequences(X_data, y_data, time_steps=1):
    """
    Creates sequences for LSTM input.
    Input:
        X_data (np.array): Feature data (samples, features)
        y_data (np.array): Target data (samples,)
        time_steps (int): Number of past time steps to use for each sequence.
    Output:
        Xs (np.array): Sequential feature data (samples, time_steps, features)
        ys (np.array): Corresponding target data (samples,)
    """
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        # Get 'time_steps' worth of features ending at index i+time_steps-1
        v = X_data[i:(i + time_steps)]
        Xs.append(v)
        # The target corresponds to the day *after* the sequence ends
        ys.append(y_data[i + time_steps])
    return np.array(Xs), np.array(ys)


def build_single_lstm_model(input_shape):
    """Builds the single LSTM model based on the Stanford example."""
    model = Sequential()
    # Input shape is (time_steps, num_features)
    model.add(Input(shape=input_shape)) # Use Input layer for clarity
    model.add(LSTM(units=128)) # Removed input_shape here, inferred by Input layer
    model.add(Dropout(0.5))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid')) # Sigmoid for binary classification

    # Using Adam optimizer as in the example
    optimizer = optimizers.Adam(learning_rate=0.001) # Corrected API
    model.compile(loss='binary_crossentropy', # Loss for binary classification
                  optimizer=optimizer,
                  metrics=['accuracy']) # Track accuracy during training
    print("Model Summary:")
    model.summary()
    return model

# --- You could add functions for build_multi_lstm_model and build_gru_model here ---
# def build_multi_lstm_model(input_shape): ...
# def build_gru_model(input_shape): ...


def main(ticker_to_train: str):
    """Loads data, prepares sequences, trains LSTM classifier, evaluates, and saves."""

    print(f"\n--- Training RNN (LSTM) Classifier for Single Stock: {ticker_to_train} ---")

    # --- 1. Load Data ---
    df_engineered = load_data_for_ticker(ticker_to_train)
    if df_engineered is None:
        return

    # --- 2. Create Target Variable ---
    print(f"\nCreating target variable ('{TARGET_COLUMN}')...")
    if 'Close' not in df_engineered.columns:
        print("Error: 'Close' column is missing. Cannot create target.")
        return
    # Predict NEXT day's direction based on previous day's close
    price_change = df_engineered['Close'].shift(-1) - df_engineered['Close']
    df_engineered[TARGET_COLUMN] = (price_change > 0).astype(int)
    print(f"Target '{TARGET_COLUMN}' created.")

    # --- 3. Drop NaN Rows (Especially crucial after shift(-1)) ---
    initial_rows = len(df_engineered)
    df_complete = df_engineered.dropna()
    rows_dropped = initial_rows - len(df_complete)
    if df_complete.empty:
        print("Error: DataFrame empty after dropping NaNs.")
        return
    print(f"Dropped {rows_dropped} rows containing NaNs.")
    print(f"Final dataset size before sequencing: {len(df_complete)} rows.")


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
    y = df_complete[TARGET_COLUMN]
    print(f"Selected {X.shape[1]} features: {feature_columns_to_use}")
    print(f"Target is '{TARGET_COLUMN}'.")
    print(f"Target value distribution:\n{y.value_counts(normalize=True)}")


    # --- 5. Split Data Chronologically BEFORE Scaling/Sequencing ---
    print(f"\nSplitting data chronologically...")
    split_index = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
    X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
    y_train_raw, y_test_raw = y.iloc[:split_index], y.iloc[split_index:]
    test_dates = X_test_raw.index[TIME_STEPS:] # Adjust test dates for sequence loss

    print(f"  Raw training data shape: {X_train_raw.shape}, {y_train_raw.shape}")
    print(f"  Raw testing data shape: {X_test_raw.shape}, {y_test_raw.shape}")

    # --- 6. Scale Features (using MinMaxScaler) ---
    print("\nScaling features (MinMaxScaler)...")
    # Scale features based ONLY on the training set to prevent data leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    print("Features scaled.")

    # --- 7. Create Sequences ---
    print(f"\nCreating sequences with TIME_STEPS={TIME_STEPS}...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, TIME_STEPS)

    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
        print(f"Error: Not enough data to create sequences with TIME_STEPS={TIME_STEPS}.")
        print(f"Need at least {TIME_STEPS + 1} data points for train and test splits respectively.")
        return

    print(f"  Training sequences shape: {X_train_seq.shape}, {y_train_seq.shape}")
    print(f"  Testing sequences shape: {X_test_seq.shape}, {y_test_seq.shape}")
    print(f"  Test dates adjusted, starting from: {test_dates[0] if len(test_dates) > 0 else 'N/A'}")


    # --- 8. Build and Train the LSTM Model ---
    print("\nBuilding LSTM model...")
    # The input shape for the model is (time_steps, num_features)
    model_input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    # --- CHOOSE YOUR MODEL HERE ---
    model = build_single_lstm_model(model_input_shape)
    # model = build_multi_lstm_model(model_input_shape) # Or use the multi-layer version
    # model = build_gru_model(model_input_shape) # Or use the GRU version

    print("\nStarting model training...")
    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', # Stop based on validation loss
                                   patience=10,       # Num epochs with no improvement
                                   mode='min',
                                   restore_best_weights=True, # Restore weights from best epoch
                                   verbose=1)

    # Train the model (using test set as validation for simplicity here, consider separate val set)
    history = model.fit(X_train_seq, y_train_seq,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_test_seq, y_test_seq), # Use test set for validation loss monitoring
                        callbacks=[early_stopping],
                        verbose=1) # Set to 1 or 2 for progress updates

    print("Model training complete.")

    # --- 9. Evaluate Model ---
    print("\nEvaluating model on the held-out test data...")
    # Evaluate returns loss and metrics (accuracy in this case)
    loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)

    # Get predictions (probabilities for class 1)
    y_pred_proba = model.predict(X_test_seq).flatten() # Flatten to 1D array
    # Convert probabilities to class labels (0 or 1) based on a 0.5 threshold
    y_pred = (y_pred_proba > 0.5).astype(int)

    # --- Classification Metrics ---
    report = classification_report(y_test_seq, y_pred, target_names=['Down/Flat', 'Up'], zero_division=0)
    conf_matrix = confusion_matrix(y_test_seq, y_pred)
    try:
        auc_score = roc_auc_score(y_test_seq, y_pred_proba)
    except ValueError:
        auc_score = float('nan')
        print("Warning: Could not calculate AUC score (likely only one class in test set sequences).")


    print(f"\n--- Evaluation Metrics for {ticker_to_train} (LSTM - Predicting Direction) ---")
    print(f"Test Loss:          {loss:.4f}")
    print(f"Test Accuracy:      {accuracy:.4f}") # From model.evaluate
    # print(f"Accuracy (Manual):  {accuracy_score(y_test_seq, y_pred):.4f}") # Should be same as above
    print(f"AUC Score:          {auc_score:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print("          Down/Flat   Up")
    print(f"Down/Flat {conf_matrix[0,0]:<10} {conf_matrix[0,1]:<10}")
    print(f"Up        {conf_matrix[1,0]:<10} {conf_matrix[1,1]:<10}")
    print("-------------------------------------------------------------")

    # --- Actual vs. Predicted Comparison (Aligned with Sequences) ---
    print(f"\n--- Actual vs. Predicted Direction for {ticker_to_train} (Sample) ---")
    if len(test_dates) == len(y_test_seq):
        comparison_df = pd.DataFrame({
            'Actual_Direction': pd.Series(y_test_seq, index=test_dates).map({0: 'Down/Flat', 1: 'Up'}),
            'Predicted_Direction': pd.Series(y_pred, index=test_dates).map({0: 'Down/Flat', 1: 'Up'}),
            'Predicted_Prob_Up': pd.Series(y_pred_proba, index=test_dates)
        })
        print("Head:")
        print(comparison_df.head())
        print("\nTail:")
        print(comparison_df.tail())
    else:
        print("Warning: Length mismatch between test dates and sequence predictions. Skipping comparison table.")
    print("-------------------------------------------------------------")


    # --- 10. Save Model and Scaler ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_type = "lstm_single" # Change if using multi-LSTM or GRU
    model_filename = f"{ticker_to_train}_{model_type}_predict_direction.keras" # Use .keras format
    scaler_filename = f"scaler_for_{ticker_to_train}_{model_type}_predict_direction.pkl"
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)
    scaler_path = os.path.join(MODEL_OUTPUT_DIR, scaler_filename)

    print(f"\nSaving trained Keras model to: {model_path}")
    model.save(model_path) # Use Keras save method

    print(f"Saving fitted scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path) # Save scaler as before

    print(f"\nRNN training script for {ticker_to_train} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RNN (LSTM/GRU) classifier model for predicting stock direction.")
    parser.add_argument("--ticker", type=str, required=True, help="The stock ticker symbol to train (e.g., AAPL)")
    parser.add_argument("--timesteps", type=int, default=TIME_STEPS, help="Number of past time steps to use for sequences")
    # Add arguments for epochs, batch size etc. if needed
    args = parser.parse_args()

    # Update global TIME_STEPS if provided via command line
    TIME_STEPS = args.timesteps

    main(ticker_to_train=args.ticker.strip().upper())