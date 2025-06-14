import pandas as pd
import numpy as np
import joblib
# Corrected import: r2_score is in sklearn.metrics, not sklearn.base
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the filename for the trained model
model_filename = 'full_income_prediction_pipeline.joblib'
data_filename = 'processed_dataset_400.csv'
id_col = 'id'
target_col = 'target_income'

print(f"--- Loading data from {data_filename} ---")
try:
    df_validation = pd.read_csv(data_filename)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: {data_filename} not found. Please ensure the file is in the same directory.")
    exit()

print(f"--- Loading trained model from {model_filename} ---")
try:
    # Load the full pipeline which includes preprocessing and the regressor
    loaded_pipeline = joblib.load(model_filename)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Trained model '{model_filename}' not found. Please run income.py first to train and save the model.")
    exit()

# Separate features for prediction. Ensure this matches the features used during training.
# The pipeline expects the same columns as were passed to X during training (excluding id and target).
# We need to recreate the same engineered features in this validation script as well.
# This ensures the loaded_pipeline processes the data correctly.

# --- Re-apply Feature Engineering steps for consistency ---
# This part must mirror the feature engineering done in the training script (income.py)
df_validation_fe = df_validation.copy()

# Example 1: Ratio of balance to credit limit
balance_credit_map = {
    'balance_credit_ratio_1': ('var_0', 'var_2'),
    'balance_credit_ratio_2': ('var_1', 'var_3'),
    'balance_credit_ratio_3': ('var_4', 'var_5'),
}

for new_col, (balance_col, credit_col) in balance_credit_map.items():
    if balance_col in df_validation_fe.columns and credit_col in df_validation_fe.columns:
        # Add epsilon to avoid division by zero; handle potential NaN results from division by NaN
        df_validation_fe[new_col] = df_validation_fe[balance_col] / (df_validation_fe[credit_col] + 1e-6)

# Example 2: Total loan amount
actual_loan_amt_cols = [
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31',
    'var_36', 'var_39', 'var_42', 'var_65', 'var_72'
]
existing_loan_cols = [col for col in actual_loan_amt_cols if col in df_validation_fe.columns]
if existing_loan_cols:
    df_validation_fe['total_loan_amount'] = df_validation_fe[existing_loan_cols].sum(axis=1)

# Example 3: Total inquiries
actual_inquiries_cols = [
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71'
]
existing_inquiries_cols = [col for col in actual_inquiries_cols if col in df_validation_fe.columns]
if existing_inquiries_cols:
    df_validation_fe['total_inquiries_all'] = df_validation_fe[existing_inquiries_cols].sum(axis=1)

# Example 4: Age-Credit Score Interaction
if 'age' in df_validation_fe.columns and 'var_32' in df_validation_fe.columns:
    df_validation_fe['age_credit_score_interaction'] = df_validation_fe['age'] * df_validation_fe['var_32']

# Prepare features (X_predict) for the loaded pipeline
# Ensure these columns match the order and presence of columns used during training
# The pipeline itself handles imputation, scaling, and one-hot encoding,
# so we just need to provide the raw features (including newly engineered ones)
# that were present in X when the model was trained.
X_predict = df_validation_fe.drop(columns=[target_col, id_col], errors='ignore')

print("--- Making predictions ---")
# Use the loaded pipeline to make predictions
predictions = loaded_pipeline.predict(X_predict)

# Add predictions to the original DataFrame (or a new one) for review
df_validation['predicted_target_income'] = predictions

print("\nSample of Predictions from the Loaded Model (first 10 rows):")
print(df_validation[[id_col, target_col, 'predicted_target_income']].head(10))

# Optional: Calculate metrics on the full dataset if you have the actual target column
if target_col in df_validation.columns:
    full_mae = mean_absolute_error(df_validation[target_col], predictions)
    full_rmse = np.sqrt(mean_squared_error(df_validation[target_col], predictions))
    full_r2 = r2_score(df_validation[target_col], predictions)
    print(f"\nMetrics on the full dataset:")
    print(f"Mean Absolute Error (MAE): {full_mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {full_rmse:.2f}")
    print(f"R-squared (R2): {full_r2:.2f}")

print("\n--- Validation/Prediction using loaded model complete ---")
