import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score # Import evaluation metrics

# --- Custom Feature Engineering Function for Location Model (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
def create_location_engineered_features(X):
    X_engineered = X.copy()
    
    balance_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'balance' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    credit_limit_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'credit_limit' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    loan_amt_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'loan_amt' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    emi_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'total_emi' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    repayment_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'repayment' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    inquiry_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and ('inquires' in col or 'inquiries' in col) and pd.api.types.is_numeric_dtype(X_engineered[col])]
    total_loans_cols_in_X = [col for col in X_engineered.columns if col.startswith('var_') and 'total_loans' in col and pd.api.types.is_numeric_dtype(X_engineered[col])]
    
    if balance_cols_in_X:
        X_engineered['total_balance'] = X_engineered[balance_cols_in_X].sum(axis=1, skipna=True)
    if credit_limit_cols_in_X:
        X_engineered['total_credit_limit'] = X_engineered[credit_limit_cols_in_X].sum(axis=1, skipna=True)
    if loan_amt_cols_in_X:
        X_engineered['total_loan_amount'] = X_engineered[loan_amt_cols_in_X].sum(axis=1, skipna=True)
    if emi_cols_in_X:
        X_engineered['total_emi_sum'] = X_engineered[emi_cols_in_X].sum(axis=1, skipna=True)
    if repayment_cols_in_X:
        X_engineered['total_repayment_sum'] = X_engineered[repayment_cols_in_X].sum(axis=1, skipna=True)
    if inquiry_cols_in_X:
        X_engineered['total_inquiries_count'] = X_engineered[inquiry_cols_in_X].sum(axis=1, skipna=True)
    if total_loans_cols_in_X:
        X_engineered['total_loans_count'] = X_engineered[total_loans_cols_in_X].sum(axis=1, skipna=True)
    
    if 'total_balance' in X_engineered.columns and 'total_credit_limit' in X_engineered.columns:
        epsilon = 1e-6 
        X_engineered['credit_utilization_ratio'] = X_engineered['total_balance'] / (X_engineered['total_credit_limit'] + epsilon)
        X_engineered['credit_utilization_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X_engineered

# --- Define Consistent Feature and Target Lists (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET]

categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

location_features_candidate = [
    'age', 'gender', 'marital_status', 'residence_ownership',
    'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_32',
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
    'var_34', 'var_35', 'var_38', 'var_59', 'var_68',
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
    'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
    'var_39', 'var_42', 'var_65', 'var_72',
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
    'var_69', 'var_70', 'var_73',
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
]


# --- 1. Load the trained Location model ---
try:
    location_model_pipeline = joblib.load('location_model_pipeline.pkl')
    print("Location model loaded successfully from 'location_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'location_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'location.py' first to train and save the model.")
    exit()

# --- 2. Load the validation dataset ---
try:
    df_validation = pd.read_csv('processed_dataset_400.csv')
    print("\nValidation dataset 'processed_dataset_400.csv' loaded successfully.")
    print("Validation DataFrame head:")
    print(df_validation.head())
    print("\nValidation DataFrame Info (before type explicit conversion):")
    df_validation.info()

    # Apply same explicit type conversion as training script
    for col in categorical_cols_to_convert:
        if col in df_validation.columns:
            df_validation[col] = df_validation[col].astype(str).replace('nan', np.nan)
    print("\nValidation DataFrame Info (after explicit type conversion):")
    df_validation.info()

except FileNotFoundError:
    print("Error: 'processed_dataset_400.csv' not found.")
    print("Please ensure the validation dataset is in the same directory as this script.")
    exit()


# --- 3. Prepare features and target for prediction and evaluation ---
# Filter features to ensure they exist in the validation DataFrame and are not excluded.
# This must match the features the model was trained on.
location_features_for_prediction = [col for col in location_features_candidate if col in df_validation.columns and col not in EXCLUDE_COLS]

X_validation = df_validation[location_features_for_prediction].copy()
y_validation_true = df_validation[TARGET].copy() # Get the true target values for evaluation


# --- 4. Generate 'location_income' predictions for the validation dataset ---
print("\nGenerating 'location_income' predictions for the validation dataset...")
y_validation_pred = location_model_pipeline.predict(X_validation)

# Ensure predictions are non-negative
y_validation_pred[y_validation_pred < 0] = 0
df_validation['location_income'] = y_validation_pred # Assign predictions to the DataFrame


# --- 5. Evaluate the model on the validation set ---
print("\nLocation Model Evaluation on Validation Set:")
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
validation_r2 = r2_score(y_validation_true, y_validation_pred)

print(f"Mean Absolute Error (MAE) on Validation Set: ${validation_mae:,.2f}")
print(f"R-squared (R2) Score on Validation Set: {validation_r2:.4f}")


# --- 6. Save the updated validation dataset ---
print("\nValidation DataFrame with 'location_income' column:")
print(df_validation[['id', 'target_income', 'location_income']].head())
print(f"Number of rows with 'location_income' predictions: {df_validation['location_income'].count()}")

output_validation_filename = 'processed_dataset_400_with_location_income.csv'
df_validation.to_csv(output_validation_filename, index=False)
print(f"\nUpdated validation dataset saved to '{output_validation_filename}'")

print("\nLocation validation process complete.")
