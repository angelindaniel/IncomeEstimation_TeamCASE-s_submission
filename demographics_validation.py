import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score # Import evaluation metrics
# No need for lightgbm import if just predicting, but keeping sklearn parts for pipeline construction

# --- Custom Feature Engineering Function (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
def create_demographic_engineered_features(X):
    X_engineered = X.copy()
    if 'age' in X_engineered.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
        X_engineered['age_bin'] = pd.cut(
            X_engineered['age'],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        ).astype(object)
    if 'age' in X_engineered.columns and 'var_32' in X_engineered.columns:
        X_engineered['age_x_credit_score'] = X_engineered['age'] * X_engineered['var_32']
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

demographic_features_candidate = [
    'age', 'gender', 'marital_status', 'residence_ownership',
    'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_32',
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
    'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
    'var_34', 'var_35', 'var_38', 'var_59', 'var_68',
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
    'var_39', 'var_42', 'var_65', 'var_72',
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
    'var_69', 'var_70', 'var_73',
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
    'var_74', 'var_75',
    'age_bin', 'age_x_credit_score'
]


# --- 1. Load the trained Demographic model ---
try:
    demographic_model_pipeline = joblib.load('demographic_model_pipeline.pkl')
    print("Demographic model loaded successfully from 'demographic_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'demographic_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'demographics.py' first to train and save the model.")
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
demographic_features_for_prediction = [col for col in demographic_features_candidate if col in df_validation.columns and col not in EXCLUDE_COLS]

X_validation = df_validation[demographic_features_for_prediction].copy()
y_validation_true = df_validation[TARGET].copy() # Get the true target values for evaluation


# --- 4. Generate 'demographic_income' predictions for the validation dataset ---
print("\nGenerating 'demographic_income' predictions for the validation dataset...")
y_validation_pred = demographic_model_pipeline.predict(X_validation)

# Ensure predictions are non-negative
y_validation_pred[y_validation_pred < 0] = 0
df_validation['demographic_income'] = y_validation_pred # Assign predictions to the DataFrame


# --- 5. Evaluate the model on the validation set ---
print("\nDemographic Model Evaluation on Validation Set:")
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
validation_r2 = r2_score(y_validation_true, y_validation_pred)

print(f"Mean Absolute Error (MAE) on Validation Set: ${validation_mae:,.2f}")
print(f"R-squared (R2) Score on Validation Set: {validation_r2:.4f}")


# --- 6. Save the updated validation dataset ---
print("\nValidation DataFrame with 'demographic_income' column:")
print(df_validation[['id', 'target_income', 'demographic_income']].head())
print(f"Number of rows with 'demographic_income' predictions: {df_validation['demographic_income'].count()}")

output_validation_filename = 'processed_dataset_400_with_demographic_income.csv'
df_validation.to_csv(output_validation_filename, index=False)
print(f"\nUpdated validation dataset saved to '{output_validation_filename}'")

print("\nDemographic validation process complete.")
