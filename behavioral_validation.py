import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score # Import evaluation metrics

# --- Define Consistent Feature and Target Lists (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET]

categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

behavioral_features_candidate = [
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', 'var_34', 'var_35',
    'var_38', 'var_59', 'var_68',
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', 'var_26', 'var_27',
    'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', 'var_39', 'var_42',
    'var_65', 'var_72',
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67', 'var_69', 'var_70', 'var_73',
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',
    'var_32',
    'var_74', 'var_75'
]

# --- 1. Load the trained Behavioral model ---
try:
    behavioral_model_pipeline = joblib.load('behavioral_model_pipeline.pkl')
    print("Behavioral model loaded successfully from 'behavioral_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'behavioral_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'behavioral.py' first to train and save the model.")
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
behavioral_features_for_prediction = [col for col in behavioral_features_candidate if col in df_validation.columns and col not in EXCLUDE_COLS]

X_validation = df_validation[behavioral_features_for_prediction].copy()
y_validation_true = df_validation[TARGET].copy() # Get the true target values for evaluation


# --- 4. Generate 'behavioral_income' predictions for the validation dataset ---
print("\nGenerating 'behavioral_income' predictions for the validation dataset...")
y_validation_pred = behavioral_model_pipeline.predict(X_validation)

# Ensure predictions are non-negative
y_validation_pred[y_validation_pred < 0] = 0
df_validation['behavioral_income'] = y_validation_pred # Assign predictions to the DataFrame


# --- 5. Evaluate the model on the validation set ---
print("\nBehavioral Model Evaluation on Validation Set:")
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
validation_r2 = r2_score(y_validation_true, y_validation_pred)

print(f"Mean Absolute Error (MAE) on Validation Set: ${validation_mae:,.2f}")
print(f"R-squared (R2) Score on Validation Set: {validation_r2:.4f}")


# --- 6. Save the updated validation dataset ---
print("\nValidation DataFrame with 'behavioral_income' column:")
print(df_validation[['id', 'target_income', 'behavioral_income']].head())
print(f"Number of rows with 'behavioral_income' predictions: {df_validation['behavioral_income'].count()}")

output_validation_filename = 'processed_dataset_400_with_behavioral_income.csv'
df_validation.to_csv(output_validation_filename, index=False)
print(f"\nUpdated validation dataset saved to '{output_validation_filename}'")

print("\nBehavioral validation process complete.")
