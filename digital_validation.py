import pandas as pd
import numpy as np
import joblib # Import joblib for loading models
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score # Import evaluation metrics
from sklearn.base import BaseEstimator, TransformerMixin # For TargetEncoder

# --- Custom TargetEncoder for categorical features (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
        self.global_mean = y.mean()

        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.cols)
            else:
                print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
                X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        for col in self.cols:
            if col not in X_df.columns:
                print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
                continue
            means = y.groupby(X_df[col]).mean()
            counts = y.groupby(X_df[col]).count()
            smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
            self.mapping[col] = smoothed_means.to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_transformed = pd.DataFrame(X, columns=self.cols)
            else:
                print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
                X_transformed = pd.DataFrame(X)
        else:
            X_transformed = X.copy()

        for col in self.cols:
            if col in self.mapping:
                X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
            else:
                X_transformed[col] = np.full(len(X_transformed), self.global_mean)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return self.cols

# --- Define Consistent Feature and Target Lists (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET]

categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]

device_features_candidate = [
    'device_model',
    'device_category',
    'platform',
    'device_manufacturer',
    'var_32',
    'var_0', 'var_2', 'var_6', 'var_9',
    'var_15',
    'var_62',
    'var_10',
    'var_11',
    'var_24',
    'var_25',
]

# --- 1. Load the trained Digital Footprint / Device Usage model ---
try:
    device_model_pipeline = joblib.load('device_model_pipeline.pkl')
    print("Digital Footprint / Device Usage model loaded successfully from 'device_model_pipeline.pkl'.")
except FileNotFoundError:
    print("Error: 'device_model_pipeline.pkl' not found.")
    print("Please ensure you have run 'digital.py' first to train and save the model.")
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
device_features_for_prediction = [col for col in device_features_candidate if col in df_validation.columns and col not in EXCLUDE_COLS]

X_validation = df_validation[device_features_for_prediction].copy()
y_validation_true = df_validation[TARGET].copy() # Get the true target values for evaluation


# --- 4. Generate 'device_income' predictions for the validation dataset ---
print("\nGenerating 'device_income' predictions for the validation dataset...")
y_validation_pred = device_model_pipeline.predict(X_validation)

# Ensure predictions are non-negative
y_validation_pred[y_validation_pred < 0] = 0
df_validation['device_income'] = y_validation_pred # Assign predictions to the DataFrame


# --- 5. Evaluate the model on the validation set ---
print("\nDigital Footprint / Device Usage Model Evaluation on Validation Set:")
validation_mae = mean_absolute_error(y_validation_true, y_validation_pred)
validation_r2 = r2_score(y_validation_true, y_validation_pred)

print(f"Mean Absolute Error (MAE) on Validation Set: ${validation_mae:,.2f}")
print(f"R-squared (R2) Score on Validation Set: {validation_r2:.4f}")


# --- 6. Save the updated validation dataset ---
print("\nValidation DataFrame with 'device_income' column:")
print(df_validation[['id', 'target_income', 'device_income']].head())
print(f"Number of rows with 'device_income' predictions: {df_validation['device_income'].count()}")

output_validation_filename = 'processed_dataset_400_with_device_income.csv'
df_validation.to_csv(output_validation_filename, index=False)
print(f"\nUpdated validation dataset saved to '{output_validation_filename}'")

print("\nDigital Footprint / Device Usage validation process complete.")
