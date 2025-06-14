import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # For saving the model
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom TargetEncoder for categorical features ---
# This class implements target encoding to prevent data leakage by fitting on training data only.
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None

    def fit(self, X, y): # y is now directly passed and used
        if y is None:
            raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
        self.global_mean = y.mean()

        # Ensure 'X' is a DataFrame to handle column names
        if not isinstance(X, pd.DataFrame):
            # If X is a numpy array, reconstruct it with expected column names for grouping
            if self.cols and len(self.cols) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.cols)
            else:
                # If we can't infer column names, proceed but warn
                print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
                X_df = pd.DataFrame(X) # Proceed with generic column names if no self.cols
        else:
            X_df = X.copy() # Work on a copy

        for col in self.cols:
            if col not in X_df.columns:
                print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
                continue

            # Calculate means for each category by grouping X_df[col] and applying to y
            means = y.groupby(X_df[col]).mean()
            counts = y.groupby(X_df[col]).count()

            # Apply smoothing
            smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
            self.mapping[col] = smoothed_means.to_dict()
        return self

    def transform(self, X):
        # Ensure 'X' is a DataFrame to handle column names
        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_transformed = pd.DataFrame(X, columns=self.cols)
            else:
                print("Warning: X is not a DataFrame and column names cannot be inferred. Proceeding without specific column name checks in TargetEncoder.")
                X_transformed = pd.DataFrame(X) # Proceed with generic column names if no self.cols
        else:
            X_transformed = X.copy() # Work on a copy

        for col in self.cols:
            if col in self.mapping:
                # Fill missing values in X_transformed[col] with np.nan before mapping.
                # Then map using the learned mapping. For unseen categories, or after mapping
                # if there are still NaNs (e.g., original NaNs), fill with global mean.
                X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
            else:
                # If column was not in mapping during fit (e.g., from a new data point),
                # fill its values with the global mean.
                X_transformed[col] = np.full(len(X_transformed), self.global_mean)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # For TargetEncoder, output features are just the input columns, but numerical.
        return self.cols


# --- 1. Load the actual dataset ---
# This script strictly relies on 'processed_dataset.csv' being present.
# If not found, a FileNotFoundError will be raised.
df = pd.read_csv('processed_dataset.csv')
print("Dataset 'processed_dataset.csv' loaded successfully.")
print("Original DataFrame head:")
print(df.head())
print("\nDataFrame Info (before type explicit conversion):")
df.info()

# --- Explicitly convert all relevant categorical columns to string type ---
# This is crucial to ensure they are treated as categorical strings consistently.
categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75'
]
for col in categorical_cols_to_convert:
    if col in df.columns:
        # Convert to string, then replace string 'nan' (if any) with actual np.nan
        df[col] = df[col].astype(str).replace('nan', np.nan)
print("\nDataFrame Info (after explicit type conversion for categorical columns):")
df.info()


# --- 2. Define Features for the Digital Footprint / Device Usage Slave Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# Define features for the Digital Footprint / Device Usage Model.
# Now includes selected financial features that might correlate with device usage patterns.
device_features_candidate = [
    'device_model',
    'device_category',
    'platform',
    'device_manufacturer',
    'var_32', # credit_score
    'var_0', 'var_2', 'var_6', 'var_9', # Example balance, credit_limit, loan_amt, emi
    'var_15', # Example inquiry count
    'var_62', # Example total loan recent
    # NEWLY ADDED FEATURES for potentially better accuracy
    'var_10', # active_credit_limit_1
    'var_11', # credit_limit_recent_1
    'var_24', # loan_amt_recent
    'var_25', # total_inquiries_recent
]

# Filter features to ensure they exist in the DataFrame and are not excluded
device_features = [col for col in device_features_candidate if col in df.columns and col not in EXCLUDE_COLS]

# Separate numerical and categorical features for the preprocessor
device_numerical_features = [col for col in device_features if pd.api.types.is_numeric_dtype(df[col])]
# Only device-specific categories are target encoded, not the newly added numerical financial features.
device_categorical_features = [col for col in ['device_model', 'device_category', 'platform', 'device_manufacturer'] if col in device_features]


print(f"\nDevice Numerical Features: {device_numerical_features}")
print(f"Device Categorical Features: {device_categorical_features}")


# --- 3. Split Data into Training and Testing Sets ---
# Drop rows where TARGET is NaN if any exist (critical for training a regression model)
df_clean = df.dropna(subset=[TARGET]).copy()

# X will include only the selected device-related features (and some financial ones now)
X_device = df_clean[device_features].copy()
y = df_clean[TARGET]

X_train_device, X_test_device, y_train, y_test = train_test_split(
    X_device, y, test_size=0.2, random_state=42
)

print(f"\nDevice Training data shape: {X_train_device.shape}")
print(f"Device Testing data shape: {X_test_device.shape}")


# --- 4. Preprocessing Pipeline for Digital Footprint / Device Usage Model ---

# Numerical pipeline: Impute with median, then scale
device_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Use Custom TargetEncoder
device_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute NaNs with most frequent before encoding
    # Pass 'cols' to the TargetEncoder.
    ('target_encoder', TargetEncoder(cols=device_categorical_features, smoothing=5.0)) # Increased smoothing slightly
])


# Combine pipelines using ColumnTransformer for device features
device_preprocessor = ColumnTransformer(
    transformers=[
        ('num', device_numerical_pipeline, device_numerical_features),
        ('cat', device_categorical_pipeline, device_categorical_features)
    ],
    remainder='drop',
    verbose_feature_names_out=False # Suppress warning about feature names in LightGBM
)

# --- 5. Build and Train the Digital Footprint / Device Usage Slave Model Pipeline ---
device_model_pipeline = Pipeline([
    ('preprocessor', device_preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=2000, learning_rate=0.02, num_leaves=60)) # Tuned HP
])

print("\nTraining the Digital Footprint / Device Usage Slave Model...")
# The y_train is passed to the fit method of the pipeline, which then correctly
# routes it to the TargetEncoder's fit method via the ColumnTransformer.
device_model_pipeline.fit(X_train_device, y_train)
print("Digital Footprint / Device Usage Slave Model training complete.")

# --- ADDED LINE: Save the trained model ---
joblib.dump(device_model_pipeline, 'device_model_pipeline.pkl')
print("\nDigital Footprint / Device Usage model saved to 'device_model_pipeline.pkl'")


# --- 6. Make Predictions with the Digital Footprint / Device Usage Slave Model ---
y_pred_device = device_model_pipeline.predict(X_test_device)

# Ensure predictions are non-negative if income cannot be negative
y_pred_device[y_pred_device < 0] = 0

# --- 7. Evaluate the Digital Footprint / Device Usage Slave Model ---
mae_device = mean_absolute_error(y_test, y_pred_device)
r2_device = r2_score(y_test, y_pred_device)

print(f"\nDigital Footprint / Device Usage Slave Model Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE) for Device Model: ${mae_device:,.2f}")
print(f"R-squared (R2) Score for Device Model: {r2_device:.4f}")

# --- 8. Generate 'device_income' column for the original DataFrame ---
print("\nGenerating 'device_income' predictions for the entire dataset...")
# Predict device income for the entire original dataset
# Pass a copy of the features from the original df to avoid issues with alignment
df['device_income'] = device_model_pipeline.predict(df[device_features].copy())

# Ensure device_income is non-negative
df.loc[df['device_income'] < 0, 'device_income'] = 0

print("\nDataFrame with 'device_income' column:")
print(df[['id', 'target_income', 'device_income']].head())
print(f"Number of rows with 'device_income' predictions: {df['device_income'].count()}")


# --- 9. Save the updated DataFrame and the device_income.csv ---

# Save the entire DataFrame with the new 'device_income' column
output_full_filename = 'processed_dataset_with_device_income.csv'
df.to_csv(output_full_filename, index=False)
print(f"\nUpdated dataset saved to '{output_full_filename}'")

# Create a new DataFrame with only 'id' and 'device_income' and save it
device_income_df = df[['id', 'device_income']].copy()
output_device_filename = 'device_income.csv'
device_income_df.to_csv(output_device_filename, index=False)
print(f"Device income predictions saved to '{output_device_filename}'")

# --- 10. Example Prediction for a New Data Point using Digital Footprint / Device Usage Model ---
# This part is just for demonstration of how to use the trained model for new data.
# The new data point needs to include ALL features that the model expects,
# including all the 'var_' columns that are part of 'device_features_candidate'.
new_data_point_device = pd.DataFrame([{
    'device_model': 'iPhone 12',
    'device_category': 'Smartphone',
    'platform': 'iOS',
    'device_manufacturer': 'Apple',
    'var_32': 750, # credit_score
    'var_0': 1000, 'var_2': 10000, 'var_6': 5000, 'var_9': 500, # Example financial vars
    'var_15': 1, # Example inquiry count
    'var_62': 2, # Example total loan recent
    # NEWLY ADDED FEATURES for new data point
    'var_10': 8000, # active_credit_limit_1
    'var_11': 7500, # credit_limit_recent_1
    'var_24': 4000, # loan_amt_recent
    'var_25': 0, # total_inquiries_recent
}])

# Explicitly convert categorical columns in the new_data_point_device to string type
for col in categorical_cols_to_convert: # Use the full comprehensive list for robustness
    if col in new_data_point_device.columns:
        new_data_point_device[col] = new_data_point_device[col].astype(str)
        # Replacing 'nan' string with actual NaN if present
        new_data_point_device[col] = new_data_point_device[col].replace('nan', np.nan)


# Ensure the new data point contains all and only the features used for training this model.
new_data_point_device_for_prediction = new_data_point_device[device_features].copy()

predicted_device_income = device_model_pipeline.predict(new_data_point_device_for_prediction)[0]
print(f"\nPredicted 'device_income' for a new data point: ${predicted_device_income:,.2f}")

# --- End of Digital Footprint / Device Usage Slave Model Code ---
