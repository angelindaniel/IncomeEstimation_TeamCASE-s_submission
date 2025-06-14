import pandas as pd
import numpy as np
import joblib # Import joblib for saving/loading models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # Import SimpleImputer for robustness
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score # Only need MAE and R2 for regression slave model

# --- 1. Load data ---
# This script strictly relies on 'processed_dataset.csv' being present.
# If not found, a FileNotFoundError will be raised.
df = pd.read_csv('processed_dataset.csv')
print("Dataset 'processed_dataset.csv' loaded successfully.")
print("Original DataFrame head:")
print(df.head())
print("\nDataFrame Info (before type explicit conversion):")
df.info()

# --- Explicitly convert known categorical columns to string type ---
# This is crucial to ensure they are treated as categorical strings consistently.
categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin', # Treating pin as categorical
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75' # Explicitly included for consistency
]
for col in categorical_cols_to_convert:
    if col in df.columns:
        # Convert to string, then replace string 'nan' with actual np.nan
        df[col] = df[col].astype(str).replace('nan', np.nan)
print("\nDataFrame Info (after explicit type conversion for categorical columns):")
df.info()


# --- 2. Define Features for the Behavioral Slave Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# Define behavioral features based on your descriptions and common financial behavioral data
# Ensure this list is comprehensive for your behavioral model
behavioral_features_candidate = [
    # Balance related
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', 'var_34', 'var_35',
    'var_38', 'var_59', 'var_68',
    # Credit limit related (active, recent, general)
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', 'var_26', 'var_27',
    'var_28', 'var_29', 'var_33', 'var_44', 'var_47',
    # Loan amount related (large tenure, primary, recent, general)
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', 'var_39', 'var_42',
    'var_65', 'var_72',
    # EMI related
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',
    # Repayment related
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67', 'var_69', 'var_70', 'var_73',
    # Inquiry related
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',
    # Loan activity/counts
    'var_40', # closed_loan
    'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66', # total_loans and recent
    # Credit score and related comments/types (considered behavioral aspects of score)
    'var_32', # credit_score (Numerical)
    'var_74', # score_comments (Categorical)
    'var_75'  # score_type (Categorical)
]

# Filter features to ensure only columns present in the DataFrame are selected and not excluded
behavioral_features = [col for col in behavioral_features_candidate if col in df.columns and col not in EXCLUDE_COLS]

# Explicitly define numerical and categorical behavioral features after type conversion
# This check relies on the df having correct dtypes after the explicit astype(str)
behavioral_numerical_features = [col for col in behavioral_features if pd.api.types.is_numeric_dtype(df[col])]
behavioral_categorical_features = [col for col in behavioral_features if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]


print(f"\nBehavioral Numerical Features: {behavioral_numerical_features}")
print(f"Behavioral Categorical Features: {behavioral_categorical_features}")

# --- 3. Feature Engineering and Preprocessing Pipeline for Behavioral Model ---

# Numerical pipeline: Impute with median, then scale
behavioral_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with mode, then One-Hot Encode
behavioral_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for unseen categories
])

# Combine pipelines using ColumnTransformer for behavioral features
# Set verbose_feature_names_out=False to avoid the UserWarning about feature names
behavioral_preprocessor = ColumnTransformer(
    transformers=[
        ('num', behavioral_numerical_pipeline, behavioral_numerical_features),
        ('cat', behavioral_categorical_pipeline, behavioral_categorical_features)
    ],
    remainder='drop', # Drop any columns not specified in behavioral_features
    verbose_feature_names_out=False # Suppress warning about feature names
)

# --- 4. Split Data into Training and Testing Sets for Behavioral Model ---
# Drop rows where TARGET is NaN if any exist (critical for training a regression model)
df_clean = df.dropna(subset=[TARGET]).copy() # Use .copy() to prevent SettingWithCopyWarning later

X_behavioral = df_clean[behavioral_features]
y = df_clean[TARGET]

X_train_behavioral, X_test_behavioral, y_train, y_test = train_test_split(
    X_behavioral, y, test_size=0.2, random_state=42
)

print(f"\nBehavioral Training data shape: {X_train_behavioral.shape}")
print(f"Behavioral Testing data shape: {X_test_behavioral.shape}")

# --- 5. Build and Train the Behavioral Slave Model ---
behavioral_model = Pipeline([
    ('preprocessor', behavioral_preprocessor),
    ('regressor', LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05)) # Keeping these HPs
])

print("\nTraining the Behavioral Slave Model...")
behavioral_model.fit(X_train_behavioral, y_train)
print("Behavioral Slave Model training complete.")

# --- ADDED LINE: Save the trained model ---
joblib.dump(behavioral_model, 'behavioral_model_pipeline.pkl')
print("\nBehavioral model saved to 'behavioral_model_pipeline.pkl'")


# --- 6. Make Predictions with the Behavioral Slave Model ---
y_pred_behavioral = behavioral_model.predict(X_test_behavioral)

# Ensure predictions are non-negative if income cannot be negative
y_pred_behavioral[y_pred_behavioral < 0] = 0

# --- 7. Evaluate the Behavioral Slave Model ---
mae_behavioral = mean_absolute_error(y_test, y_pred_behavioral)
r2_behavioral = r2_score(y_test, y_pred_behavioral)

print(f"\nBehavioral Slave Model Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE) for Behavioral Model: ${mae_behavioral:,.2f}")
print(f"R-squared (R2) Score for Behavioral Model: {r2_behavioral:.4f}")

# --- 8. Generate 'behavioral_income' column for the original DataFrame ---
print("\nGenerating 'behavioral_income' predictions for the entire dataset...")
# Predict behavioral income for the entire original dataset
# Pass a copy of the features from the original df to avoid issues with alignment
df['behavioral_income'] = behavioral_model.predict(df[behavioral_features].copy())

# Ensure behavioral_income is non-negative
# Use .loc for safe assignment to avoid SettingWithCopyWarning
df.loc[df['behavioral_income'] < 0, 'behavioral_income'] = 0

print("\nDataFrame with 'behavioral_income' column:")
print(df[['id', 'target_income', 'behavioral_income']].head())
print(f"Number of rows with 'behavioral_income' predictions: {df['behavioral_income'].count()}")


# --- 9. Save the updated DataFrame and the behavioral_income.csv ---

# Save the entire DataFrame with the new 'behavioral_income' column
output_full_filename = 'processed_dataset_with_behavioral_income.csv'
df.to_csv(output_full_filename, index=False)
print(f"\nUpdated dataset saved to '{output_full_filename}'")

# Create a new DataFrame with only 'id' and 'behavioral_income' and save it
behavioral_income_df = df[['id', 'behavioral_income']].copy()
output_behavioral_filename = 'behavioral_income.csv'
behavioral_income_df.to_csv(output_behavioral_filename, index=False)
print(f"Behavioral income predictions saved to '{output_behavioral_filename}'")


# --- Example Prediction for a New Data Point using Behavioral Model ---
# This part is just for demonstration of how to use the trained model for new data.
new_data_point_behavioral = pd.DataFrame([{
    'var_0': 1000, 'var_1': 500, 'var_2': 10000, 'var_3': 15000, 'var_4': 2000,
    'var_5': 20000, 'var_6': 5000, 'var_7': 7000, 'var_8': 1000,
    'var_9': 500, 'var_10': 9000, 'var_11': 8000, 'var_12': 12000, 'var_13': 20000,
    'var_14': 15000, 'var_15': 1, 'var_16': 0, 'var_17': 700, 'var_18': 300,
    'var_19': 600, 'var_20': 10000, 'var_21': 900, 'var_22': 50000, 'var_23': 60000,
    'var_24': 8000, 'var_25': 0, 'var_26': 70000, 'var_27': 80000,
    'var_28': 90000, 'var_29': 100000, 'var_30': 1500, 'var_31': 12000, 'var_32': 750,
    'var_33': 110000, 'var_34': 1800, 'var_35': 2200, 'var_36': 15000, 'var_37': 400,
    'var_38': 2500, 'var_39': 18000, 'var_40': 5, 'var_41': 800, 'var_42': 22000,
    'var_43': 900, 'var_44': 120000, 'var_45': 2, 'var_46': 600, 'var_47': 130000,
    'var_48': 500, 'var_49': 600, 'var_50': 700, 'var_51': 400, 'var_52': 800,
    'var_53': 10, 'var_54': 3, 'var_55': 900, 'var_56': 300, 'var_57': 12,
    'var_58': 1, 'var_59': 1000, 'var_60': 11, 'var_61': 0, 'var_62': 2,
    'var_63': 15, 'var_64': 18, 'var_65': 25000, 'var_66': 20, 'var_67': 1000,
    'var_68': 3000, 'var_69': 1100, 'var_70': 1200, 'var_71': 0, 'var_72': 28000,
    'var_73': 1300,
    'var_74': 'Good Payment History', # Ensure string type here
    'var_75': 'A'                      # Ensure string type here
}])

# Explicitly convert 'var_74' and 'var_75' in the new_data_point_behavioral
# This handles the case where the new data point might be created directly without
# the full DataFrame's type conversion.
for col in ['var_74', 'var_75']:
    if col in new_data_point_behavioral.columns:
        new_data_point_behavioral[col] = new_data_point_behavioral[col].astype(str)

# Ensure the new data point only contains the behavioral features used for training
new_data_point_behavioral = new_data_point_behavioral[behavioral_features]

predicted_behavioral_income = behavioral_model.predict(new_data_point_behavioral)[0]
print(f"\nPredicted 'behavioral_income' for a new data point: ${predicted_behavioral_income:,.2f}")

# --- End of Behavioral Slave Model Code ---
