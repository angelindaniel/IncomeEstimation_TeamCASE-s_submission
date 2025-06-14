import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # Import joblib for saving/loading models

# --- Custom Feature Engineering Function ---
# This function will be applied as a step in the sklearn pipeline.
def create_demographic_engineered_features(X):
    # It's crucial to work on a copy to avoid SettingWithCopyWarning
    X_engineered = X.copy()

    # --- Age Binning ---
    # Convert continuous age into categorical bins.
    if 'age' in X_engineered.columns:
        bins = [0, 25, 35, 45, 55, 65, 100] # Define age ranges
        labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+'] # Labels for bins
        X_engineered['age_bin'] = pd.cut(
            X_engineered['age'],
            bins=bins,
            labels=labels,
            right=False, # Interval is [min, max)
            include_lowest=True # Include the first value in the lowest bin
        ).astype(object) # Ensure the output is an object type for OneHotEncoder

    # --- Interaction Term: Age multiplied by Credit Score ---
    # This can capture non-linear relationships or specific segments (e.g., older individuals with high credit scores).
    if 'age' in X_engineered.columns and 'var_32' in X_engineered.columns: # var_32 is credit_score
        X_engineered['age_x_credit_score'] = X_engineered['age'] * X_engineered['var_32']

    # Handle any potential infinite values that might arise from divisions (e.g., if you add custom ratios)
    # This step is good practice for robustness.
    X_engineered.replace([np.inf, -np.inf], np.nan, inplace=True)

    return X_engineered


# --- 1. Load the actual dataset ---
# This script strictly relies on 'processed_dataset.csv' being present.
# If not found, a FileNotFoundError will be raised.
df = pd.read_csv('processed_dataset.csv')
print("Dataset 'processed_dataset.csv' loaded successfully.")
print("Original DataFrame head:")
print(df.head())
print("\nDataFrame Info (before type explicit conversion):")
df.info()

# --- Explicitly convert known categorical columns to string type ---
# This is crucial to ensure they are treated as categorical strings consistently
# throughout the pipeline, including new data points.
categorical_cols_to_convert = [
    'gender', 'marital_status', 'city', 'state', 'residence_ownership',
    'pin', # Explicitly treating pin as categorical for location-based features
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75' # Ensure these are treated as categorical
]
for col in categorical_cols_to_convert:
    if col in df.columns:
        # Convert to string, then replace string 'nan' (if any) with actual np.nan
        df[col] = df[col].astype(str).replace('nan', np.nan)
print("\nDataFrame Info (after explicit type conversion for categorical columns):")
df.info()


# --- 2. Define Features for the Demographic Slave Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# Define a comprehensive list of features for the Demographic Slave Model.
# This includes general demographic, location, device, and relevant 'var_' columns
# that complement demographic information (like credit limits, loan counts, EMIs, inquiries).
demographic_features_candidate = [
    'age', 'gender', 'marital_status', 'residence_ownership',
    'city', 'state', 'pin', # Pin code as categorical geographical indicator
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_32',  # credit_score

    # Including all credit limit related 'var_' columns
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23',
    'var_26', 'var_27', 'var_28', 'var_29', 'var_33', 'var_44', 'var_47',

    # Including all balance related 'var_' columns
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30',
    'var_34', 'var_35', 'var_38', 'var_59', 'var_68',

    # Including all loan amount related 'var_' columns
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36',
    'var_39', 'var_42', 'var_65', 'var_72',

    # Including all EMI related 'var_' columns
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56',

    # Including all repayment related 'var_' columns (can be contextual for financial stability)
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67',
    'var_69', 'var_70', 'var_73',

    # Including all inquiry related 'var_' columns
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71',

    # Including all total_loans and related activity 'var_' columns
    'var_40',  # closed_loan
    'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66',

    'var_74', # score_comments - treat as categorical
    'var_75', # score_type - treat as categorical

    # New engineered features (placeholder names, actual columns created by FunctionTransformer)
    'age_bin',
    'age_x_credit_score'
]

# Filter features to ensure they exist in the DataFrame and are not excluded
demographic_features = [col for col in demographic_features_candidate if col in df.columns and col not in EXCLUDE_COLS]

# --- 3. Split Data into Training and Testing Sets ---
# Drop rows where TARGET is NaN if any exist (critical for training a regression model)
df_clean = df.dropna(subset=[TARGET]).copy()

# X will include all the features that will be passed to the custom transformer first
X_demographic = df_clean[demographic_features].copy()
y = df_clean[TARGET]

X_train_demographic, X_test_demographic, y_train, y_test = train_test_split(
    X_demographic, y, test_size=0.2, random_state=42
)

# --- 4. Identify Numerical and Categorical Features AFTER potential custom transformations ---
# Apply custom feature engineering on a small sample to get the actual column names
# and their types after feature engineering. This ensures ColumnTransformer is accurate.
sample_df_for_type_detection = create_demographic_engineered_features(X_train_demographic.head())

demographic_numerical_features = [col for col in sample_df_for_type_detection.columns
                                  if pd.api.types.is_numeric_dtype(sample_df_for_type_detection[col])]
demographic_categorical_features = [col for col in sample_df_for_type_detection.columns
                                    if pd.api.types.is_object_dtype(sample_df_for_type_detection[col]) or pd.api.types.is_string_dtype(sample_df_for_type_detection[col])]

print(f"\nDemographic Numerical Features (after engineering): {demographic_numerical_features}")
print(f"Demographic Categorical Features (after engineering): {demographic_categorical_features}")

# --- 5. Preprocessing Pipeline for Demographic Model ---

# Numerical pipeline: Impute with median, then scale
demographic_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with mode, then One-Hot Encode
demographic_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' for unseen categories
])

# Combine pipelines using ColumnTransformer for demographic features
demographic_preprocessor = ColumnTransformer(
    transformers=[
        ('num', demographic_numerical_pipeline, demographic_numerical_features),
        ('cat', demographic_categorical_pipeline, demographic_categorical_features)
    ],
    remainder='drop', # Drop any columns not explicitly handled
    verbose_feature_names_out=False # Suppress warning about feature names in LightGBM
)

# --- 6. Build and Train the Demographic Slave Model Pipeline ---
demographic_model_pipeline = Pipeline([
    ('feature_engineer', FunctionTransformer(create_demographic_engineered_features, validate=False)), # Custom FE step
    ('preprocessor', demographic_preprocessor),
    ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=1500, learning_rate=0.03)) # Increased estimators, reduced LR
])

print("\nTraining the Demographic Slave Model...")
demographic_model_pipeline.fit(X_train_demographic, y_train)
print("Demographic Slave Model training complete.")

# --- ADDED LINE: Save the trained model ---
joblib.dump(demographic_model_pipeline, 'demographic_model_pipeline.pkl')
print("\nDemographic model saved to 'demographic_model_pipeline.pkl'")


# --- 7. Make Predictions with the Demographic Slave Model ---
y_pred_demographic = demographic_model_pipeline.predict(X_test_demographic)

# Ensure predictions are non-negative if income cannot be negative
y_pred_demographic[y_pred_demographic < 0] = 0

# --- 8. Evaluate the Demographic Slave Model ---
mae_demographic = mean_absolute_error(y_test, y_pred_demographic)
r2_demographic = r2_score(y_test, y_pred_demographic)

print(f"\nDemographic Slave Model Evaluation on Test Set:")
print(f"Mean Absolute Error (MAE) for Demographic Model: ${mae_demographic:,.2f}")
print(f"R-squared (R2) Score for Demographic Model: {r2_demographic:.4f}")

# --- 9. Generate 'demographic_income' column for the original DataFrame ---
print("\nGenerating 'demographic_income' predictions for the entire dataset...")
# Predict demographic income for the entire original dataset
# Pass a copy of the features from the original df to avoid issues with alignment
df['demographic_income'] = demographic_model_pipeline.predict(df[demographic_features].copy())

# Ensure demographic_income is non-negative
df.loc[df['demographic_income'] < 0, 'demographic_income'] = 0

print("\nDataFrame with 'demographic_income' column:")
print(df[['id', 'target_income', 'demographic_income']].head())
print(f"Number of rows with 'demographic_income' predictions: {df['demographic_income'].count()}")


# --- 10. Save the updated DataFrame and the demographic_income.csv ---

# Save the entire DataFrame with the new 'demographic_income' column
output_full_filename = 'processed_dataset_with_demographic_income.csv'
df.to_csv(output_full_filename, index=False)
print(f"\nUpdated dataset saved to '{output_full_filename}'")

# Create a new DataFrame with only 'id' and 'demographic_income' and save it
demographic_income_df = df[['id', 'demographic_income']].copy()
output_demographic_filename = 'demographic_income.csv'
demographic_income_df.to_csv(output_demographic_filename, index=False)
print(f"Demographic income predictions saved to '{output_demographic_filename}'")

# --- 11. Example Prediction for a New Data Point using Demographic Model ---
# This part is just for demonstration of how to use the trained model for new data.
# The new data point needs to include ALL features that the model expects,
# including all the 'var_' columns that are part of 'demographic_features_candidate'.
new_data_point_demographic = pd.DataFrame([{
    'age': 35, 'gender': 'Male', 'marital_status': 'Married', 'residence_ownership': 'Owned',
    'city': 'Mumbai', 'state': 'MH', 'pin': '400001', # Pin as string
    'device_model': 'iPhone 12', 'device_category': 'Smartphone', 'platform': 'iOS', 'device_manufacturer': 'Apple',
    'var_32': 750, # credit_score

    # Ensure all original 'var_' columns that are part of demographic_features_candidate are included
    'var_0': 1000, 'var_1': 500, 'var_4': 2000, 'var_8': 1000,
    'var_18': 300, 'var_19': 600, 'var_21': 900, 'var_30': 1500,
    'var_34': 1800, 'var_35': 2200, 'var_38': 2500, 'var_59': 1000, 'var_68': 3000,

    'var_2': 10000, 'var_3': 15000, 'var_5': 20000, 'var_10': 9000, 'var_11': 8000, 'var_12': 12000,
    'var_22': 50000, 'var_23': 60000, 'var_26': 70000, 'var_27': 80000, 'var_28': 90000, 'var_29': 100000,
    'var_33': 110000, 'var_44': 120000, 'var_47': 130000,

    'var_6': 5000, 'var_7': 7000, 'var_13': 20000, 'var_14': 15000, 'var_20': 10000, 'var_24': 8000,
    'var_31': 12000, 'var_36': 15000, 'var_39': 18000, 'var_42': 22000, 'var_65': 25000, 'var_72': 28000,

    'var_9': 500, 'var_17': 700, 'var_41': 800, 'var_43': 900, 'var_46': 600, 'var_51': 400, 'var_56': 300,

    'var_37': 400, 'var_48': 500, 'var_49': 600, 'var_50': 700, 'var_52': 800, 'var_55': 900,
    'var_67': 1000, 'var_69': 1100, 'var_70': 1200, 'var_73': 1300,

    'var_15': 1, 'var_16': 0, 'var_25': 0, 'var_45': 2, 'var_58': 1, 'var_61': 0, 'var_71': 0,

    'var_40': 5, 'var_53': 10, 'var_54': 3, 'var_57': 12, 'var_60': 11, 'var_62': 2,
    'var_63': 15, 'var_64': 18, 'var_66': 20,

    'var_74': 'Good Payment History', # score_comments
    'var_75': 'A' # score_type
}])

# Explicitly convert categorical columns in the new_data_point_demographic to string type
# This is vital for consistency with the training data preprocessing.
for col in categorical_cols_to_convert:
    if col in new_data_point_demographic.columns:
        new_data_point_demographic[col] = new_data_point_demographic[col].astype(str)

# Ensure the new data point contains all and only the features used for training this model.
# The 'demographic_features' list includes the original features + new engineered feature names.
new_data_point_demographic_for_prediction = new_data_point_demographic[demographic_features].copy()

predicted_demographic_income = demographic_model_pipeline.predict(new_data_point_demographic_for_prediction)[0]
print(f"\nPredicted 'demographic_income' for a new data point: ${predicted_demographic_income:,.2f}")

# --- End of Demographic Slave Model Code ---
