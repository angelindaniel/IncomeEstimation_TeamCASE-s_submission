# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib # For saving the model

# # --- 1. Load the original dataset and all slave model predictions ---
# try:
#     df_original = pd.read_csv('processed_dataset.csv')
#     print("Original dataset 'processed_dataset.csv' loaded successfully.")
#     print("Original DataFrame head:")
#     print(df_original.head())
#     print("\nOriginal DataFrame Info:")
#     df_original.info()

#     # --- Load slave model predictions ---
#     df_behavioral = pd.read_csv('behavioral_income.csv')
#     df_demographic = pd.read_csv('demographic_income.csv')
#     df_location = pd.read_csv('location_income.csv')
#     df_device = pd.read_csv('device_income.csv')
#     print("\nSlave model predictions loaded successfully.")

#     # --- Explicitly convert categorical columns in original df to string type ---
#     # This ensures consistency for any original features passed to Master
#     categorical_cols_to_convert = [
#         'gender', 'marital_status', 'city', 'state', 'residence_ownership',
#         'pin',
#         'device_model', 'device_category', 'platform', 'device_manufacturer',
#         'var_74', 'var_75'
#     ]
#     for col in categorical_cols_to_convert:
#         if col in df_original.columns:
#             df_original[col] = df_original[col].astype(str).replace('nan', np.nan)
#     print("\nOriginal DataFrame Info (after explicit type conversion for categorical columns):")
#     df_original.info()

# except FileNotFoundError as e:
#     print(f"Error: Required file not found - {e}")
#     print("Please ensure 'processed_dataset.csv', 'behavioral_income.csv', 'demographic_income.csv', 'location_income.csv', and 'device_income.csv' are in the same directory.")
#     exit() # Exit if crucial files are missing


# # --- 2. Merge all slave model predictions into the original DataFrame ---
# # Ensure merging is done on 'id' and handle potential duplicate columns if any
# df_master = df_original.copy()

# # Merge slave predictions, using suffixes to distinguish if column names overlap (e.g., 'id')
# df_master = pd.merge(df_master, df_behavioral[['id', 'behavioral_income']], on='id', how='left')
# df_master = pd.merge(df_master, df_demographic[['id', 'demographic_income']], on='id', how='left')
# df_master = pd.merge(df_master, df_location[['id', 'location_income']], on='id', how='left')
# df_master = pd.merge(df_master, df_device[['id', 'device_income']], on='id', how='left')

# print("\nDataFrame after merging slave predictions (head):")
# print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income']].head())
# print(f"Total rows in master DataFrame: {len(df_master)}")


# # --- 3. Define Features for the Master Model and Target ---
# TARGET = 'target_income'
# EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# # Master model features: predictions from slave models + selected original key features
# master_features_candidate = [
#     'behavioral_income',
#     'demographic_income',
#     'location_income',
#     'device_income',
#     'age',         # Original key feature
#     'var_32',      # Original credit_score feature
# ]

# # Filter master features to ensure they exist in the DataFrame
# master_features = [col for col in master_features_candidate if col in df_master.columns and col not in EXCLUDE_COLS]

# # Separate numerical and categorical features for the preprocessor
# # For the master model, the slave predictions are numerical. Original features might be mixed.
# master_numerical_features = [col for col in master_features if pd.api.types.is_numeric_dtype(df_master[col])]
# master_categorical_features = [col for col in master_features if pd.api.types.is_object_dtype(df_master[col]) or pd.api.types.is_string_dtype(df_master[col])]

# print(f"\nMaster Numerical Features: {master_numerical_features}")
# print(f"Master Categorical Features: {master_categorical_features}")


# # --- 4. Split Data into Training and Testing Sets ---
# # Drop rows where TARGET or any of the master features are NaN
# # It's crucial for the master model to have complete input features.
# df_clean_master = df_master.dropna(subset=[TARGET] + master_features).copy()

# X_master = df_clean_master[master_features].copy()
# y_master = df_clean_master[TARGET]

# X_train_master, X_test_master, y_train_master, y_test_master = train_test_split(
#     X_master, y_master, test_size=0.2, random_state=42
# )

# print(f"\nMaster Training data shape: {X_train_master.shape}")
# print(f"Master Testing data shape: {X_test_master.shape}")


# # --- 5. Preprocessing Pipeline for Master Model ---

# # Numerical pipeline: Impute with median (for original features), then scale
# master_numerical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# # Categorical pipeline: Impute with mode, then One-Hot Encode (if any original categorical features are included)
# master_categorical_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Combine pipelines using ColumnTransformer for master features
# master_preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', master_numerical_pipeline, master_numerical_features),
#         ('cat', master_categorical_pipeline, master_categorical_features)
#     ],
#     remainder='drop',
#     verbose_feature_names_out=False
# )

# # --- 6. Build and Train the Master Model Pipeline ---
# master_model_pipeline = Pipeline([
#     ('preprocessor', master_preprocessor),
#     ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.03)) # Tunable HPs
# ])

# print("\nTraining the Master Model...")
# master_model_pipeline.fit(X_train_master, y_train_master)
# print("Master Model training complete.")

# # --- ADDED LINE: Save the trained Master model ---
# joblib.dump(master_model_pipeline, 'master_model_pipeline.pkl')
# print("\nMaster model saved to 'master_model_pipeline.pkl'")


# # --- 7. Make Predictions with the Master Model ---
# y_pred_master = master_model_pipeline.predict(X_test_master)

# # Ensure predictions are non-negative if income cannot be negative
# y_pred_master[y_pred_master < 0] = 0

# # --- 8. Evaluate the Master Model ---
# mae_master = mean_absolute_error(y_test_master, y_pred_master)
# r2_master = r2_score(y_test_master, y_pred_master)

# print(f"\nMaster Model Evaluation on Test Set:")
# print(f"Mean Absolute Error (MAE) for Master Model: ${mae_master:,.2f}")
# print(f"R-squared (R2) Score for Master Model: {r2_master:.4f}")

# # --- 9. Generate 'final_predicted_income' column for the entire dataset ---
# print("\nGenerating 'final_predicted_income' predictions for the entire dataset...")
# # Make sure to use the 'df_master' DataFrame (which includes original features and slave predictions)
# # for prediction, ensuring all necessary columns are available.
# # Use the filtered master_features for prediction.
# df_master['final_predicted_income'] = master_model_pipeline.predict(df_master[master_features].copy())

# # Ensure final_predicted_income is non-negative
# df_master.loc[df_master['final_predicted_income'] < 0, 'final_predicted_income'] = 0

# print("\nDataFrame with 'final_predicted_income' column:")
# print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income', 'final_predicted_income']].head())
# print(f"Number of rows with 'final_predicted_income' predictions: {df_master['final_predicted_income'].count()}")


# # --- 10. Save the final DataFrame ---
# # Save the entire DataFrame with all original columns and new income predictions
# output_final_filename = 'processed_dataset_with_all_income_predictions.csv'
# df_master.to_csv(output_final_filename, index=False)
# print(f"\nFinal dataset saved to '{output_final_filename}'")


# # --- End of Master Model Code ---

#--------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold # Import KFold for cross-validation
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib # For saving the model

# --- 1. Load the original dataset and all slave model predictions ---
try:
    df_original = pd.read_csv('processed_dataset.csv')
    print("Original dataset 'processed_dataset.csv' loaded successfully.")
    print("Original DataFrame head:")
    print(df_original.head())
    print("\nOriginal DataFrame Info:")
    df_original.info()

    # --- Load slave model predictions ---
    df_behavioral = pd.read_csv('behavioral_income.csv')
    df_demographic = pd.read_csv('demographic_income.csv')
    df_location = pd.read_csv('location_income.csv')
    df_device = pd.read_csv('device_income.csv')
    print("\nSlave model predictions loaded successfully.")

    # --- Explicitly convert categorical columns in original df to string type ---
    # This ensures consistency for any original features passed to Master
    categorical_cols_to_convert = [
        'gender', 'marital_status', 'city', 'state', 'residence_ownership',
        'pin',
        'device_model', 'device_category', 'platform', 'device_manufacturer',
        'var_74', 'var_75'
    ]
    for col in categorical_cols_to_convert:
        if col in df_original.columns:
            df_original[col] = df_original[col].astype(str).replace('nan', np.nan)
    print("\nOriginal DataFrame Info (after explicit type conversion for categorical columns):")
    df_original.info()

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    print("Please ensure 'processed_dataset.csv', 'behavioral_income.csv', 'demographic_income.csv', 'location_income.csv', and 'device_income.csv' are in the same directory.")
    exit() # Exit if crucial files are missing


# --- 2. Merge all slave model predictions into the original DataFrame ---
# Ensure merging is done on 'id' and handle potential duplicate columns if any
df_master = df_original.copy()

# Merge slave predictions, using suffixes to distinguish if column names overlap (e.g., 'id')
df_master = pd.merge(df_master, df_behavioral[['id', 'behavioral_income']], on='id', how='left')
df_master = pd.merge(df_master, df_demographic[['id', 'demographic_income']], on='id', how='left')
df_master = pd.merge(df_master, df_location[['id', 'location_income']], on='id', how='left')
df_master = pd.merge(df_master, df_device[['id', 'device_income']], on='id', how='left')

print("\nDataFrame after merging slave predictions (head):")
print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income']].head())
print(f"Total rows in master DataFrame: {len(df_master)}")


# --- 3. Define Features for the Master Model and Target ---
TARGET = 'target_income'
EXCLUDE_COLS = ['id', TARGET] # Exclude 'id' and the 'TARGET' itself from features

# Master model features: predictions from slave models + selected original key features
# --- EXPANDED FEATURES FOR MASTER MODEL ---
master_features_candidate = [
    'behavioral_income',
    'demographic_income',
    'location_income',
    'device_income',
    'age',         # Original key feature
    'var_32',      # Original credit_score feature
    'financial_health_score', # Added: powerful engineered feature
    'total_balance',          # Added: powerful engineered feature
    'avg_credit_util',        # Added: powerful engineered feature
    'loan_to_income_1',       # Added: powerful engineered feature
    'loan_to_income_2',       # Added: powerful engineered feature
]

# Filter master features to ensure they exist in the DataFrame
master_features = [col for col in master_features_candidate if col in df_master.columns and col not in EXCLUDE_COLS]

# Separate numerical and categorical features for the preprocessor
# For the master model, the slave predictions are numerical. Original features might be mixed.
master_numerical_features = [col for col in master_features if pd.api.types.is_numeric_dtype(df_master[col])]
master_categorical_features = [col for col in master_features if pd.api.types.is_object_dtype(df_master[col]) or pd.api.types.is_string_dtype(df_master[col])]

print(f"\nMaster Numerical Features: {master_numerical_features}")
print(f"Master Categorical Features: {master_categorical_features}")


# --- 4. Split Data into Training and Testing Sets ---
# Drop rows where TARGET or any of the master features are NaN
# It's crucial for the master model to have complete input features.
df_clean_master = df_master.dropna(subset=[TARGET] + master_features).copy()

X_master = df_clean_master[master_features].copy()
y_master = df_clean_master[TARGET]

# Removed direct train_test_split as KFold will handle splits
print(f"\nMaster Data shape for CV: {X_master.shape}")


# --- 5. Preprocessing Pipeline for Master Model ---

# Numerical pipeline: Impute with median (for original features), then scale
master_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Impute with mode, then One-Hot Encode (if any original categorical features are included)
master_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer for master features
master_preprocessor = ColumnTransformer(
    transformers=[
        ('num', master_numerical_pipeline, master_numerical_features),
        ('cat', master_categorical_pipeline, master_categorical_features)
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# --- 6. Build and Train the Master Model Pipeline with Aggressive Hyperparameter Tuning & Cross-Validation ---

# Aggressive Hyperparameter Tuning for LGBMRegressor
# Increased n_estimators, decreased learning_rate, adjusted num_leaves and min_child_samples
lgbm_regressor = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=3000,      # Increased from 1000
    learning_rate=0.01,     # Decreased from 0.03
    num_leaves=32,          # Adjusted (default is 31, often reducing helps with overfitting)
    max_depth=-1,           # No limit on tree depth
    min_child_samples=40,   # Increased from 20 (default) to reduce overfitting
    reg_alpha=0.1,          # L1 regularization (alpha) - Added/Tuned
    reg_lambda=0.1,         # L2 regularization (lambda) - Added/Tuned
    colsample_bytree=0.7,   # Subsample columns for each tree (feature_fraction equivalent)
    subsample=0.7,          # Subsample rows for each tree (bagging_fraction equivalent)
    subsample_freq=1,       # Frequency for bagging (bagging_freq equivalent)
    # n_jobs=-1 will use all available cores, but can cause issues in some environments.
    # It's better to manage based on environment. Default is usually fine.
)

master_model_pipeline = Pipeline([
    ('preprocessor', master_preprocessor),
    ('regressor', lgbm_regressor)
])

print("\nTraining the Master Model with Cross-Validation...")

# K-Fold Cross-Validation Setup
n_splits = 5 # Number of folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store metrics for each fold
fold_maes = []
fold_r2s = []
oof_preds = np.zeros(len(X_master)) # Out-of-fold predictions for later use if needed

for fold, (train_index, test_index) in enumerate(kf.split(X_master, y_master)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")
    X_train_fold, X_test_fold = X_master.iloc[train_index], X_master.iloc[test_index]
    y_train_fold, y_test_fold = y_master.iloc[train_index], y_master.iloc[test_index]

    # Fit the pipeline for this fold
    master_model_pipeline.fit(X_train_fold, y_train_fold)
    y_pred_fold = master_model_pipeline.predict(X_test_fold)

    # Ensure predictions are non-negative
    y_pred_fold[y_pred_fold < 0] = 0

    mae_fold = mean_absolute_error(y_test_fold, y_pred_fold)
    r2_fold = r2_score(y_test_fold, y_pred_fold)

    fold_maes.append(mae_fold)
    fold_r2s.append(r2_fold)
    oof_preds[test_index] = y_pred_fold # Store out-of-fold predictions

    print(f"Fold {fold+1} MAE: ${mae_fold:,.2f}")
    print(f"Fold {fold+1} R2: {r2_fold:.4f}")

print("\n--- Cross-Validation Complete ---")
print(f"Average MAE across {n_splits} folds: ${np.mean(fold_maes):,.2f} +/- ${np.std(fold_maes):,.2f}")
print(f"Average R2 across {n_splits} folds: {np.mean(fold_r2s):.4f} +/- {np.std(fold_r2s):.4f}")

# Re-train the model on the full dataset for deployment/final predictions
print("\nRetraining Master Model on full dataset for final use...")
master_model_pipeline.fit(X_master, y_master)
print("Master Model retraining complete on full dataset.")

# --- ADDED LINE: Save the trained Master model ---
joblib.dump(master_model_pipeline, 'master_model_pipeline.pkl')
print("\nMaster model saved to 'master_model_pipeline.pkl'")


# --- 7. Make Predictions with the Re-trained Master Model (on full dataset) ---
# Note: For evaluation on a truly unseen test set, you'd usually use X_test_master from initial split.
# However, for consistency with previous outputs, we will use the full df_master.
# The CV metrics above are the most reliable indicators of generalization.
y_pred_full_dataset = master_model_pipeline.predict(df_master[master_features].copy()) # Predict on the full cleaned dataset

# Ensure predictions are non-negative if income cannot be negative
y_pred_full_dataset[y_pred_full_dataset < 0] = 0

# --- 8. Evaluation on the FULL training dataset (not representative of unseen data) ---
# This is largely for consistency with previous scripts, the CV metrics are more reliable.
mae_full = mean_absolute_error(df_master.dropna(subset=[TARGET] + master_features)[TARGET], y_pred_full_dataset)
r2_full = r2_score(df_master.dropna(subset=[TARGET] + master_features)[TARGET], y_pred_full_dataset)

print(f"\nMaster Model Evaluation on Full Training Data (for reference, CV is primary metric):")
print(f"Mean Absolute Error (MAE) for Master Model: ${mae_full:,.2f}")
print(f"R-squared (R2) Score for Master Model: {r2_full:.4f}")

# --- 9. Generate 'final_predicted_income' column for the entire dataset ---
print("\nGenerating 'final_predicted_income' predictions for the entire dataset...")
# Make sure to use the 'df_master' DataFrame (which includes original features and slave predictions)
# for prediction, ensuring all necessary columns are available.
# Use the filtered master_features for prediction.
df_master['final_predicted_income'] = master_model_pipeline.predict(df_master[master_features].copy())

# Ensure final_predicted_income is non-negative
df_master.loc[df_master['final_predicted_income'] < 0, 'final_predicted_income'] = 0

print("\nDataFrame with 'final_predicted_income' column:")
print(df_master[['id', 'target_income', 'behavioral_income', 'demographic_income', 'location_income', 'device_income', 'final_predicted_income']].head())
print(f"Number of rows with 'final_predicted_income' predictions: {df_master['final_predicted_income'].count()}")


# --- 10. Save the final DataFrame ---
# Save the entire DataFrame with all original columns and new income predictions
output_final_filename = 'processed_dataset_with_all_income_predictions.csv'
df_master.to_csv(output_final_filename, index=False)
print(f"\nFinal dataset saved to '{output_final_filename}'")


# --- End of Master Model Code ---
