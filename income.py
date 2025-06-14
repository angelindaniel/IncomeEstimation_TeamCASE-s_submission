import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib # Import joblib for saving/loading the model

# --- 1. Load the Dataset ---
print("--- Loading Dataset: processed_dataset_400.csv ---")

# Load the provided dataset
df = pd.read_csv("processed_dataset_400.csv") # Updated to use the uploaded file

# --- 2. Initial Data Exploration ---
print("\n--- Initial Data Exploration ---")
print("Dataset Head:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Define numerical and categorical columns based on the provided schema and initial data types
# The `df.select_dtypes` might not perfectly capture all 'category' types if they are numeric-like
# or if they are loaded as 'object' due to mixed types or NaNs.
# We'll explicitly define them based on the user's provided column descriptions.

# Numerical columns identified from user's description (excluding ID and target)
numerical_cols = [
    'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'pin', 'var_5', 'var_6', 'var_7',
    'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15',
    'var_16', 'var_17', 'var_18', 'var_19', 'var_20', 'var_21', 'var_22', 'var_23',
    'var_24', 'var_25', 'var_26', 'var_27', 'age', 'var_28', 'var_29', 'var_30',
    'var_31', 'var_32', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38',
    'var_39', 'var_40', 'var_41', 'var_42', 'var_43', 'var_44', 'var_45', 'var_46',
    'var_47', 'var_48', 'var_49', 'var_50', 'var_51', 'var_52', 'var_53', 'var_54',
    'var_55', 'var_56', 'var_57', 'var_58', 'var_59', 'var_60', 'var_61', 'var_62',
    'var_63', 'var_64', 'var_65', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70',
    'var_71', 'var_72', 'var_73'
]

# Categorical columns identified from user's description
categorical_cols = [
    'var_74', 'var_75', 'gender', 'marital_status', 'city', 'state',
    'residence_ownership', 'device_model', 'device_category', 'platform',
    'device_manufacturer'
]

target_col = 'target_income'
id_col = 'id'

# Ensure all columns exist in the DataFrame before proceeding
# This is a safeguard if the actual CSV differs from the description
all_expected_cols = numerical_cols + categorical_cols + [target_col, id_col]
missing_cols_in_df = [col for col in all_expected_cols if col not in df.columns]
if missing_cols_in_df:
    print(f"Warning: The following columns from the schema are missing in the loaded DataFrame: {missing_cols_in_df}")
    # Remove missing columns from our lists to avoid errors
    numerical_cols = [col for col in numerical_cols if col not in missing_cols_in_df]
    categorical_cols = [col for col in categorical_cols if col not in missing_cols_in_df]


print(f"\nNumerical Columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")

# --- 3. Feature Engineering ---
print("\n--- Feature Engineering ---")

# Step 3.1: Handling Missing Values and Encoding Categorical Features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Impute missing numerical values with median
    ('scaler', StandardScaler()) # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing categorical values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
])

# Step 3.2: Creating New Features (Example: Ratio and Interaction Features)
df_fe = df.copy() # Create a copy to work on

# Example 1: Ratio of balance to credit limit
# Mapping columns to their descriptive names for clarity
balance_credit_map = {
    'balance_credit_ratio_1': ('var_0', 'var_2'), # balance_1 / credit_limit_1
    'balance_credit_ratio_2': ('var_1', 'var_3'), # balance_2 / credit_limit_2
    'balance_credit_ratio_3': ('var_4', 'var_5'), # balance_3 / credit_limit_3
}

for new_col, (balance_col, credit_col) in balance_credit_map.items():
    if balance_col in df_fe.columns and credit_col in df_fe.columns:
        df_fe[new_col] = df_fe[balance_col] / (df_fe[credit_col] + 1e-6) # Add epsilon to avoid division by zero
        if new_col not in numerical_cols: # Add new feature to numerical columns list if not already present
            numerical_cols.append(new_col)


# Example 2: Total loan amount
actual_loan_amt_cols = [
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31',
    'var_36', 'var_39', 'var_42', 'var_65', 'var_72'
]
# Filter for columns that actually exist in the DataFrame
existing_loan_cols = [col for col in actual_loan_amt_cols if col in df_fe.columns]
if existing_loan_cols:
    df_fe['total_loan_amount'] = df_fe[existing_loan_cols].sum(axis=1)
    if 'total_loan_amount' not in numerical_cols:
        numerical_cols.append('total_loan_amount')
else:
    print("Warning: No loan amount columns found to create 'total_loan_amount'.")


# Example 3: Total inquiries
actual_inquiries_cols = [
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71'
]
existing_inquiries_cols = [col for col in actual_inquiries_cols if col in df_fe.columns]
if existing_inquiries_cols:
    df_fe['total_inquiries_all'] = df_fe[existing_inquiries_cols].sum(axis=1)
    if 'total_inquiries_all' not in numerical_cols:
        numerical_cols.append('total_inquiries_all')
else:
    print("Warning: No inquiry columns found to create 'total_inquiries_all'.")


# Example 4: Age-Credit Score Interaction
if 'age' in df_fe.columns and 'var_32' in df_fe.columns: # var_32 is credit_score
    df_fe['age_credit_score_interaction'] = df_fe['age'] * df_fe['var_32']
    if 'age_credit_score_interaction' not in numerical_cols:
        numerical_cols.append('age_credit_score_interaction')
else:
    print("Warning: 'age' or 'var_32' (credit_score) not found to create interaction feature.")


# Separate features (X) and target (y)
X = df_fe.drop(columns=[target_col, id_col], errors='ignore') # Drop 'id' and target, ignore if 'id' not present
y = df_fe[target_col]

# Re-define preprocessor with the *updated* lists of numerical and categorical columns
# (These lists were updated as new features were created and added to numerical_cols)
# Ensure the columns passed to ColumnTransformer actually exist in X.
final_numerical_cols = [col for col in numerical_cols if col in X.columns]
final_categorical_cols = [col for col in categorical_cols if col in X.columns]


preprocessor_final = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, final_numerical_cols),
        ('cat', categorical_transformer, final_categorical_cols)
    ],
    remainder='passthrough' # Keep other columns (e.g., if there are any not in our defined lists)
)


# --- 4. Model Training Preparation ---
print("\n--- Model Training Preparation ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 5. Model Selection and Training ---
print("\n--- Model Training ---")

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Create the full pipeline
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor_final),
                                ('regressor', model)])

# Train the model
print("Training the Random Forest Regressor model...")
full_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- Save the trained model to a file ---
model_filename = 'full_income_prediction_pipeline.joblib'
joblib.dump(full_pipeline, model_filename)
print(f"\n--- Model Saved ---")
print(f"Trained model saved to '{model_filename}'")


# --- 6. Model Evaluation ---
print("\n--- Model Evaluation ---")
y_pred = full_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# --- 7. Model Inference/Prediction on Full Dataset ---
print("\n--- Model Inference on Full Dataset ---")

# Make predictions on the entire original dataset (X)
# We use X here, which is df_fe.drop(columns=[target_col, id_col]),
# so it contains all the features, including the engineered ones.
all_predictions = full_pipeline.predict(X)

# Add predictions to a new DataFrame for easy viewing
df_results = pd.DataFrame({
    'id': df[id_col],
    'actual_target_income': df[target_col],
    'predicted_target_income': all_predictions
})

print("\nSample of Actual vs. Predicted Target Income (first 10 rows):")
print(df_results.head(10))

print("\n--- Feature Engineering & ML Process Complete ---")
print("The 'target_income' was predicted using Random Forest Regressor.")
