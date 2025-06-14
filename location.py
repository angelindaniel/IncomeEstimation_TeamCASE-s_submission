import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb # Using LGBMClassifier for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib # For saving the model
from sklearn.base import BaseEstimator, TransformerMixin # For Custom TargetEncoder

# --- Custom TargetEncoder for categorical features ---
# This class implements target encoding. Reusing from previous scripts.
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = None

    def fit(self, X, y):
        if y is None:
            raise ValueError("TargetEncoder requires 'y' during fit for target mean calculation.")
        
        # Ensure y is numerical for mean calculation. For classification, this needs a numerical target.
        # We will map 'Good', 'Average', 'Poor' to numbers for internal calculation, then map back.
        # This assumes y is a Series.
        if y.dtype == 'object' or y.dtype == 'string':
            self.target_map = {
                'Poor': 0,
                'Average': 1,
                'Good': 2
            }
            y_numeric = y.map(self.target_map)
        else:
            y_numeric = y

        self.global_mean = y_numeric.mean()

        if not isinstance(X, pd.DataFrame):
            # If X is a numpy array, try to reconstruct with expected column names
            if self.cols and len(self.cols) == X.shape[1]:
                X_df = pd.DataFrame(X, columns=self.cols)
            else:
                # Fallback to generic column names if column names cannot be inferred
                X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        for col in self.cols:
            if col not in X_df.columns:
                print(f"Warning: Column '{col}' not found in X during TargetEncoder fit. Skipping.")
                continue

            # Calculate means for each category
            means = y_numeric.groupby(X_df[col]).mean()
            counts = y_numeric.groupby(X_df[col]).count()

            # Apply smoothing
            smoothed_means = (means * counts + self.global_mean * self.smoothing) / (counts + self.smoothing)
            self.mapping[col] = smoothed_means.to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.cols and len(self.cols) == X.shape[1]:
                X_transformed = pd.DataFrame(X, columns=self.cols)
            else:
                X_transformed = pd.DataFrame(X)
        else:
            X_transformed = X.copy()

        for col in self.cols:
            if col in self.mapping:
                # Map categories to their encoded values. Fill unseen categories/NaNs with global mean.
                X_transformed[col] = X_transformed[col].map(self.mapping[col]).fillna(self.global_mean)
            else:
                # If column was not in mapping during fit (e.g., new column), fill with global mean.
                X_transformed[col] = np.full(len(X_transformed), self.global_mean)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # Output features are just the input columns, but numerical.
        return self.cols


# --- 1. Load the original dataset and Master Model predictions ---
try:
    df_original = pd.read_csv('processed_dataset.csv')
    print("Original dataset 'processed_dataset.csv' loaded successfully.")
    
    # Load the dataset with final_predicted_income from the Master Model
    df_master_predictions = pd.read_csv('processed_dataset_with_all_income_predictions.csv')
    print("Master model predictions loaded successfully.")

    # Merge final_predicted_income into the original DataFrame
    df = pd.merge(df_original, df_master_predictions[['id', 'final_predicted_income']], on='id', how='left')
    print("\nDataFrame after merging final income predictions (head):")
    print(df[['id', 'var_32', 'target_income', 'final_predicted_income']].head())
    print("\nDataFrame Info (before type explicit conversion):")
    df.info()

    # --- Explicitly convert categorical columns to string type ---
    categorical_cols_to_convert = [
        'gender', 'marital_status', 'city', 'state', 'residence_ownership',
        'pin',
        'device_model', 'device_category', 'platform', 'device_manufacturer',
        'var_74', 'var_75'
    ]
    for col in categorical_cols_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)
    print("\nDataFrame Info (after explicit type conversion for categorical columns):")
    df.info()

except FileNotFoundError as e:
    print(f"Error: Required file not found - {e}")
    print("Please ensure 'processed_dataset.csv' and 'processed_dataset_with_all_income_predictions.csv' are in the same directory.")
    exit()


# --- 2. Define the Target Variable: Creditworthiness Labels ---
TARGET = 'creditworthiness_label'
CREDIT_SCORE_COL = 'var_32' # The column containing credit scores

# Drop rows where credit_score is NaN, as we need it to define the target
df_clean = df.dropna(subset=[CREDIT_SCORE_COL]).copy()

# Define creditworthiness categories
def assign_creditworthiness(score):
    if score >= 700:
        return 'Good'
    elif 600 <= score < 700:
        return 'Average'
    else: # score < 600
        return 'Poor'

df_clean[TARGET] = df_clean[CREDIT_SCORE_COL].apply(assign_creditworthiness)
print(f"\nCreditworthiness Distribution:\n{df_clean[TARGET].value_counts()}")


# --- 3. Define Features for the Creditworthiness Model ---
EXCLUDE_COLS = ['id', 'target_income', CREDIT_SCORE_COL, TARGET]

creditworthiness_features_candidate = [
    'final_predicted_income', # Crucial input from our Master Income Model
    # Core Financial & Credit-related
    'financial_health_score',
    'total_balance',
    'avg_credit_util',
    'loan_to_income_1',
    'loan_to_income_2',
    # Key Behavioral/Loan Activity Indicators (from var_ descriptions)
    'var_0', 'var_1', 'var_4', 'var_8', 'var_18', 'var_19', 'var_21', 'var_30', # Balances
    'var_2', 'var_3', 'var_5', 'var_10', 'var_11', 'var_12', 'var_22', 'var_23', # Credit limits
    'var_6', 'var_7', 'var_13', 'var_14', 'var_20', 'var_24', 'var_31', 'var_36', # Loan amounts
    'var_9', 'var_17', 'var_41', 'var_43', 'var_46', 'var_51', 'var_56', # EMIs
    'var_37', 'var_48', 'var_49', 'var_50', 'var_52', 'var_55', 'var_67', # Repayments
    'var_15', 'var_16', 'var_25', 'var_45', 'var_58', 'var_61', 'var_71', # Inquiries
    'var_40', 'var_53', 'var_54', 'var_57', 'var_60', 'var_62', 'var_63', 'var_64', 'var_66', # Loan counts/activity
    # Demographic / Other relevant contextual features
    'age',
    'gender', 'marital_status', 'residence_ownership',
    'city', 'state', 'pin',
    'device_model', 'device_category', 'platform', 'device_manufacturer',
    'var_74', 'var_75', # Score comments/type (if not too redundant with var_32 itself)
]

# Filter features to ensure they exist in the DataFrame and are not excluded
creditworthiness_features = [col for col in creditworthiness_features_candidate if col in df_clean.columns and col not in EXCLUDE_COLS]

# Split categorical features for different encoding strategies
high_cardinality_for_target_encoding = [
    'city', 'pin', 'device_model' # Likely culprits for memory issues with OneHotEncoder
]
low_cardinality_for_onehot_encoding = [
    col for col in creditworthiness_features if col not in high_cardinality_for_target_encoding and (pd.api.types.is_object_dtype(df_clean[col]) or pd.api.types.is_string_dtype(df_clean[col]))
]

# Ensure no overlap and all desired categorical features are covered
print(f"\nCategorical Features for Target Encoding: {high_cardinality_for_target_encoding}")
print(f"Categorical Features for One-Hot Encoding: {low_cardinality_for_onehot_encoding}")

# Separate numerical features
creditworthiness_numerical_features = [col for col in creditworthiness_features if pd.api.types.is_numeric_dtype(df_clean[col])]


# --- 4. Split Data into Training and Testing Sets ---
X = df_clean[creditworthiness_features].copy()
y = df_clean[TARGET].copy()

# Stratified split to maintain class proportions in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # Stratify for classification
)

print(f"\nCreditworthiness Training data shape: {X_train.shape}")
print(f"Creditworthiness Testing data shape: {X_test.shape}")
print(f"Training Target Distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing Target Distribution:\n{y_test.value_counts(normalize=True)}")


# --- 5. Preprocessing Pipeline for Creditworthiness Model ---

# Numerical pipeline: Impute with MEAN (to avoid memory error), then scale
# CHANGED: strategy='median' to strategy='mean'
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# One-Hot Encoding pipeline for low cardinality categorical features
onehot_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Target Encoding pipeline for high cardinality categorical features
target_categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before encoding
    ('target_encoder', TargetEncoder(cols=high_cardinality_for_target_encoding, smoothing=10.0)) # Increased smoothing
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, creditworthiness_numerical_features),
        ('onehot_cat', onehot_categorical_pipeline, low_cardinality_for_onehot_encoding),
        ('target_cat', target_categorical_pipeline, high_cardinality_for_target_encoding) # New target encoding step
    ],
    remainder='drop',
    verbose_feature_names_out=False
)

# --- 6. Build and Train the Creditworthiness Classification Model ---
lgbm_classifier = lgb.LGBMClassifier(
    random_state=42,
    objective='multiclass',
    num_class=len(y.unique()),
    n_estimators=2000, # Increased estimators for potentially better performance
    learning_rate=0.02, # Slightly reduced learning rate
    num_leaves=40,     # Adjusted num_leaves
    max_depth=-1,
    min_child_samples=30, # Adjusted min_child_samples
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.7,
    subsample=0.7,
    subsample_freq=1,
)

creditworthiness_model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_classifier)
])

print("\nTraining the Creditworthiness Classification Model...")
# The y_train is passed to the fit method of the pipeline, which then routes it
# correctly to the TargetEncoder's fit method within the ColumnTransformer.
creditworthiness_model_pipeline.fit(X_train, y_train)
print("Creditworthiness Classification Model training complete.")

# --- Save the trained model ---
joblib.dump(creditworthiness_model_pipeline, 'creditworthiness_model_pipeline.pkl')
print("\nCreditworthiness model saved to 'creditworthiness_model_pipeline.pkl'")


# --- 7. Make Predictions and Evaluate the Model ---
y_pred = creditworthiness_model_pipeline.predict(X_test)

print("\nCreditworthiness Model Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# --- 8. Generate 'predicted_creditworthiness_label' for the entire dataset ---
print("\nGenerating 'predicted_creditworthiness_label' for the entire dataset...")
# Predict on the full DataFrame (df, not df_clean) to get predictions for all original IDs
df['predicted_creditworthiness_label'] = creditworthiness_model_pipeline.predict(df[creditworthiness_features].copy())

print("\nDataFrame with 'predicted_creditworthiness_label' column (head):")
print(df[['id', 'var_32', 'creditworthiness_label', 'predicted_creditworthiness_label', 'final_predicted_income']].head())
print(f"Number of rows with 'predicted_creditworthiness_label' predictions: {df['predicted_creditworthiness_label'].count()}")


# --- 9. Save the final DataFrame ---
output_final_filename = 'processed_dataset_with_creditworthiness_predictions.csv'
df.to_csv(output_final_filename, index=False)
print(f"\nFinal dataset with creditworthiness predictions saved to '{output_final_filename}'")

print("\nCreditworthiness classification process complete.")
