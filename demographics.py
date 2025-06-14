import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix

# 1️⃣ Load data
df = pd.read_csv('processed_dataset.csv')

# 2️⃣ Select useful columns
keep_cols = [
    'age', 'gender',
    'target_income',
    'var_32',  # credit_score
    'var_62',  # total_loan_recent
    'var_54',  # closed_total_loans
]

balance_cols = [col for col in df.columns if col.startswith('var_') and 'balance' in col]
loan_amt_cols = [col for col in df.columns if col.startswith('var_') and 'loan_amt' in col]

df = df[keep_cols + balance_cols + loan_amt_cols].dropna()

# 3️⃣ Feature engineering
df['age_score'] = df['age'].rank(pct=True) * 0.15
df['income_score'] = df['target_income'].rank(pct=True) * 0.15
df['demo_weight'] = (df['gender'] == 'Female') * 0.05
df['credit_score_norm'] = df['var_32'].rank(pct=True) * 0.15
df['active_loan_score'] = df['var_62'].rank(pct=True) * 0.1
df['closed_loan_score'] = df['var_54'].rank(pct=True) * 0.1

df['leverage'] = df[balance_cols + loan_amt_cols].sum(axis=1)
df['leverage_score'] = -df['leverage'].rank(pct=True) * 0.1

raw_score = (
    df['age_score'] + df['income_score'] + df['demo_weight'] +
    df['credit_score_norm'] + df['active_loan_score'] +
    df['closed_loan_score'] + df['leverage_score']
)
df['demographic_potential_score'] = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())

# 4️⃣ Define features and target
cat_cols = ['gender']
num_cols = [
    'age', 'target_income', 'var_32', 'var_62', 'var_54'
] + balance_cols + loan_amt_cols

X = df[cat_cols + num_cols]
y = df['demographic_potential_score']

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Preprocessing + LightGBM pipeline
pipeline = Pipeline([
    ('preprocess', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])),
    ('model', LGBMRegressor(n_estimators=150, random_state=42))
])

pipeline.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Convert regression to binary class (score > 0.5)
y_test_cls = (y_test > 0.5).astype(int)
y_pred_cls = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls))
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))

# 8️⃣ Save results to CSV
df['predicted_score'] = pipeline.predict(X)
df.to_csv('demographic_output.csv', index=False)
print("Output saved to 'demographic_output.csv'")

# 9️⃣ Persist model
joblib.dump(pipeline, 'demographic_model.pkl')
print("Model saved to 'demographic_model.pkl'")