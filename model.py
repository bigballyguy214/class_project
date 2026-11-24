import pandas as pd

# Load the dataset
df = pd.read_csv('data/survey_regression_data.csv')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Target and updated list of features including match factors and constraints
target_col = 'overall_rating'
feature_cols = [
    'genre_match',
    'vibe_match',
    'size_match',
    'over_budget',
    'crowd_age_gap',
    'different_borough',
    'wait_over_limit_min'
]

print(f"Using Features: {feature_cols}\nTarget: '{target_col}'")

# Drop rows with missing values in these columns for simplicity
data_clean = df[feature_cols + [target_col]].dropna()

X = data_clean[feature_cols]
y = data_clean[target_col]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate and print results
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\nModel trained successfully!")
for idx, feat in enumerate(feature_cols):
    print(f"Coefficient for {feat:20s}: {model.coef_[idx]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R^2 Score on Train: {train_score:.4f}")
print(f"R^2 Score on Test: {test_score:.4f}")