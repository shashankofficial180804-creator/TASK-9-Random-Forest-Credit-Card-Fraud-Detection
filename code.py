# Task 9: Credit Card Fraud Detection using Random Forest
# Dataset: Kaggle Credit Card Fraud Dataset

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# 1. Create required directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 2. Load Dataset
df = pd.read_csv("creditcard.csv")

# 3. Check class imbalance
print("Class Distribution:")
print(df["Class"].value_counts())

# 4. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 5. Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Feature Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Baseline Model: Logistic Regression
log_reg = LogisticRegression(max_iter=3000)

log_reg.fit(X_train_scaled, y_train)
lr_preds = log_reg.predict(X_test_scaled)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_preds))

# 8. Random Forest Model (No scaling needed
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_preds))

# 9. Feature Importance Plot
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances - Random Forest")
plt.bar(range(10), importances[indices][:10])
plt.xticks(range(10), X.columns[indices][:10], rotation=45)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

# 10. Save Best Model
joblib.dump(rf, "models/random_forest_fraud.pkl")

print("\nRandom Forest model saved successfully!")
