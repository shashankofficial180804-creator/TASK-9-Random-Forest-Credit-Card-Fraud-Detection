# TASK-9-Random-Forest-Credit-Card-Fraud-Detection
Credit card fraud detection using Random Forest on an imbalanced Kaggle dataset. Includes baseline Logistic Regression comparison, stratified sampling, feature importance analysis, and model persistence. Focuses on precision, recall, and F1-score for real-world fraud detection.

# Credit Card Fraud Detection – Random Forest

This project detects fraudulent credit card transactions using Random Forest on an imbalanced dataset. A Logistic Regression model is used as a baseline for comparison. Performance is evaluated using precision, recall, and F1-score instead of accuracy. Feature importance analysis highlights key fraud indicators, and the final model is saved for reuse.

Credit Card Fraud Detection using Random Forest

📌 Overview

Credit card fraud is a major financial risk due to the large number of daily digital transactions. This project focuses on detecting fraudulent credit card transactions using machine learning, with special emphasis on handling highly imbalanced data. A Random Forest ensemble model is used and compared against a Logistic Regression baseline.

🎯 Objective

Detect fraudulent transactions accurately
Handle extreme class imbalance effectively
Compare baseline and ensemble learning models
Evaluate models using appropriate metrics (Precision, Recall, F1-score)

📊 Dataset

Source: Kaggle Credit Card Fraud Dataset
Transactions: 284,807
Fraud Cases: 492 (highly imbalanced)

Features:

V1 to V28 (PCA-transformed features)
Time, Amount
Class (Target: 0 = Non-Fraud, 1 = Fraud)

🛠 Tools & Technologies

Python
Pandas, NumPy
Scikit-learn
Matplotlib
Joblib
GitHub

🧠 Methodology

1. Data Loading & Exploration
Loaded dataset and analyzed class distribution
Identified severe class imbalance
2. Preprocessing
Separated features and target variable
Used stratified train-test split to preserve fraud ratio
3. Baseline Model
Trained Logistic Regression for performance comparison
4. Ensemble Model
Trained Random Forest with 100 decision trees
Leveraged ensemble learning to improve minority class detection
5. Evaluation
Used Precision, Recall, and F1-score
Accuracy was avoided due to class imbalance
6. Feature Importance
Analyzed and visualized top contributing features
7. Model Saving
Saved trained Random Forest model using Joblib

📈 Results

Logistic Regression performed reasonably but struggled with fraud recall
Random Forest significantly improved fraud detection recall and F1-score
Ensemble learning proved more effective for imbalanced datasets

📉 Feature Importance

Random Forest feature importance analysis highlights the most influential transaction features contributing to fraud detection. This helps in understanding model decisions and improving transparency.

📦 Project Structure

credit-card-fraud-detection-rf/
│
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── models/
│   └── random_forest_fraud.pkl
├── outputs/
│   └── feature_importance.png
├── README.md
├── requirements.txt
└── report.pdf

🚀 How to Run

Clone the repository
Install dependencies:
pip install -r requirements.txt
Run the notebook or Python script
View results and saved model

📌 Key Learnings

Importance of choosing correct evaluation metrics
Handling imbalanced datasets effectively
Understanding ensemble learning advantages
Practical fraud detection pipeline implementation

✅ Conclusion

This project demonstrates a complete end-to-end machine learning workflow for fraud detection. The Random Forest model outperforms the baseline and proves suitable for real-world imbalanced classification problems.
