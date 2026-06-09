# Auto Insurance Fraud Detection Dashboard

## Overview

This project demonstrates an end-to-end machine learning solution for detecting potentially fraudulent automobile insurance claims.

Using a synthetic dataset of 15,000 insurance claims, the application applies feature selection techniques, trains multiple machine learning models, and provides an interactive Streamlit dashboard for fraud analytics and risk scoring.

The project was developed as a portfolio demonstration of data science, machine learning, analytics, and dashboard development skills.

---

## Business Problem

Insurance fraud costs insurers billions of dollars annually through exaggerated, fabricated, or organized fraudulent claims.

The objective of this project is to:

* Identify patterns associated with fraudulent claims
* Compare multiple machine learning models
* Provide an investigator-friendly dashboard
* Support risk-based claim review processes

---

## Dataset

A synthetic dataset was generated containing 15,000 automobile insurance claims.

### Features

* Age
* Gender
* Province
* Postal Code
* Years Licensed
* Employment Status
* Vehicle Brand
* Vehicle Age
* Vehicle Use
* Vehicle Value
* Claim Type
* Claim Amount
* Previous Claims
* Police Report Filed
* Witness Present
* Policy Tenure
* Claim-to-Vehicle-Value Ratio

### Target Variable

* Fraud Flag (0 = Non-Fraud, 1 = Fraud)

---

## Feature Selection

Three regularization techniques were used to identify important predictors:

### LASSO (L1 Regularization)

Used to eliminate less important features and perform variable selection.

### Ridge (L2 Regularization)

Used to evaluate coefficient stability and reduce overfitting.

### Elastic Net

Combines LASSO and Ridge penalties to balance feature selection and coefficient shrinkage.

A consolidated ranking table was created to compare feature importance across all three methods.

---

## Machine Learning Models

The following models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The best-performing model is automatically saved and used by the dashboard risk scoring tool.

---

## Dashboard Features

### Executive Dashboard

Provides:

* Total Claims
* Fraud Cases
* Fraud Rate
* Average Claim Amount

### Fraud Analytics

Interactive visualizations showing:

* Fraud Rate by Province
* Fraud Rate by Claim Type
* Fraud Rate by Vehicle Brand
* Fraud Rate by Gender

### Feature Selection

Displays:

* LASSO Rankings
* Ridge Rankings
* Elastic Net Rankings
* Average Feature Importance Rankings

### Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Claim Risk Scoring Tool

Allows users to enter claim information and receive:

* Fraud Probability
* Predicted Class
* Risk Level

### Confusion Matrices

Visual comparison of:

* Logistic Regression
* Random Forest
* Gradient Boosting

---

## Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Dashboard

* Streamlit

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/auto-insurance-fraud-detection.git

cd auto-insurance-fraud-detection

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

---

## Future Enhancements

Potential future improvements include:

* SHAP Explainability
* XGBoost Model
* Fraud Investigation Recommendations
* Scenario Analysis
* Real-Time Claim Scoring API
* Geospatial Fraud Mapping
* Explainable AI Dashboard

---

## Author

Feda Bashbishi

University of Waterloo – Master of Data Science and Artificial Intelligence (MDSAI)

Email: [fbashbis@uwaterloo.ca](mailto:fbashbis@uwaterloo.ca)
