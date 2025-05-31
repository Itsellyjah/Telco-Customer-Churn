#!/usr/bin/env python3
# Telco Customer Churn - Tableau Data Preparation
# This script prepares data files optimized for Tableau visualizations

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# Create directory for Tableau data files
if not os.path.exists('tableau_data'):
    os.makedirs('tableau_data')
    print("Created 'tableau_data' directory for Tableau files")

# 1. Load the original dataset
print("Loading the original dataset...")
df = pd.read_csv('telco_customer_churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in TotalCharges
if df['TotalCharges'].isnull().sum() > 0:
    mean_total_charges = df['TotalCharges'].mean()
    df['TotalCharges'].fillna(mean_total_charges, inplace=True)

# Create binary Churn variable
df['ChurnBinary'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 2. Create main dataset for Tableau with additional features
print("Creating main dataset for Tableau...")
df_tableau = df.copy()

# Create tenure groups for better visualization
tenure_bins = [0, 12, 24, 36, 48, 60, 72]
tenure_labels = ['0-12 months', '13-24 months', '25-36 months', 
                '37-48 months', '49-60 months', '61-72 months']
df_tableau['TenureGroup'] = pd.cut(df_tableau['tenure'], bins=tenure_bins, labels=tenure_labels)

# Create monthly charges bins
monthly_bins = [0, 30, 50, 70, 90, 120]
monthly_labels = ['$0-30', '$30-50', '$50-70', '$70-90', '$90-120']
df_tableau['MonthlyChargesBin'] = pd.cut(df_tableau['MonthlyCharges'], bins=monthly_bins, labels=monthly_labels)

# Save the main dataset
df_tableau.to_csv('tableau_data/telco_main_data.csv', index=False)
print("Saved main dataset with additional features")

# 3. Create summary datasets for specific visualizations

# 3.1 Churn by categorical variables
print("Creating categorical summaries...")
categorical_vars = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'TenureGroup', 'MonthlyChargesBin'
]

churn_by_category = pd.DataFrame()

for var in categorical_vars:
    # Calculate counts and churn rate by category
    summary = df_tableau.groupby(var).agg(
        Total_Customers=('customerID', 'count'),
        Churned_Customers=('ChurnBinary', 'sum'),
        Churn_Rate=('ChurnBinary', 'mean')
    ).reset_index()
    
    # Add variable name column
    summary['Variable'] = var
    summary.rename(columns={var: 'Category'}, inplace=True)
    
    # Append to the main summary dataframe
    churn_by_category = pd.concat([churn_by_category, summary])

# Save categorical summary
churn_by_category.to_csv('tableau_data/churn_by_category.csv', index=False)
print("Saved categorical summary data")

# 3.2 Service impact analysis
print("Creating service impact analysis...")
service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies']

service_impact = pd.DataFrame()

for service in service_columns:
    # Filter to only include 'Yes' and 'No' (exclude 'No internet service')
    service_data = df_tableau[df_tableau[service].isin(['Yes', 'No'])]
    
    # Calculate churn rate for each service option
    summary = service_data.groupby(service).agg(
        Total_Customers=('customerID', 'count'),
        Churned_Customers=('ChurnBinary', 'sum'),
        Churn_Rate=('ChurnBinary', 'mean')
    ).reset_index()
    
    # Add service name column
    summary['Service'] = service
    summary.rename(columns={service: 'Has_Service'}, inplace=True)
    
    # Append to the main summary dataframe
    service_impact = pd.concat([service_impact, summary])

# Calculate the difference in churn rate (No - Yes)
service_diff = service_impact.pivot(index='Service', columns='Has_Service', values='Churn_Rate')
service_diff['Difference'] = service_diff['No'] - service_diff['Yes']
service_diff.reset_index(inplace=True)

# Save service impact data
service_impact.to_csv('tableau_data/service_impact.csv', index=False)
service_diff.to_csv('tableau_data/service_difference.csv', index=False)
print("Saved service impact analysis data")

# 3.3 Numerical variables analysis
print("Creating numerical variables analysis...")
numerical_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Create summary statistics by churn status
numerical_summary = df_tableau.groupby('Churn')[numerical_vars].agg(['mean', 'median', 'std']).reset_index()
numerical_summary.columns = ['Churn'] + [f'{col}_{stat}' for col in numerical_vars for stat in ['mean', 'median', 'std']]

# Save numerical summary
numerical_summary.to_csv('tableau_data/numerical_summary.csv', index=False)
print("Saved numerical variables summary")

# 3.4 Create churn probability data using logistic regression
print("Creating churn probability data...")

# Prepare data for modeling
# One-hot encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Remove customerID from encoding
categorical_cols.remove('Churn')      # Remove target variable from encoding

# Create a copy for modeling
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

# Drop unnecessary columns
X = df_model.drop(['customerID', 'Churn', 'ChurnBinary'], axis=1)
y = df_model['ChurnBinary']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train a logistic regression model
model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42)
model.fit(X, y)

# Get predicted probabilities
df_model['Churn_Probability'] = model.predict_proba(X)[:, 1]

# Create a dataset with original features and churn probability
churn_prob_data = df.copy()
churn_prob_data['Churn_Probability'] = df_model['Churn_Probability']

# Save churn probability data
churn_prob_data.to_csv('tableau_data/churn_probability.csv', index=False)
print("Saved churn probability data")

# 3.5 Model performance comparison
print("Creating model performance data...")
models = ['Logistic Regression', 'Tuned Logistic Regression', 'Gradient Boosting', 
          'LightGBM', 'Random Forest', 'XGBoost', 'Decision Tree']

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Sample model performance data (replace with actual results if available)
performance_data = pd.DataFrame({
    'Model': models,
    'Accuracy': [0.809, 0.806, 0.798, 0.793, 0.786, 0.784, 0.745],
    'Precision': [0.672, 0.674, 0.655, 0.625, 0.623, 0.608, 0.518],
    'Recall': [0.529, 0.524, 0.495, 0.513, 0.484, 0.492, 0.430],
    'F1 Score': [0.592, 0.590, 0.564, 0.564, 0.545, 0.544, 0.470],
    'ROC AUC': [0.848, 0.849, 0.839, 0.834, 0.824, 0.822, 0.679]
})

# Save model performance data
performance_data.to_csv('tableau_data/model_performance.csv', index=False)

# Create long format for easier visualization
performance_long = pd.melt(performance_data, id_vars=['Model'], value_vars=metrics, 
                         var_name='Metric', value_name='Value')
performance_long.to_csv('tableau_data/model_performance_long.csv', index=False)
print("Saved model performance data")

# 3.6 Feature importance data
print("Creating feature importance data...")
# Sample feature importance data (replace with actual results if available)
feature_importance = pd.DataFrame({
    'Feature': [
        'Contract_Two year', 'InternetService_Fiber optic', 'Contract_One year', 
        'tenure', 'MonthlyToTotalRatio', 'TotalCharges', 'MonthlyCharges',
        'PaymentMethod_Electronic check', 'OnlineSecurity_Yes', 'TechSupport_Yes'
    ],
    'Importance': [0.151, 0.117, 0.071, 0.066, 0.030, 0.015, 0.014, 0.012, 0.011, 0.010],
    'Model': ['Logistic Regression'] * 10
})

# Add data for other models
other_models = [
    ('Random Forest', [
        'TotalCharges', 'MonthlyToTotalRatio', 'MonthlyCharges', 'tenure',
        'InternetService_Fiber optic', 'PaymentMethod_Electronic check',
        'gender_Male', 'PaperlessBilling_Yes', 'Contract_Two year', 'OnlineSecurity_Yes'
    ], [0.149, 0.149, 0.144, 0.115, 0.040, 0.037, 0.025, 0.024, 0.023, 0.022]),
    
    ('Gradient Boosting', [
        'MonthlyToTotalRatio', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check',
        'Contract_Two year', 'Contract_One year', 'MonthlyCharges', 'TotalCharges',
        'tenure', 'PaperlessBilling_Yes', 'InternetService_No'
    ], [0.301, 0.192, 0.124, 0.066, 0.051, 0.050, 0.047, 0.029, 0.024, 0.020])
]

for model_name, features, importances in other_models:
    model_data = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Model': [model_name] * 10
    })
    feature_importance = pd.concat([feature_importance, model_data])

# Save feature importance data
feature_importance.to_csv('tableau_data/feature_importance.csv', index=False)
print("Saved feature importance data")

print("\nData preparation for Tableau completed! All files saved to the 'tableau_data' directory.")
