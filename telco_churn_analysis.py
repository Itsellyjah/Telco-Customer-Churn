#!/usr/bin/env python3
# Telco Customer Churn Analysis
# This script performs exploratory data analysis and preprocessing on the Telco Customer Churn dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Data Loading and Exploration
print("="*80)
print("LOADING AND EXPLORING THE DATASET")
print("="*80)

# Load the dataset
df = pd.read_csv('telco_customer_churn.csv')

# Display basic information about the dataset
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts())
    print("-"*40)

# Check for class imbalance (Churn vs. Non-Churn)
print("\nClass Distribution (Churn):")
print(df['Churn'].value_counts())
print(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")

# Check numerical columns statistics
print("\nNumerical Columns Statistics:")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print(df[numerical_cols].describe())

# Check if TotalCharges has non-numeric entries
print("\nChecking TotalCharges for non-numeric entries:")
print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
# Convert TotalCharges to numeric and check for errors
total_charges_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"Number of non-numeric entries in TotalCharges: {total_charges_numeric.isna().sum()}")

# 2. SQL-like Queries using Pandas
print("\n" + "="*80)
print("SQL-LIKE QUERIES USING PANDAS")
print("="*80)

# Contract type distribution
print("\nContract Type Distribution:")
contract_distribution = df.groupby('Contract').size().reset_index(name='Count')
print(contract_distribution)

# Average monthly charges by contract type
print("\nAverage Monthly Charges by Contract Type:")
avg_charges_by_contract = df.groupby('Contract')['MonthlyCharges'].mean().reset_index()
print(avg_charges_by_contract)

# Churn rate by contract type
print("\nChurn Rate by Contract Type:")
churn_by_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean()
).reset_index(name='Churn_Rate')
print(churn_by_contract)

# Using pandas for SQL-like queries
print("\nUsing pandas for SQL-like queries:")
# Equivalent to: SELECT Contract, COUNT(*) as Count FROM df GROUP BY Contract
print("\nContract distribution:")
print(df.groupby('Contract').size().reset_index(name='Count'))

# Equivalent to: SELECT Contract, AVG(MonthlyCharges) as Avg_Monthly_Charges FROM df GROUP BY Contract
print("\nAverage monthly charges by contract:")
print(df.groupby('Contract')['MonthlyCharges'].mean().reset_index(name='Avg_Monthly_Charges'))

# Equivalent to the complex SQL query for churn rate by contract
print("\nChurn statistics by contract:")
churn_stats = df.groupby('Contract').apply(
    lambda x: pd.Series({
        'Churned': (x['Churn'] == 'Yes').sum(),
        'Total': len(x),
        'Churn_Rate': (x['Churn'] == 'Yes').mean()
    })
).reset_index()
print(churn_stats)

# 3. Data Preprocessing
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

# 3.1 Handling Missing Values
print("\nHandling Missing Values:")
# Convert TotalCharges to numeric, coercing errors to NaN
df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
print(f"Missing values after conversion: {df_processed.isnull().sum()}")

# Impute missing TotalCharges with the mean
if df_processed['TotalCharges'].isnull().sum() > 0:
    mean_total_charges = df_processed['TotalCharges'].mean()
    df_processed['TotalCharges'].fillna(mean_total_charges, inplace=True)
    print(f"Imputed missing TotalCharges with mean: {mean_total_charges:.2f}")

# 3.2 Converting Categorical Variables
print("\nConverting Categorical Variables:")

# Convert 'SeniorCitizen' from 0/1 to 'No'/'Yes' for consistency
df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# One-hot encode categorical variables
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Remove customerID from encoding
categorical_cols.remove('Churn')      # Remove target variable from encoding

print(f"Categorical columns to encode: {categorical_cols}")
df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
print(f"Shape after one-hot encoding: {df_encoded.shape}")
print(f"New columns sample: {df_encoded.columns[:10].tolist()}...")

# 3.3 Feature Engineering
print("\nFeature Engineering:")

# Create tenure groups
tenure_bins = [0, 12, 24, 36, 48, 60, 72]
tenure_labels = ['0-12 months', '13-24 months', '25-36 months', 
                '37-48 months', '49-60 months', '61-72 months']
df_encoded['TenureGroup'] = pd.cut(df_encoded['tenure'], bins=tenure_bins, labels=tenure_labels)
print("Created TenureGroup feature")

# Calculate ratio of MonthlyCharges to TotalCharges
# To avoid division by zero, we'll add a small epsilon
epsilon = 1e-10
df_encoded['MonthlyToTotalRatio'] = df_encoded['MonthlyCharges'] / (df_encoded['TotalCharges'] + epsilon)
print("Created MonthlyToTotalRatio feature")

# One-hot encode the new TenureGroup feature
df_encoded = pd.get_dummies(df_encoded, columns=['TenureGroup'], drop_first=True)

# 3.4 Scaling Numerical Features
print("\nScaling Numerical Features:")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyToTotalRatio']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print(f"Scaled numerical features: {numerical_cols}")

# 3.5 Train-Test Split
print("\nTrain-Test Split:")
# Drop customerID as it's not needed for modeling
X = df_encoded.drop(['customerID', 'Churn'], axis=1)
y = (df_encoded['Churn'] == 'Yes').astype(int)  # Convert to binary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Class distribution in training set: {pd.Series(y_train).value_counts(normalize=True)}")
print(f"Class distribution in testing set: {pd.Series(y_test).value_counts(normalize=True)}")

# 4. Save Processed Data
print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)

# Save the processed dataframes
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("Saved processed data to CSV files")

print("\nAnalysis and preprocessing complete!")
