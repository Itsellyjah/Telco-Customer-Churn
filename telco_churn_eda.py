#!/usr/bin/env python3
# Telco Customer Churn - Exploratory Data Analysis
# This script performs detailed EDA on the Telco Customer Churn dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('telco_customer_churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values in TotalCharges
if df['TotalCharges'].isnull().sum() > 0:
    mean_total_charges = df['TotalCharges'].mean()
    df['TotalCharges'].fillna(mean_total_charges, inplace=True)
    print(f"Filled {df['TotalCharges'].isnull().sum()} missing values in TotalCharges")

# Convert SeniorCitizen from 0/1 to No/Yes for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# Create a binary Churn variable for easier analysis
df['ChurnBinary'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Create a directory for saving plots
import os
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots' directory for saving visualizations")

# 1. DISTRIBUTION OF NUMERICAL VARIABLES
print("\n1. Analyzing distribution of numerical variables...")

# Tenure distribution by churn status
plt.figure(figsize=(14, 7))
sns.histplot(data=df, x='tenure', hue='Churn', bins=20, kde=True, element='step')
plt.title('Distribution of Tenure by Churn Status')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.savefig('plots/tenure_distribution.png')
plt.close()

# Monthly Charges distribution by churn status
plt.figure(figsize=(14, 7))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=20, kde=True, element='step')
plt.title('Distribution of Monthly Charges by Churn Status')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Count')
plt.savefig('plots/monthly_charges_distribution.png')
plt.close()

# Total Charges distribution by churn status
plt.figure(figsize=(14, 7))
sns.histplot(data=df, x='TotalCharges', hue='Churn', bins=20, kde=True, element='step')
plt.title('Distribution of Total Charges by Churn Status')
plt.xlabel('Total Charges ($)')
plt.ylabel('Count')
plt.savefig('plots/total_charges_distribution.png')
plt.close()

# Box plots for numerical variables by churn status
plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure by Churn Status')

plt.subplot(1, 3, 2)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn Status')

plt.subplot(1, 3, 3)
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('Total Charges by Churn Status')

plt.tight_layout()
plt.savefig('plots/numerical_boxplots.png')
plt.close()

# 2. CORRELATION ANALYSIS
print("\n2. Performing correlation analysis...")

# Create a correlation dataframe with numerical variables
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnBinary']]
correlation = numeric_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
plt.title('Correlation Heatmap of Numerical Variables')
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 3. CATEGORICAL VARIABLES ANALYSIS
print("\n3. Analyzing categorical variables...")

# Function to create and save churn rate plots for categorical variables
def plot_churn_rate_by_category(column, title=None, figsize=(12, 6)):
    if title is None:
        title = f'Churn Rate by {column}'
    
    # Calculate churn rate by category
    churn_rate = df.groupby(column)['ChurnBinary'].mean().sort_values(ascending=False)
    count = df.groupby(column).size()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Churn rate by category
    churn_rate.plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title(f'Churn Rate by {column}')
    ax1.set_ylabel('Churn Rate')
    ax1.set_ylim(0, 1)
    
    # Plot 2: Count by category and churn status
    sns.countplot(x=column, hue='Churn', data=df, ax=ax2)
    ax2.set_title(f'Count by {column} and Churn Status')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'plots/churn_by_{column.lower()}.png')
    plt.close()

# Analyze key categorical variables
categorical_vars = [
    'Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines'
]

for var in categorical_vars:
    plot_churn_rate_by_category(var)

# 4. KEY BUSINESS QUESTIONS
print("\n4. Addressing key business questions...")

# Q1: Do short-term contract customers churn more than long-term ones?
print("\nQ1: Do short-term contract customers churn more than long-term ones?")
contract_churn = df.groupby('Contract')['ChurnBinary'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
contract_churn.columns = ['Churn Rate', 'Count']
print(contract_churn)

# Q2: Are high monthly charges associated with higher churn?
print("\nQ2: Are high monthly charges associated with higher churn?")
# Create bins for monthly charges
bins = [0, 30, 50, 70, 90, 120]
labels = ['$0-30', '$30-50', '$50-70', '$70-90', '$90-120']
df['MonthlyChargesBin'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels)

monthly_charge_churn = df.groupby('MonthlyChargesBin')['ChurnBinary'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
monthly_charge_churn.columns = ['Churn Rate', 'Count']
print(monthly_charge_churn)

# Visualize monthly charges bins vs churn rate
plt.figure(figsize=(12, 6))
sns.barplot(x='MonthlyChargesBin', y='ChurnBinary', data=df, estimator=np.mean, ci=None, palette='viridis')
plt.title('Churn Rate by Monthly Charges Range')
plt.xlabel('Monthly Charges Range')
plt.ylabel('Churn Rate')
plt.savefig('plots/churn_by_monthly_charges_bin.png')
plt.close()

# Q3: How do add-on services impact churn?
print("\nQ3: How do add-on services impact churn?")
# Create a summary table of churn rates for different services
service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
service_impact = pd.DataFrame()

for service in service_columns:
    # Calculate churn rate for each service option
    service_churn = df.groupby(service)['ChurnBinary'].agg(['mean', 'count'])
    service_churn.columns = [f'{service} Churn Rate', f'{service} Count']
    
    # Only keep Yes and No options (exclude 'No internet service')
    if 'No internet service' in service_churn.index:
        service_churn = service_churn.drop('No internet service')
    
    # Calculate difference in churn rate (No - Yes)
    diff = service_churn.loc['No', f'{service} Churn Rate'] - service_churn.loc['Yes', f'{service} Churn Rate']
    service_impact.loc[service, 'Churn Rate (No)'] = service_churn.loc['No', f'{service} Churn Rate']
    service_impact.loc[service, 'Churn Rate (Yes)'] = service_churn.loc['Yes', f'{service} Churn Rate']
    service_impact.loc[service, 'Difference'] = diff
    service_impact.loc[service, 'Impact'] = 'Reduces Churn' if diff > 0 else 'Increases Churn'

print(service_impact.sort_values(by='Difference', ascending=False))

# Visualize the impact of services on churn
plt.figure(figsize=(14, 8))
service_impact_sorted = service_impact.sort_values(by='Difference', ascending=False)
x = np.arange(len(service_impact_sorted.index))
width = 0.35

plt.bar(x - width/2, service_impact_sorted['Churn Rate (No)'], width, label='Without Service')
plt.bar(x + width/2, service_impact_sorted['Churn Rate (Yes)'], width, label='With Service')

plt.xlabel('Service')
plt.ylabel('Churn Rate')
plt.title('Impact of Services on Churn Rate')
plt.xticks(x, service_impact_sorted.index, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('plots/service_impact_on_churn.png')
plt.close()

# 5. MULTIVARIATE ANALYSIS
print("\n5. Performing multivariate analysis...")

# Contract type and tenure vs churn
plt.figure(figsize=(14, 8))
sns.boxplot(x='Contract', y='tenure', hue='Churn', data=df, palette='Set2')
plt.title('Tenure by Contract Type and Churn Status')
plt.savefig('plots/tenure_contract_churn.png')
plt.close()

# Contract type and monthly charges vs churn
plt.figure(figsize=(14, 8))
sns.boxplot(x='Contract', y='MonthlyCharges', hue='Churn', data=df, palette='Set2')
plt.title('Monthly Charges by Contract Type and Churn Status')
plt.savefig('plots/monthly_charges_contract_churn.png')
plt.close()

# Create a scatter plot of tenure vs monthly charges colored by churn
plt.figure(figsize=(12, 8))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, alpha=0.7)
plt.title('Tenure vs Monthly Charges by Churn Status')
plt.savefig('plots/tenure_vs_monthly_charges.png')
plt.close()

# 6. ADVANCED VISUALIZATIONS
print("\n6. Creating advanced visualizations...")

# Create a pair plot for numerical variables
plt.figure(figsize=(16, 12))
sns.pairplot(df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']], hue='Churn')
plt.savefig('plots/pair_plot.png')
plt.close()

# Create a violin plot for monthly charges by internet service and churn
plt.figure(figsize=(14, 8))
sns.violinplot(x='InternetService', y='MonthlyCharges', hue='Churn', data=df, split=True, inner='quart')
plt.title('Monthly Charges by Internet Service and Churn Status')
plt.savefig('plots/monthly_charges_internet_churn.png')
plt.close()

# 7. SUMMARY STATISTICS
print("\n7. Generating summary statistics...")

# Overall churn rate
overall_churn_rate = df['ChurnBinary'].mean()
print(f"\nOverall Churn Rate: {overall_churn_rate:.2%}")

# Top 5 factors with highest churn rate
categorical_vars_no_id = [col for col in categorical_vars if col != 'customerID']
churn_rates = []

for var in categorical_vars_no_id:
    for category in df[var].unique():
        subset = df[df[var] == category]
        if len(subset) > 100:  # Only consider categories with sufficient data
            churn_rate = subset['ChurnBinary'].mean()
            churn_rates.append((var, category, churn_rate, len(subset)))

churn_rate_df = pd.DataFrame(churn_rates, columns=['Variable', 'Category', 'Churn Rate', 'Count'])
print("\nTop 5 Factors with Highest Churn Rate:")
print(churn_rate_df.sort_values('Churn Rate', ascending=False).head(5))

# Top 5 factors with lowest churn rate
print("\nTop 5 Factors with Lowest Churn Rate:")
print(churn_rate_df.sort_values('Churn Rate').head(5))

# Average Monthly Charges for churned vs non-churned customers
avg_charges = df.groupby('Churn')['MonthlyCharges'].mean()
print("\nAverage Monthly Charges:")
print(avg_charges)

# Average Tenure for churned vs non-churned customers
avg_tenure = df.groupby('Churn')['tenure'].mean()
print("\nAverage Tenure (months):")
print(avg_tenure)

print("\nEDA completed! All visualizations saved to the 'plots' directory.")
