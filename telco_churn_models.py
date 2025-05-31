#!/usr/bin/env python3
# Telco Customer Churn - Machine Learning Models
# This script builds and evaluates various ML models for churn prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create a directory for saving model results
import os
if not os.path.exists('model_results'):
    os.makedirs('model_results')
    print("Created 'model_results' directory for saving model outputs")

# 1. DATA LOADING AND PREPROCESSING
print("="*80)
print("LOADING AND PREPROCESSING DATA")
print("="*80)

# Load the preprocessed data from the previous script
print("Loading preprocessed data...")
try:
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    print(f"Loaded preprocessed data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
except FileNotFoundError:
    print("Preprocessed files not found. Loading and preprocessing the original dataset...")
    
    # Load original dataset
    df = pd.read_csv('telco_customer_churn.csv')
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values in TotalCharges
    if df['TotalCharges'].isnull().sum() > 0:
        mean_total_charges = df['TotalCharges'].mean()
        df['TotalCharges'].fillna(mean_total_charges, inplace=True)
    
    # Create binary target variable
    df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop customerID and original Churn column
    df = df.drop(['customerID', 'Churn'], axis=1)
    
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = df.drop('Churn_Binary', axis=1)
    y = df['Churn_Binary']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Processed original data: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

# Print class distribution
print(f"\nClass distribution in training set: {np.bincount(y_train)}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# 2. MODEL EVALUATION FUNCTION
print("\n" + "="*80)
print("SETTING UP MODEL EVALUATION FRAMEWORK")
print("="*80)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model with detailed metrics and visualizations
    """
    # Start timer
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Training Time: {train_time:.2f} seconds")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'model_results/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'model_results/roc_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Create precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.savefig(f'model_results/pr_curve_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Return metrics for comparison
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Training Time': train_time
    }

# 3. BASELINE MODEL: LOGISTIC REGRESSION
print("\n" + "="*80)
print("TRAINING BASELINE MODEL: LOGISTIC REGRESSION")
print("="*80)

# Initialize and evaluate logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg_metrics = evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

# Cross-validation for logistic regression
print("\nPerforming cross-validation for Logistic Regression...")
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC AUC scores: {cv_scores}")
print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 4. DECISION TREE
print("\n" + "="*80)
print("TRAINING DECISION TREE")
print("="*80)

# Initialize and evaluate decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt_metrics = evaluate_model(dt, X_train, X_test, y_train, y_test, "Decision Tree")

# 5. RANDOM FOREST
print("\n" + "="*80)
print("TRAINING RANDOM FOREST")
print("="*80)

# Initialize and evaluate random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_metrics = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# 6. GRADIENT BOOSTING
print("\n" + "="*80)
print("TRAINING GRADIENT BOOSTING")
print("="*80)

# Initialize and evaluate gradient boosting model
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_metrics = evaluate_model(gb, X_train, X_test, y_train, y_test, "Gradient Boosting")

# 7. XGBOOST
print("\n" + "="*80)
print("TRAINING XGBOOST")
print("="*80)

# Initialize and evaluate XGBoost model
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb_metrics = evaluate_model(xgb, X_train, X_test, y_train, y_test, "XGBoost")

# 8. LIGHTGBM
print("\n" + "="*80)
print("TRAINING LIGHTGBM")
print("="*80)

# Initialize and evaluate LightGBM model
lgbm = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_metrics = evaluate_model(lgbm, X_train, X_test, y_train, y_test, "LightGBM")

# 9. MODEL COMPARISON
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# Collect all metrics
all_metrics = [log_reg_metrics, dt_metrics, rf_metrics, gb_metrics, xgb_metrics, lgbm_metrics]
metrics_df = pd.DataFrame(all_metrics)
metrics_df = metrics_df.set_index('Model')

# Print comparison table
print("\nModel Comparison:")
print(metrics_df)

# Plot model comparison
plt.figure(figsize=(14, 8))
metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_results/model_comparison.png')
plt.close()

# 10. HYPERPARAMETER TUNING
print("\n" + "="*80)
print("HYPERPARAMETER TUNING")
print("="*80)

# Identify the best performing model from the initial evaluation
best_model_name = metrics_df['ROC AUC'].idxmax()
print(f"\nBest performing model based on ROC AUC: {best_model_name}")

# Define parameter grids for different models
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 70],
        'subsample': [0.8, 1.0]
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
}

# Select the best model and its parameter grid
if best_model_name in param_grids:
    print(f"Performing hyperparameter tuning for {best_model_name}...")
    
    # Initialize the best model
    if best_model_name == 'Logistic Regression':
        best_model = LogisticRegression(random_state=42)
    elif best_model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'Random Forest':
        best_model = RandomForestClassifier(random_state=42)
    elif best_model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(random_state=42)
    elif best_model_name == 'XGBoost':
        best_model = XGBClassifier(random_state=42)
    elif best_model_name == 'LightGBM':
        best_model = LGBMClassifier(random_state=42)
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grids[best_model_name],
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate the tuned model
    tuned_model = grid_search.best_estimator_
    tuned_metrics = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")
    
    # Compare with the untuned model
    print("\nComparison of untuned vs. tuned model:")
    untuned_dict = metrics_df.loc[best_model_name].to_dict()
    comparison_data = [untuned_dict, tuned_metrics]
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
else:
    print(f"No parameter grid defined for {best_model_name}. Skipping hyperparameter tuning.")

# 11. FEATURE IMPORTANCE ANALYSIS
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Function to plot feature importance
def plot_feature_importance(model, X_train, model_name):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame of features and importances
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'model_results/feature_importance_{model_name.replace(" ", "_").lower()}.png')
        plt.close()
        
        # Return top features
        return feature_importance
    else:
        print(f"Model {model_name} doesn't have feature_importances_ attribute.")
        return None

# Analyze feature importance for tree-based models
tree_models = {
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'XGBoost': xgb,
    'LightGBM': lgbm
}

# Store feature importances from all models
all_importances = {}

for name, model in tree_models.items():
    print(f"\nAnalyzing feature importance for {name}...")
    importance_df = plot_feature_importance(model, X_train, name)
    if importance_df is not None:
        all_importances[name] = importance_df
        print(f"Top 10 important features for {name}:")
        print(importance_df.head(10))

# For logistic regression, analyze coefficients
if hasattr(log_reg, 'coef_'):
    # Get coefficients
    coefficients = log_reg.coef_[0]
    
    # Create DataFrame of features and coefficients
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(20))
    plt.title('Top 20 Feature Coefficients - Logistic Regression')
    plt.tight_layout()
    plt.savefig('model_results/feature_coefficients_logistic_regression.png')
    plt.close()
    
    print("\nTop 10 features by coefficient magnitude for Logistic Regression:")
    print(coef_df[['Feature', 'Coefficient']].head(10))

# 12. CONCLUSION
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Identify the best model after tuning
if 'tuned_metrics' in locals():
    best_model_metrics = tuned_metrics
    best_model_name = f"Tuned {best_model_name}"
else:
    best_model_metrics = metrics_df.loc[metrics_df['ROC AUC'].idxmax()]
    best_model_name = metrics_df['ROC AUC'].idxmax()

print(f"\nBest performing model: {best_model_name}")
print(f"Best model metrics: Accuracy={best_model_metrics['Accuracy']:.4f}, "
      f"Precision={best_model_metrics['Precision']:.4f}, "
      f"Recall={best_model_metrics['Recall']:.4f}, "
      f"F1 Score={best_model_metrics['F1 Score']:.4f}, "
      f"ROC AUC={best_model_metrics['ROC AUC']:.4f}")

# Identify common important features across models
print("\nCommon important features across models:")
common_features = set()

for name, importance_df in all_importances.items():
    top_features = importance_df.head(10)['Feature'].tolist()
    if not common_features:
        common_features = set(top_features)
    else:
        common_features = common_features.intersection(set(top_features))

print(f"Features appearing in top 10 for all tree-based models: {common_features}")

print("\nMachine learning model development completed!")
