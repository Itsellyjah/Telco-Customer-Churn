# Telco Customer Churn Analysis Report

## Executive Summary

This report presents a comprehensive analysis of customer churn for a telecommunications company. Using data from 7,043 customers, we explored patterns of customer attrition, identified key factors influencing churn, and developed predictive models to identify at-risk customers. The overall churn rate is 26.54%, with significant variations across different customer segments.

## 1. Introduction

Customer churn, the loss of clients or customers, is a critical concern for telecommunications companies due to the high cost of acquiring new customers compared to retaining existing ones. This analysis aims to:

1. Identify patterns and factors associated with customer churn
2. Develop accurate predictive models to identify at-risk customers
3. Provide actionable recommendations to reduce churn rates

## 2. Dataset Overview

The dataset contains information about 7,043 customers with 21 attributes including:

- **Demographics**: Gender, senior citizen status, partner and dependent status
- **Account Information**: Tenure, contract type, payment method, paperless billing
- **Services**: Phone, internet, online security, tech support, streaming services
- **Financial**: Monthly charges, total charges
- **Target Variable**: Churn (Yes/No)

## 3. Exploratory Data Analysis Findings

### 3.1 Customer Demographics and Churn

- **Senior Citizens** have a higher churn rate (41.7%) compared to non-seniors (23.6%)
- Customers without **partners** or **dependents** are more likely to churn
- **Gender** has minimal impact on churn rates

### 3.2 Contract and Billing Factors

- **Contract Type** is the strongest predictor of churn:
  - Month-to-month contracts: 42.7% churn rate
  - One-year contracts: 11.3% churn rate
  - Two-year contracts: 2.8% churn rate
- **Payment Method** significantly impacts churn:
  - Electronic check users have the highest churn rate (45.3%)
  - Other payment methods have significantly lower churn rates
- **Paperless Billing** customers have higher churn rates (33.6%) than paper billing customers (16.7%)

### 3.3 Services and Churn

- **Internet Service**:
  - Fiber optic customers have the highest churn rate (41.9%)
  - DSL customers have a moderate churn rate (19.3%)
  - No internet service has the lowest churn rate (7.4%)
- **Security and Support Services** significantly reduce churn:
  - Online Security reduces churn by 27.2 percentage points
  - Tech Support reduces churn by 26.5 percentage points
  - Online Backup reduces churn by 18.4 percentage points
  - Device Protection reduces churn by 16.6 percentage points
- **Entertainment Services** have minimal impact on churn:
  - Streaming Movies reduces churn by 3.7 percentage points
  - Streaming TV reduces churn by 3.5 percentage points

### 3.4 Financial Factors and Tenure

- **Monthly Charges**:
  - Higher monthly charges are generally associated with higher churn
  - Customers paying $70-90 per month have the highest churn rate (37.8%)
  - Customers paying $0-30 per month have the lowest churn rate (9.8%)
- **Tenure**:
  - Strong negative correlation with churn (-0.35)
  - Average tenure for churned customers: 18.0 months
  - Average tenure for non-churned customers: 37.6 months

## 4. Machine Learning Models

We developed and evaluated multiple machine learning models to predict customer churn:

### 4.1 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.9% | 67.2% | 52.9% | 59.2% | 84.8% |
| Tuned Logistic Regression | 80.6% | 67.4% | 52.4% | 59.0% | 84.9% |
| Gradient Boosting | 79.8% | 65.5% | 49.5% | 56.4% | 83.9% |
| LightGBM | 79.3% | 62.5% | 51.3% | 56.4% | 83.4% |
| Random Forest | 78.6% | 62.3% | 48.4% | 54.5% | 82.4% |
| XGBoost | 78.4% | 60.8% | 49.2% | 54.4% | 82.2% |
| Decision Tree | 74.5% | 51.8% | 43.0% | 47.0% | 67.9% |

### 4.2 Feature Importance

The most important features for predicting churn across different models are:

**Consistently Important Features (across all models):**
- Tenure
- Monthly-to-Total Charges Ratio

**Top Features by Model Type:**

**Logistic Regression (coefficient magnitude):**
1. Contract_Two year (-1.51): Strong negative impact on churn
2. InternetService_Fiber optic (1.17): Strong positive impact on churn
3. Contract_One year (-0.71): Negative impact on churn
4. Tenure (-0.66): Negative impact on churn

**Tree-based Models (average importance):**
1. Monthly-to-Total Charges Ratio
2. Total Charges
3. Monthly Charges
4. Internet Service (Fiber optic)
5. Contract Type (Two year)
6. Payment Method (Electronic check)

## 5. Key Insights

1. **Contract Type** is the single most important factor in customer retention. Customers on month-to-month contracts are 15 times more likely to churn than those on two-year contracts.

2. **Fiber Optic Service** has unexpectedly high churn rates despite being a premium service, suggesting potential issues with service quality, pricing, or competition.

3. **Security and Support Services** significantly reduce churn, while entertainment services have minimal impact.

4. **Customer Tenure** is strongly associated with loyalty. The first year is the most critical period for customer retention.

5. **Electronic Check Payment** is associated with significantly higher churn rates compared to other payment methods.

6. **Price Sensitivity** exists but varies by segment. Higher monthly charges generally correlate with higher churn, but the relationship isn't perfectly linear.

## 6. Recommendations

### 6.1 Strategic Recommendations

1. **Incentivize Longer Contracts**
   - Offer significant discounts or benefits for customers willing to commit to one or two-year contracts
   - Develop special promotions to convert month-to-month customers to longer-term contracts

2. **Improve Fiber Optic Service Experience**
   - Investigate service quality issues in the fiber optic segment
   - Consider competitive pricing adjustments if necessary
   - Enhance customer support specifically for fiber optic customers

3. **Promote Security and Support Services**
   - Bundle security and support services at a discount
   - Highlight the value of these services in marketing materials
   - Offer free trials of these services to high-risk customers

4. **Enhance Early Customer Experience**
   - Implement special onboarding and support programs for customers in their first year
   - Increase touchpoints and satisfaction surveys during the critical first 12 months
   - Develop an early warning system to identify at-risk new customers

5. **Address Electronic Payment Issues**
   - Investigate why electronic check customers churn at higher rates
   - Offer incentives for automatic payments through other methods
   - Improve the electronic payment experience

### 6.2 Implementation Plan

1. **Customer Segmentation**
   - Implement the tuned Logistic Regression model to score all customers by churn risk
   - Create targeted retention strategies for each risk segment

2. **Proactive Retention Program**
   - Develop specific interventions for high-risk customers
   - Create a dashboard for customer service to identify at-risk customers during interactions

3. **Service Improvements**
   - Address quality issues in fiber optic service
   - Enhance the value proposition of security and support services

4. **Contract Strategy**
   - Revise contract offerings and incentives to encourage longer commitments
   - Develop a conversion strategy for month-to-month customers

5. **Continuous Monitoring**
   - Implement ongoing monitoring of churn rates by segment
   - Regularly update predictive models with new data

## 7. Conclusion

This analysis provides a clear roadmap for reducing customer churn through targeted interventions. By focusing on contract types, service quality, and early customer experience, the company can significantly improve retention rates. The predictive models developed can effectively identify at-risk customers, allowing for proactive retention efforts.

The potential business impact is substantial: even a 5% reduction in the churn rate could result in significant revenue retention and reduced customer acquisition costs.

## Appendix: Tableau Visualization Guide

The accompanying Tableau dashboard provides interactive visualizations of:

1. Churn rates by customer segment
2. Impact of services on churn
3. Contract type analysis
4. Tenure patterns and churn probability
5. Model performance metrics
6. Feature importance visualization

The dashboard is designed to be used by business stakeholders to understand churn patterns and by customer service representatives to identify at-risk customers.
