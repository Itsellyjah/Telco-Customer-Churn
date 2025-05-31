# Telco Customer Churn Analysis and Prediction

## Project Overview
This project analyzes customer churn in a telecommunications company using data analysis and machine learning techniques. The goal is to identify key factors influencing customer churn and develop predictive models to help the company implement targeted retention strategies.

## Dataset
The analysis uses the Telco Customer Churn dataset, which contains information about 7,043 customers with 21 features including demographics, account information, services subscribed, and churn status.

## Project Structure
- `telco_churn_analysis.py`: Initial data loading, preprocessing, and feature engineering
- `telco_churn_eda.py`: Exploratory data analysis with visualizations
- `telco_churn_models.py`: Machine learning model development and evaluation
- `tableau_data_prep.py`: Preparation of data files for Tableau visualizations
- `Telco_Customer_Churn_Report.md`: Comprehensive report of findings and recommendations
- `plots/`: Directory containing EDA visualizations
- `model_results/`: Directory containing model evaluation plots and metrics
- `tableau_data/`: Directory containing prepared data files for Tableau

## Key Findings
- Month-to-month contracts have significantly higher churn rates
- Customers with higher monthly charges tend to churn more frequently
- Security and tech support services are associated with lower churn rates
- Senior citizens and customers paying by electronic check have higher churn rates
- Tenure is a strong predictor of churn, with newer customers more likely to leave

## Models Developed
- Logistic Regression (baseline and tuned)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## Technologies Used
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost, lightgbm
- Tableau (for visualization)

## How to Use
1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the analysis scripts in the following order:
   - `python telco_churn_analysis.py`
   - `python telco_churn_eda.py`
   - `python telco_churn_models.py`
   - `python tableau_data_prep.py`
4. Review the generated report and visualizations

## Tableau Visualizations
The `tableau_data/` directory contains prepared CSV files for creating Tableau dashboards focused on:
- Customer demographics and churn rates
- Service impact analysis
- Churn prediction probabilities
- Model performance metrics
