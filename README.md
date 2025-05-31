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
- `Telco Customer Dashboard Overview.twb`: Tableau workbook with interactive dashboards for churn analysis

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
The project includes a complete Tableau workbook (`Telco Customer Dashboard Overview.twb`) with interactive dashboards for analyzing customer churn. The dashboards provide:

- Executive overview with key churn metrics and trends
- Customer segmentation analysis
- Service impact visualization
- Churn risk assessment with customizable threshold
- Model performance evaluation

The `tableau_data/` directory contains the prepared CSV files that power these dashboards:
- Customer demographics and churn rates
- Service impact analysis
- Churn prediction probabilities
- Model performance metrics

### Using the Tableau Dashboard
1. Open the `.twb` file with Tableau Desktop or Tableau Reader
2. Navigate between dashboard tabs to explore different aspects of the churn analysis
3. Use interactive filters to segment customers and identify high-risk groups
4. Adjust the churn risk threshold parameter to customize retention targeting
