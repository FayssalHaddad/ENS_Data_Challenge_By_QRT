# ENS Data Challenge: Electricity Price Analysis

## Overview
This project is part of the ENS Data Challenge by QRT, focusing on explaining the variations in electricity futures prices using a comprehensive data analysis approach.

## Data
- **Training Data**: `X_train_NHkHMNU.csv`, `y_train.csv`
- **Testing Data**: `X_test_final.csv`, `y_test_random_final.csv`
- The dataset includes various features related to electricity prices and their futures contracts.

## Analysis Process
1. **Data Preprocessing**: Handling missing values, removing outliers, and feature scaling.
2. **Exploratory Data Analysis (EDA)**: Distribution analysis of the target variable and correlation checks.
3. **Feature Engineering**: Selection and extraction of significant features for the model.
4. **Model Development and Evaluation**: Implementation of multiple models including Linear Regression, Random Forest, Ridge, Lasso, ElasticNet, and Neural Networks.
5. **Model Comparison**: Spearman correlation calculation to determine the best fitting model.
6. **Feature Importance Analysis**: Identifying the most impactful variables in the Random Forest model.

## Key Libraries
- `statsmodels`
- `numpy`
- `pandas`
- `scipy`
- `sklearn`
- `matplotlib`
- `seaborn`
- `tensorflow`

## How to Run
The project is developed as a Python script, suitable for execution in a Python environment such as Spyder or any other Python IDE. Ensure all required libraries are installed before running the script.

## Conclusion
The study concludes with a comparison of model performances, highlighting the Random Forest method as the most effective for this challenge, and presents the top variables impacting electricity futures prices. You can find detailed results in "ENS_Data_Challenge" .pdf file.

