# California Housing Price Prediction

## Project Overview

This project focuses on predicting housing prices in California using the California housing dataset. The primary goal is to build predictive models and evaluate their performance using metrics such as R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).

## Steps Taken

### 1. Importing Necessary Libraries
- **Data Handling**: `pandas`, `numpy`
- **Modeling**: `RandomForestRegressor`, `Ridge`, `LinearRegression`, `SVR`
- **Metrics**: `r2_score`, `mean_squared_error`, `mean_absolute_error`
- **Model Tuning and Validation**: `GridSearchCV`, `RandomizedSearchCV`, `train_test_split`, `cross_val_score`
- **Preprocessing**: `StandardScaler`
- **Visualization**: `matplotlib`, `seaborn`

### 2. Dataset Overview
The project utilizes the California housing dataset, accessed via `sklearn.datasets.fetch_california_housing`. This dataset contains information about housing prices and relevant features such as location, population, and income.

### 3. Data Preprocessing
- **Loading the Dataset**: Fetched using Scikit-learn’s API.
- **Feature Inspection**: Evaluated features for relevance and correlation.
- **Scaling Features**: Standardized numerical features using `StandardScaler`.

### 4. Exploratory Data Analysis (EDA)
- **Visualization**: Used `seaborn` and `matplotlib` to visualize distributions and correlations.
- **Correlation Analysis**: Highlighted relationships between features and the target variable.

### 5. Modeling
#### Models Used:
- Linear Regression
- Ridge Regression
- Support Vector Regression (SVR)
- Random Forest Regressor

#### Training and Validation:
- Split data into training and testing sets using `train_test_split`.
- Applied cross-validation for robust performance evaluation.

### 6. Hyperparameter Tuning
- Used `GridSearchCV` and `RandomizedSearchCV` to find the optimal hyperparameters for models.

### 7. Evaluation
#### Metrics:
- R-squared (Coefficient of Determination)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

#### Visualization:
- Plotted predictions versus actual values to assess performance visually.

### 8. Results and Insights
- Reported the best-performing model based on evaluation metrics.
- Provided insights into feature importance and model behavior.

## Tools and Libraries
- **Programming Language**: Python
- **Primary Libraries**: Scikit-learn, pandas, numpy, matplotlib, seaborn

## Future Work
- Explore additional features or external datasets to enhance model accuracy.
- Experiment with other machine learning models, such as gradient boosting frameworks (e.g., XGBoost, LightGBM).
- Deploy the best model for real-world applications.
