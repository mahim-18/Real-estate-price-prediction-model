# Real Estate Price Prediction using Machine Learning

This project aims to predict real estate prices using various machine learning techniques. The dataset is preprocessed, and models like Linear Regression, Decision Tree, and Random Forest are used to make predictions.

## Steps Involved

### 1. Importing Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

These libraries help in data manipulation, visualization, and splitting the dataset into training and testing sets.

### 2. Loading and Exploring the Dataset
```python
housing = pd.read_csv("data.csv")
housing.head()
housing.describe()
```

- The dataset is loaded using `pandas.read_csv()`.
- The first few rows and a statistical summary of the dataset are displayed.

### 3. Stratified Shuffle Split
```python
from sklearn.model_selection import StratifiedShuffleSplit
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for housing_train_index, housing_test_index in stratSplit.split(housing, housing['CHAS']):
    strat_housing_train = housing.loc[housing_train_index]
    strat_housing_test = housing.loc[housing_test_index]
```

- **Why Stratified Shuffle Split?**
  - Ensures the train and test sets maintain the same proportion of important categorical features (like `CHAS`) as in the original dataset.
  - This reduces sampling bias and improves model generalization.

### 4. Finding Correlations
```python
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
```

- **Why Use a Correlation Matrix?**
  - Helps identify which features are strongly or weakly correlated with the target variable (`MEDV`).
  - Features with high correlation can be useful predictors, while those with low correlation may be less relevant.
  - Pearson’s correlation coefficient ranges between -1 and 1, where:
    - `1` indicates a strong positive correlation.
    - `-1` indicates a strong negative correlation.
    - `0` indicates no correlation.

### 5. Data Visualization
```python
from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize=(20, 15))
housing.plot(kind='scatter', x='RM', y='MEDV')
```

- Scatter plots help visualize feature relationships.

### 6. Handling Missing Values with SimpleImputer
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
```

- **Why SimpleImputer?**
  - Replaces missing values with the median of the respective column.
  - Ensures consistency when handling missing data, preventing issues in model training.

### 7. Feature Scaling and Pipeline Creation
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])
housing_new_tr = my_pipeline.fit_transform(housing)
```

- **Why Feature Scaling?**
  - Ensures all numerical features are on a similar scale, improving model performance.
  - Two common methods:
    - **Min-Max Scaling**: `(value - min) / (max - min)`
    - **Standardization**: `(value - mean) / standard deviation`
- **Why Use a Pipeline?**
  - Automates preprocessing steps like imputing missing values and scaling features.
  - Ensures the same transformations are applied to both training and test data.

### 8. Selecting and Training a Model
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housing_new_tr, housing_labels)
```

- Random Forest Regressor is chosen and trained using the transformed dataset.

### 9. Model Evaluation
```python
from sklearn.metrics import mean_squared_error
predictions = model.predict(housing_new_tr)
mse = mean_squared_error(housing_labels, predictions)
rmse = np.sqrt(mse)
```

- Root Mean Squared Error (RMSE) is calculated to evaluate model performance.

### 10. Cross-Validation for Better Evaluation
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_new_tr, housing_labels, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
```

- **Why Use Cross-Validation?**
  - Prevents overfitting by training the model on different subsets of the dataset.
  - Provides a better estimate of model performance across different data splits.
  - The dataset is split into multiple folds, and the model is evaluated on each fold.

### 11. Saving and Loading the Model using Joblib
```python
from joblib import dump, load
dump(model, 'realestate.joblib')
```

- **Why Use Joblib?**
  - Saves the trained model for later use without retraining.
  - Joblib is optimized for large NumPy arrays, making it more efficient than pickle.
  - The `dump()` function is used to save the model.
  - The `load()` function is used to reload the saved model.

### 12. Testing on Test Data
```python
X_test = strat_housing_test.drop('MEDV', axis=1)
Y_test = strat_housing_test['MEDV'].copy()
X_prepared = my_pipeline.transform(X_test)
predicted_data = model.predict(X_prepared)
rmse_predicted = np.sqrt(mean_squared_error(predicted_data, Y_test))
```

- The model is tested on the test dataset, and RMSE is calculated.

### 13. Loading and Using the Model for Predictions
```python
from joblib import load
import numpy as np

model = load('realestate.joblib')
features = np.array([[-0.44352175,  3.12628155, -1.35893781, -0.27288841, -1.0542567 ,
        0.5009123 , -1.3938808 ,  2.19312325, -0.65766683, -0.78557904,
       -0.69277865,  0.39131918, -0.94116739]])
model.predict(features)
```

- This tests the model by loading it from disk and making predictions on a sample input.

## Conclusion
- The model achieves an RMSE of **2.96** on the test data, indicating decent prediction accuracy.
- Future improvements can include hyperparameter tuning and trying different models.

