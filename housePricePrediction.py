

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

"""importing the boston house price dataset"""

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

"""The California Housing dataset has been loaded. Let's inspect its features and target."""

print(housing.DESCR)
print(housing.feature_names)
print(housing.target_names)

# loading ds to pandas df
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['target'] = housing.target
housing_df.head()

housing_df.shape

housing_df.isnull().sum()

housing_df.describe()

correlation = housing_df.corr()
## constructing heatmap to understnad the correlation
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

"""Splitting the data and target"""

X=housing_df.drop(['target'], axis=1)
 Y=housing_df['target']

print(X)
print(Y)

"""Splitting the data into Training and test data"""

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
print(X.shape,X_train.shape,X_test.shape)

"""Model Training

XGBoost Regressor
"""

model = XGBRegressor()

model.fit(X_train,Y_train)

"""Evaluation

prediction on training data
"""



## accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

## R squared error
score_1=metrics.r2_score(Y_train,training_data_prediction)
print("R squared error : ",score_1)
## Mean absolute error
score_2=metrics.mean_absolute_error(Y_train,training_data_prediction)
print("Mean absolute error : ",score_2)

"""Visualization the actual prices and predicted prices

Prediction on Test Data
"""

plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()



test_data_prediction = model.predict(X_test)
print(test_data_prediction)

## R squared error
score_1=metrics.r2_score(Y_test,test_data_prediction)
print("R squared error : ",score_1)

# Mean Absolute error
score_2=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("Mean absolute error : ",score_2)

