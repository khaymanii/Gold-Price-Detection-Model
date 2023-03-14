
# Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data Collection and Processing

gold_data = pd.read_csv('gld_price_data.csv')
gold_data.head()
gold_data.tail()
gold_data.shape
gold_data.info()
gold_data.isnull().sum()
gold_data.describe()


# Data analysis : Positive and Negative correlation

correlation = gold_data.corr()
plt.figure(figsize =(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# Correlation values of GLD

print(correlation['GLD'])

# Checking the distribution of the GLD Price

sns.distplot(gold_data['GLD'],color='green')


# Splitting the features and target variable

X = gold_data.drop(['Date', 'GLD'], axis=1)
y = gold_data['GLD']

print(X)
print(y)


# Splitting into training and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


# Model Training

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,y_train)


# Model Evaluation : Test Data

test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)


# R squared error

error_score = metrics.r2_score(y_test, test_data_prediction)
print('R squared error : ', error_score)


# Compare the actual and predicted values in plot

y_test = list(y_test)


plt.plot(y_test, color='blue', label = 'Actual Values')
plt.plot(test_data_prediction, color='green', label = "Predicted Values")
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

