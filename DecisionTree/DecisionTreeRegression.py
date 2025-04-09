#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
data_path = "AAPL_preprocessed.csv"
aapl_data = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
print(aapl_data.head())

# Select features and target variable
X = aapl_data[['Open', 'High', 'Low', 'Volume']]
y = aapl_data['Close']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree model
dt_regressor = DecisionTreeRegressor(random_state=42)

# Train model
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred = dt_regressor.predict(X_test)

# Calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 计算误差指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(mse, mae, mape)


# In[ ]:




