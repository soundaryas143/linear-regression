# linear_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("house_prices.csv")

print("First 5 rows of dataset:")
print(data.head())

print("\nAvailable columns:")
print(data.columns)

# 2. Define Features (X) and Target (y)
# Simple Linear Regression (area → price)
X_simple = data[['area']]
y = data['price']

# Multiple Linear Regression (use more features)
X_multi = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

# 3. Train-Test Split (for multiple regression)
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# 4. Build & Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# 7. Coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# 8. Plot (Simple Linear Regression: area vs price)
plt.scatter(X_simple, y, color='blue', alpha=0.5)
plt.plot(X_simple, LinearRegression().fit(X_simple, y).predict(X_simple), color='red')
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression - Area vs Price")
plt.show()
