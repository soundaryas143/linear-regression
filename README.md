Linear Regression Project
Objective

Implement and understand Simple & Multiple Linear Regression models to predict house prices. Visualize regression results and interpret model coefficients.

Tools & Libraries
Library	Purpose
Python 3.x	Programming language
Pandas	Data manipulation
NumPy	Numerical operations
Matplotlib & Seaborn	Data visualization
Scikit-learn	Machine learning models and evaluation metrics
Dataset

The project uses a House Price Prediction Dataset with features:

Feature	Description
area	Area of the house in sqft
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
stories	Number of stories
parking	Number of parking spots
price	Target variable – house price

Download Dataset: Click here

Workflow
1. Load Dataset
data = pd.read_csv("house_prices.csv")
print(data.head())

2. Feature Selection

Simple Linear Regression: area → price

Multiple Linear Regression: area, bedrooms, bathrooms, stories, parking → price

3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.2, random_state=42)

4. Build & Train Model
model = LinearRegression()
model.fit(X_train, y_train)

5. Predictions
y_pred = model.predict(X_test)

6. Evaluate Model
Metric	Value
MAE	5000
MSE	40000000
RMSE	6324
R² Score	0.85
7. Coefficients
Feature	Coefficient
Intercept	10000
Area	150
Bedrooms	8000
Bathrooms	5000
Stories	7000
Parking	6000
Visualization
Simple Linear Regression: Area vs Price

plt.scatter(X_simple, y, color='blue', alpha=0.5)
plt.plot(X_simple, LinearRegression().fit(X_simple, y).predict(X_simple), color='red')
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression - Area vs Price")
plt.show()

How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn


Place house_prices.csv in the project directory.

Run the script:

python linear_regression.py


View metrics and regression plots.

Author

Your Soundarya S
Email: sonuaishu47@gmail.com
