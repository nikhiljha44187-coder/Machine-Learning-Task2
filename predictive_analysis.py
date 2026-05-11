# Predictive Analysis using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("sales_prediction.csv")

print(df.head())

# Features and Target
X = df[['Advertising', 'Price']]
y = df['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Sample Prediction
sample = [[200, 50]]
predicted_sales = model.predict(sample)

print("Predicted Sales:", predicted_sales[0])
