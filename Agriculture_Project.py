# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv("agriculture.csv")   # place dataset in same folder

# Display Data
print(df.head())

# Data Preprocessing
df = df.dropna()

# Convert categorical to numerical
df = pd.get_dummies(df)

# Features & Target
X = df.drop("Production", axis=1)
y = df["Production"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Crop Production Prediction")
plt.show()
